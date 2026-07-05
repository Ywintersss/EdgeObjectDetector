"""
train.py — Train the EdgeObjectDetector YOLO model on the RPC dataset.

Usage:
    python train.py --smoke          # fast end-to-end pipeline test (few images, 2 epochs)
    python train.py                  # full training run (uses GPU if available)
    python train.py --epochs 100 --imgsz 640 --batch 16

The RPC dataset has 200 retail-product classes. Labels are pre-converted to
YOLO format under dataset/labels/{train,val}; see preprocessing.py.
"""

import argparse
import sys
from pathlib import Path

# Ultralytics is imported lazily inside main() so that --help works even if the
# environment is not fully set up, and so import errors are reported clearly.

# Resolve paths relative to this file so the script works from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_data_path(data: str) -> Path:
    """Resolve the --data yaml to an absolute path (relative names are project-rooted)."""
    p = Path(data)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments controlling the training run."""
    parser = argparse.ArgumentParser(description="Train EdgeObjectDetector (YOLO11).")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base model/weights (default: yolo11n.pt — nano, edge-friendly).")
    parser.add_argument("--data", default="dataset_640.yml",
                        help="Dataset yaml. Defaults to the 640px-resized set (much faster). "
                             "Use 'dataset.yml' for the full-resolution originals.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs (see --patience).")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Training image size (px). 512 ~36%% less GPU work than 640.")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (-1 = auto). 32 keeps the GPU busy for nano@512 on 8GB.")
    parser.add_argument("--cache", default="False",
                        help="Image cache: 'False' (default). Source imgs are 1751px -> a full "
                             "'disk'/'ram' cache needs ~873GB and will NOT fit; keep off.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stop after N epochs with no val improvement.")
    parser.add_argument("--device", default=None,
                        help="Device: 'cpu', '0', '0,1', etc. Default: auto-detect GPU.")
    parser.add_argument("--name", default="rpc_yolo11n", help="Run name under runs/detect/.")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast pipeline test: tiny data fraction, 2 epochs, small imgsz.")
    return parser.parse_args()


def resolve_device(requested: str | None):
    """Pick the training device.

    Returns the user's explicit choice, otherwise GPU 0 when CUDA is available,
    otherwise CPU. Kept isolated so device logic is easy to test/change.
    """
    if requested is not None:
        return requested
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        # If torch is missing, let Ultralytics surface the real error later.
        return "cpu"


def build_train_kwargs(args: argparse.Namespace) -> dict:
    """Assemble the keyword arguments passed to model.train().

    In --smoke mode we deliberately shrink everything so the run finishes in a
    couple of minutes on CPU and only exercises the pipeline, not accuracy.
    """
    device = resolve_device(args.device)

    if args.smoke:
        return dict(
            data=str(resolve_data_path(args.data)),
            epochs=2,
            imgsz=320,          # small images -> fast forward/backward passes
            batch=4,
            fraction=0.01,      # ~1% of the training set is enough to prove wiring
            device=device,
            name=f"{args.name}_smoke",
            workers=2,
            plots=True,
            verbose=True,
        )

    # Allow --cache False to fully disable caching; otherwise pass the string through.
    cache = False if str(args.cache).lower() in ("false", "none", "0") else args.cache

    return dict(
        data=str(resolve_data_path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cache=cache,          # 'disk' avoids re-decoding 53k JPEGs every epoch
        patience=args.patience,  # early stop once val mAP plateaus
        device=device,
        name=args.name,
        plots=True,
        verbose=True,
    )


def main() -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    args = parse_args()

    # Fail fast with a clear message if the dataset config is missing.
    data_path = resolve_data_path(args.data)
    if not data_path.exists():
        print(f"ERROR: dataset config not found at {data_path}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ERROR: ultralytics not installed ({exc}). Run: pip install ultralytics",
              file=sys.stderr)
        return 1

    kwargs = build_train_kwargs(args)
    print(f"Starting {'SMOKE' if args.smoke else 'FULL'} training run on device="
          f"{kwargs['device']} with model={args.model}")

    try:
        model = YOLO(args.model)          # downloads pretrained weights on first use
        results = model.train(**kwargs)   # runs training; raises on hard failures
    except Exception as exc:  # noqa: BLE001 — surface any training failure to the caller
        print(f"ERROR: training failed: {exc}", file=sys.stderr)
        return 1

    # Report where the trained weights landed so the user can find them.
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"\nDone. Best weights: {Path(save_dir) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
