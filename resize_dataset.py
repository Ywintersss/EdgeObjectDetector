"""
resize_dataset.py — One-time offline downscale of the RPC images for faster training.

The source images are ~1751x1751, so decoding+resizing them every epoch dominates
training time. This script rewrites them at a smaller max side (default 640) into a
new `dataset_640/` tree and copies the labels across.

YOLO labels are NORMALIZED (center-x/y/w/h as fractions of image size), so they are
resolution-independent and require NO modification when the image is resized.

Usage:
    python resize_dataset.py --verify --limit 8   # trial: 8 imgs/split + drawn-box check
    python resize_dataset.py                       # full run (train, val, test)
    python resize_dataset.py --size 512 --dst dataset_512
"""

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resize_one(task: tuple[str, str, int, int]) -> tuple[str, str]:
    """Resize a single image, preserving aspect ratio, only downscaling.

    Runs in a worker process. Returns (status, message) where status is
    'ok' | 'copied' | 'error' so the parent can tally results without crashing
    the whole batch on one bad file.
    """
    src, dst, size, quality = task
    try:
        img = cv2.imread(src)
        if img is None:
            return ("error", f"unreadable: {src}")

        h, w = img.shape[:2]
        scale = size / max(h, w)
        if scale < 1.0:  # only shrink; never upscale a smaller image
            new_w, new_h = round(w * scale), round(h * scale)
            # INTER_AREA is the correct interpolation for downscaling (avoids moire).
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok = cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return ("error", f"write failed: {dst}")
        return ("ok", dst)
    except Exception as exc:  # noqa: BLE001 — never let one file abort the pool
        return ("error", f"{src}: {exc}")


def build_tasks(src_dir: Path, dst_dir: Path, size: int, quality: int,
                limit: int | None) -> list[tuple]:
    """Enumerate (src, dst, size, quality) resize jobs for one image split."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        files = files[:limit]
    # Keep the same filename; always write .jpg so the labels' basenames still match.
    return [(str(f), str(dst_dir / (f.stem + ".jpg")), size, quality) for f in files]


def resize_split(name: str, src_dir: Path, dst_dir: Path, size: int, quality: int,
                 workers: int, limit: int | None) -> int:
    """Resize every image in one split using a process pool. Returns error count."""
    tasks = build_tasks(src_dir, dst_dir, size, quality, limit)
    if not tasks:
        print(f"  [{name}] no images found in {src_dir}")
        return 0

    ok = errors = 0
    print(f"  [{name}] resizing {len(tasks)} images -> {dst_dir} ...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(resize_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            status, msg = fut.result()
            if status == "error":
                errors += 1
                print(f"    WARN {msg}", file=sys.stderr)
            else:
                ok += 1
            if i % 5000 == 0:
                print(f"    [{name}] {i}/{len(tasks)} done")
    print(f"  [{name}] finished: {ok} ok, {errors} errors")
    return errors


def copy_labels(src_labels: Path, dst_labels: Path, limit: int | None) -> None:
    """Copy label .txt files unchanged (normalized coords need no resize)."""
    if not src_labels.exists():
        print(f"  [labels] none at {src_labels} (skipping)")
        return
    dst_labels.mkdir(parents=True, exist_ok=True)
    files = sorted(src_labels.glob("*.txt"))
    if limit is not None:
        files = files[:limit]
    for f in files:
        shutil.copy2(f, dst_labels / f.name)
    print(f"  [labels] copied {len(files)} -> {dst_labels}")


def verify_boxes(dst_images: Path, dst_labels: Path, out_dir: Path, count: int) -> None:
    """Draw denormalized boxes on a few resized images so boxes can be eyeballed.

    Proves the resize preserved geometry: normalized boxes should still tightly
    frame the objects on the smaller image.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = 0
    for img_path in sorted(dst_images.glob("*.jpg")):
        label_path = dst_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = (float(p) for p in parts)
            # Denormalize YOLO box back to pixel corners on the resized image.
            x1 = int((xc - bw / 2) * W); y1 = int((yc - bh / 2) * H)
            x2 = int((xc + bw / 2) * W); y2 = int((yc + bh / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out = out_dir / f"verify_{img_path.name}"
        cv2.imwrite(str(out), img)
        drawn += 1
        if drawn >= count:
            break
    print(f"  [verify] wrote {drawn} annotated images -> {out_dir}")


def write_data_yaml(dst_root: Path, size: int) -> None:
    """Emit a dataset yaml pointing at the resized tree, reusing the 200 class names."""
    # Read source with utf-8 (ignore any stray bytes) and keep only the names block.
    src_yaml = PROJECT_ROOT / "dataset.yml"
    names_block = []
    capture = False
    for line in src_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("names:"):
            capture = True
        if capture:
            names_block.append(line)

    out_yaml = PROJECT_ROOT / f"dataset_{size}.yml"
    header = [
        f"# dataset_{size}.yaml - resized copy of the RPC dataset ({size}px max side)",
        f"path: {dst_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "nc: 200",
    ]
    # Explicit utf-8 so the file is valid regardless of the OS default encoding.
    out_yaml.write_text("\n".join(header + names_block) + "\n", encoding="utf-8")
    print(f"  [yaml] wrote {out_yaml}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline-resize the RPC dataset for faster training.")
    p.add_argument("--size", type=int, default=640, help="Max side in pixels (default 640).")
    p.add_argument("--dst", default=None, help="Destination root (default dataset_<size>).")
    p.add_argument("--quality", type=int, default=90, help="Output JPEG quality (default 90).")
    p.add_argument("--workers", type=int, default=0, help="Worker processes (0 = auto).")
    p.add_argument("--limit", type=int, default=None, help="Max images per split (for trials).")
    p.add_argument("--verify", action="store_true", help="Also render drawn-box previews.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import os
    workers = args.workers or max(1, (os.cpu_count() or 4))
    dst_root = Path(args.dst) if args.dst else PROJECT_ROOT / f"dataset_{args.size}"

    if not SRC_ROOT.exists():
        print(f"ERROR: source dataset not found at {SRC_ROOT}", file=sys.stderr)
        return 1

    print(f"Resizing to {args.size}px -> {dst_root}  (workers={workers}, "
          f"limit={args.limit or 'all'})")

    total_errors = 0
    # test has images but no labels; process it too for later evaluation.
    for split in ("train", "val", "test"):
        src_img = SRC_ROOT / "images" / split
        if not src_img.exists():
            continue
        total_errors += resize_split(split, src_img, dst_root / "images" / split,
                                     args.size, args.quality, workers, args.limit)
        copy_labels(SRC_ROOT / "labels" / split, dst_root / "labels" / split, args.limit)

    write_data_yaml(dst_root, args.size)

    if args.verify:
        verify_boxes(dst_root / "images" / "val", dst_root / "labels" / "val",
                     dst_root / "verify_preview", count=min(8, args.limit or 8))

    if total_errors:
        print(f"\nCompleted with {total_errors} errors (see warnings above).", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
