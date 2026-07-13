"""
export_int8.py — Export the trained coarse-17 YOLO11n to INT8 TFLite at several
input sizes, measure the accuracy cost of each, and assemble a laptop deploy bundle.

Usage:
    python export_int8.py --sizes 320                  # validate the toolchain first (fast)
    python export_int8.py --sizes 640,448,320          # full sweep -> export/report.md
    python export_int8.py --bundle 320                 # assemble deploy/ for the laptop

Requires Linux x86 + edgetpu_compiler (WSL2 is fine); check with `edgetpu_compiler --version`.

INT8 calibration reads REAL cluttered images (dataset_real_blend.yml). Calibrating on
the wrong distribution silently costs accuracy that looks like "quantization is lossy".

WHY format="edgetpu" IS MANDATORY, not merely one option among several:
Ultralytics applies its INT8 box-normalization mitigation (tf_wrapper/_tf_decode_boxes,
which scales boxes to 0..1) under `if fmt == "edgetpu"` and NOWHERE else. Export via
"tflite" or "saved_model" and the boxes stay in PIXEL units (0..425). The detect head
concatenates boxes and class scores into ONE output tensor, so per-tensor INT8
quantization must pick a single scale for both -- and pixel-scale boxes drag that scale
to ~1.77, coarser than the entire 0..1 probability range. Every class score then
quantizes to {0, 1.77}, nothing clears the confidence threshold, and the model scores
mAP 0.000 on every split while looking perfectly healthy: right shape, right dtypes,
right file size, 4 ms latency. Measured, not theorized.

Two artifact-level guards run on every export, because a green flag proves nothing:
  verify_full_integer()    - graph I/O really is int8/uint8 (the Edge TPU accepts no other)
  verify_score_resolution() - the output scale can still REPRESENT a confidence
The second exists because the first passed the mAP-0.000 model happily.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

# FP32 baseline measured on the held-out splits (see the design spec).
BASELINE = {"cluttered_map50": 0.995, "cluttered_map": 0.879,
            "single_map50": 0.995, "single_map": 0.945}

EXPECTED_CLASSES = 17

# Coarsest usable quantization step for the output tensor, which carries class
# probabilities in 0..1. At 0.05 a confidence still has ~20 levels — plenty to clear a
# 0.25 threshold. A correct (box-normalized) export measures ~0.0039 (1/255, ~256
# levels); the broken pixel-box export measured 1.77, which cannot represent ANY
# probability. The gap between right and wrong is ~450x, so this bound is not delicate.
MAX_OUTPUT_SCALE = 0.05


class FormatLevelExportError(ValueError):
    """An export failure caused by the FORMAT, so it recurs identically at every size.

    `_run_sweep` aborts on these instead of re-running a ~10-minute export per remaining
    size to relearn the same fact. Size-specific failures stay ordinary exceptions and
    let the sweep continue.

    A ValueError subclass so existing `except ValueError` handling keeps working.
    """


class NotFullyIntegerError(FormatLevelExportError):
    """Raised when an exported artifact's graph I/O is not fully integer."""


class ScoreResolutionError(FormatLevelExportError):
    """Raised when the output tensor's INT8 scale is too coarse to encode a confidence.

    The failure this catches is silent: the artifact is a valid, fully-integer,
    correctly-shaped TFLite model that detects NOTHING, because every class score has
    been quantized to a two-value set. It cost a full export plus two val passes to
    surface as mAP 0.000; this guard catches it from the artifact alone, in seconds.
    """


def parse_sizes(text: str) -> list[int]:
    """Parse '640,448,320' -> [640, 448, 320]. Raises ValueError on anything not a positive int."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"no sizes parsed from {text!r}")
    sizes = []
    for p in parts:
        try:
            n = int(p)
        except ValueError as exc:
            raise ValueError(f"invalid size {p!r} in {text!r}") from exc
        if n <= 0:
            raise ValueError(f"size must be positive, got {n}")
        sizes.append(n)
    return sizes


def load_class_names(yaml_path) -> list[str]:
    """Class names ordered by CLASS INDEX (not dict insertion order).

    Ultralytics writes names as {0: 'alcohol', 1: 'candy', ...}. Iterating that dict in
    insertion order would mislabel every detection while crashing nothing — so we sort
    explicitly by index.
    """
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"dataset yaml not found: {p}")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    # An empty or comment-only yaml parses to None, not {} -- doc["names"] would then
    # raise a raw TypeError that main()'s except (KeyError, ValueError) does not catch,
    # letting a traceback escape instead of the clean ERROR/return-1 pattern.
    if doc is None:
        raise ValueError(f"{p} is empty or contains no YAML mapping (expected a 'names' key)")
    names = doc["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def gate_verdict(baseline_map: float, model_map: float) -> str:
    """Classify the INT8 accuracy drop per the spec's quantization gate.

    Drop is in PERCENTAGE POINTS of mAP@50-95: <2 clean, 2-5 acceptable, >5 RED FLAG.
    A RED FLAG means the first suspect is the calibration data, not the input size.
    """
    # Round before comparing: raw float64 arithmetic can drift an exact 5.0pt
    # drop to 5.000000000000004, which would wrongly fail the <=5.0 test and
    # trigger a false RED FLAG. Precision to 1e-9 is far tighter than the
    # spec's 0.1pt reporting granularity, so it only absorbs float noise.
    drop_points = round((baseline_map - model_map) * 100.0, 9)
    if drop_points < 2.0:
        return "clean"
    if drop_points <= 5.0:
        return "acceptable"
    return "RED FLAG"


def build_export_kwargs(size: int, calib_yaml: str, export_format: str = "edgetpu") -> dict:
    """Ultralytics export kwargs for a fully-integer TFLite model.

    int8=True is non-negotiable: the Edge TPU executes ONLY fully-integer models.
    `data` supplies the calibration images that set the quantization ranges.

    format="edgetpu" is equally non-negotiable, and NOT merely because we target Coral:
    it is the only format for which Ultralytics applies the INT8 box-normalization that
    keeps class scores representable at all (see the module docstring). Exporting INT8
    via "tflite" or "saved_model" yields a model that detects nothing. The other formats
    stay reachable only for diagnosis (e.g. a float32 reference to bisect against).
    """
    return {"format": export_format, "int8": True, "imgsz": size, "data": calib_yaml}


def _assert_integer_io(in_dtype, out_dtype, name: str) -> None:
    """Raise unless BOTH graph input and output dtypes are integer (int8/uint8).

    Pure so it is directly unit-testable — the artifact-opening half (which needs a
    real interpreter and a real model) is kept separate in verify_full_integer().
    """
    integer_dtypes = (np.int8, np.uint8)
    bad = {"input": in_dtype, "output": out_dtype}
    offenders = {k: v for k, v in bad.items() if v not in integer_dtypes}
    if not offenders:
        return
    detail = ", ".join(f"{k}={np.dtype(v).name}" for k, v in offenders.items())
    raise NotFullyIntegerError(
        f"{name} is NOT a fully-integer graph ({detail}; both must be int8/uint8). "
        "The Edge TPU executes only fully-integer models. "
        "Remedy: re-run with --export-format edgetpu.")


def _assert_score_resolution(out_scale: float, name: str) -> None:
    """Raise unless the output tensor's INT8 scale can still encode a confidence.

    Pure, so the arithmetic is unit-testable without a real model — the artifact-opening
    half lives in verify_score_resolution().

    The output tensor holds box coords AND class probabilities under ONE shared scale.
    Without edgetpu's box normalization the boxes arrive in pixel units and force that
    scale far above 1.0, at which point every probability rounds to a two-value set and
    the model detects nothing while appearing entirely healthy.
    """
    if out_scale <= MAX_OUTPUT_SCALE:
        return
    levels = 1.0 / out_scale if out_scale else float("inf")
    raise ScoreResolutionError(
        f"{name} has an output quantization scale of {out_scale:.4f} — too coarse to "
        f"represent a class confidence (only ~{levels:.1f} levels across the 0..1 "
        f"probability range; the limit is {MAX_OUTPUT_SCALE}). Class scores have "
        "collapsed to a near-binary set, so this model will detect NOTHING and score "
        "mAP 0.000 despite being a valid, correctly-shaped, fully-integer TFLite file. "
        "Cause: the boxes were left in PIXEL units, dragging up the scale shared by "
        "boxes and scores. Remedy: re-run with --export-format edgetpu, the only format "
        "for which Ultralytics normalizes the boxes to 0..1.")


def _open_interpreter(path: Path):
    """Open a .tflite with whichever LiteRT interpreter is installed."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as exc:
            # Never skip the artifact checks: silently passing on the project's
            # non-negotiable constraints is worse than a loud failure.
            raise RuntimeError(
                "cannot verify the exported model: no LiteRT interpreter available. "
                "Install one: pip install ai-edge-litert"
            ) from exc

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def verify_full_integer(tflite_path) -> None:
    """Open the PRODUCED artifact and assert its graph I/O is fully integer.

    We assert on the ARTIFACT, not on the export kwargs: int8=True is merely deprecated
    sugar for quantize=8, and does not by itself imply integer graph I/O.
    """
    path = Path(tflite_path)
    interpreter = _open_interpreter(path)
    in_dtype = interpreter.get_input_details()[0]["dtype"]
    out_dtype = interpreter.get_output_details()[0]["dtype"]
    _assert_integer_io(in_dtype, out_dtype, path.name)


def verify_score_resolution(tflite_path) -> None:
    """Open the PRODUCED artifact and assert its output scale can encode a confidence.

    verify_full_integer() is not enough on its own: it passed the mAP-0.000 model
    without complaint, because that model's dtypes really were int8. Integer-ness and
    usability are different properties, and only this one is about detection surviving.
    """
    path = Path(tflite_path)
    interpreter = _open_interpreter(path)
    out = interpreter.get_output_details()[0]
    scale = float(out["quantization"][0])
    _assert_score_resolution(scale, path.name)


def _resolve_export_artifact(produced: Path) -> Path:
    """Normalize what model.export() returned to the CPU-runnable, fully-integer .tflite.

    Three shapes arrive here:
      - a plain .tflite FILE            -> use it
      - a `*_saved_model/` DIRECTORY    -> the *_full_integer_quant.tflite inside it
      - a compiled `*_edgetpu.tflite`   -> its CPU-runnable sibling

    That last case is the one format="edgetpu" actually returns. The compiled file only
    executes with a real Edge TPU delegate attached, so validating it on this desktop
    (and running it in the laptop demo) would fail. Its sibling — the same quantized
    graph, uncompiled — is what we measure and bundle. The compiled file is preserved
    separately by export_one_size() for the board.
    """
    produced = Path(produced)
    if produced.is_dir():
        candidates = sorted(produced.glob("*_full_integer_quant.tflite"))
        if not candidates:
            raise ValueError(
                f"export returned directory {produced} but it contains no "
                f"*_full_integer_quant.tflite — the Edge TPU needs a fully-integer "
                f"artifact. Found: {[p.name for p in produced.glob('*.tflite')]}")
        return candidates[0]
    if produced.is_file():
        if produced.name.endswith("_edgetpu.tflite"):
            sibling = produced.with_name(produced.name[: -len("_edgetpu.tflite")] + ".tflite")
            if not sibling.exists():
                raise ValueError(
                    f"edgetpu compile produced {produced.name} but its CPU-runnable "
                    f"sibling {sibling.name} (the *_full_integer_quant.tflite the "
                    f"compiler consumed) is missing — nothing can be validated.")
            return sibling
        return produced
    raise FileNotFoundError(f"export produced nothing at {produced}")


def verify_class_count(names: list[str]) -> None:
    """Fail loudly unless the model/labels carry exactly 17 coarse classes.

    A wrong class count means every detection is confidently mislabeled while nothing
    crashes — the worst failure mode, because it looks like success.
    """
    if len(names) != EXPECTED_CLASSES:
        raise ValueError(
            f"expected {EXPECTED_CLASSES} classes, got {len(names)} — wrong model or yaml?")


def export_one_size(weights, size: int, calib_yaml: str, out_dir: Path,
                    export_format: str = "edgetpu") -> Path:
    """Export weights -> INT8 TFLite at `size`. Returns the path of the copied artifact."""
    import shutil

    from ultralytics import YOLO

    model = YOLO(str(weights))

    # Check the MODEL's class count, not just the yaml's. Pointing --weights at a
    # stale 200-class checkpoint would otherwise bundle a 17-line classes.txt over a
    # 200-class model: every detection confidently mislabeled, nothing crashing.
    # Checked on the PyTorch side because `names` is always populated there; a
    # TFLite-side check is unreliable (AutoBackend invents synthetic names when the
    # metadata is missing).
    verify_class_count(list(model.names.values()))

    returned = Path(model.export(**build_export_kwargs(size, str(calib_yaml), export_format)))
    produced = _resolve_export_artifact(returned)

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"rpc_coarse17_int8_{size}.tflite"
    shutil.copy2(produced, dest)

    # Keep the compiled Edge TPU binary too — it is the artifact the board actually
    # runs, and re-exporting to recover it would cost the full export again. It cannot
    # be validated here (it needs a real TPU delegate), so it is preserved, not measured.
    if returned.is_file() and returned.name.endswith("_edgetpu.tflite"):
        shutil.copy2(returned, out_dir / f"rpc_coarse17_int8_{size}_edgetpu.tflite")

    # Assert the ARTIFACT, not the flag — and assert BOTH properties. A model can be
    # perfectly integer and still detect nothing (see ScoreResolutionError). Failures
    # here propagate to _run_sweep, which records FAILED. Do not swallow them.
    verify_full_integer(dest)
    verify_score_resolution(dest)
    return dest


def val_tflite(tflite_path, data_yaml, size: int) -> dict:
    """Validate an exported TFLite model. Returns mAP@50, mAP@50-95, and CPU latency (ms)."""
    from ultralytics import YOLO

    model = YOLO(str(tflite_path), task="detect")
    metrics = model.val(data=str(data_yaml), imgsz=size, verbose=False)
    return {
        "map50": float(metrics.box.map50),
        "map": float(metrics.box.map),
        # Indexed, not .get(..., 0.0): latency is a PUBLISHED column of the
        # deliverable table, so a missing key must fail loudly rather than
        # print a fabricated 0.0 ms that reads as a measurement.
        "latency_ms": float(metrics.speed["inference"]),
    }


def build_report_table(rows: list[dict]) -> str:
    """Render the size-vs-accuracy table. FAILED sizes are shown, never dropped."""
    lines = [
        "# INT8 Export Report",
        "",
        f"FP32 baseline — cluttered mAP@50-95: **{BASELINE['cluttered_map']:.3f}**, "
        f"single-item: **{BASELINE['single_map']:.3f}**",
        "",
        "Latency is desktop CPU, for comparing sizes against each other only — "
        "it is NOT a Coral FPS forecast.",
        "",
        "**Caveat — the measured drop is a LOWER BOUND (optimistic).** INT8 calibration "
        "and the cluttered eval both draw from the same `real_eval` images (Ultralytics "
        "calibrates from the `val:` split, which `dataset_real_blend.yml` points at "
        "`real_eval`). Quantization ranges are therefore tuned on the very images the "
        "quantization cost is measured against; on unseen scenes expect a larger drop.",
        "",
        "**Caveat — Δ is NOT purely the quantization cost below 640.** The FP32 baseline "
        "was measured at imgsz **640** (the training size), so for any smaller size the "
        "Δ column sums TWO costs: quantization AND the resolution drop. **Only the 640 "
        "row isolates quantization.** Read a large Δ at 320/448 as a resolution "
        "trade-off first — a falling mAP@50-95 while mAP@50 holds up is looser boxes "
        "(resolution), not damaged scores (quantization).",
        "",
        "| Size | Cluttered mAP@50 | Cluttered mAP@50-95 | Δ vs FP32 (pts) | Gate | "
        "Single mAP@50-95 | File | CPU latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -x["size"]):
        if r["status"] != "ok":
            # Show the actual status (FAILED vs. SKIPPED) rather than hardcoding
            # FAILED -- a size the sweep never attempted must never read as tested.
            status = r["status"]
            lines.append(
                f"| {r['size']} | {status} | {status} | — | {status} | — | — | {r['error']} |")
            continue
        delta = (r["cluttered_map"] - BASELINE["cluttered_map"]) * 100.0
        verdict = gate_verdict(BASELINE["cluttered_map"], r["cluttered_map"])
        lines.append(
            f"| {r['size']} | {r['cluttered_map50']:.3f} | {r['cluttered_map']:.3f} | "
            f"{delta:+.1f} | {verdict} | {r['single_map']:.3f} | "
            f"{r['bytes'] / 1e6:.1f} MB | {r['latency_ms']:.1f} ms |")
    return "\n".join(lines) + "\n"


def write_bundle(tflite_path, names: list[str], deploy_dir) -> None:
    """Assemble the laptop bundle: the chosen model + a DERIVED classes.txt.

    classes.txt is generated from the yaml names block, never hand-typed — a
    hand-typed ordering is exactly how you get silently mislabeled detections.
    """
    import shutil

    verify_class_count(names)
    deploy_dir = Path(deploy_dir)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # Exactly ONE model may ever sit in deploy/. Bundling 448 after 320 would
    # otherwise leave both, and the demo could silently run the size you did not pick.
    src = Path(tflite_path).resolve()
    for stale in deploy_dir.glob("*.tflite"):
        if stale.resolve() != src:          # never delete the model we are about to copy
            stale.unlink()

    shutil.copy2(tflite_path, deploy_dir / Path(tflite_path).name)
    (deploy_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def _run_sweep(args, sizes: list[int]) -> int:
    """Export + validate each size, then write export/report.md. Returns an exit code."""
    out_dir = PROJECT_ROOT / "export"
    rows = []
    aborted = False
    for idx, size in enumerate(sizes):
        print(f"\n=== size {size}: exporting INT8 (calibrating on {args.calib}) ===")
        try:
            tflite = export_one_size(args.weights, size, args.calib, out_dir,
                                     export_format=args.export_format)
            cluttered = val_tflite(tflite, args.calib, size)
            single = val_tflite(tflite, args.single, size)
            rows.append({
                "size": size, "status": "ok",
                "cluttered_map50": cluttered["map50"], "cluttered_map": cluttered["map"],
                "single_map50": single["map50"], "single_map": single["map"],
                "bytes": tflite.stat().st_size, "latency_ms": cluttered["latency_ms"],
            })
            print(f"  size {size}: cluttered mAP@50-95 = {cluttered['map']:.3f} "
                  f"({gate_verdict(BASELINE['cluttered_map'], cluttered['map'])})")
        except FormatLevelExportError as exc:
            # A FORMAT-level failure (float graph I/O, or an output scale too coarse to
            # encode a confidence). Neither has anything to do with input resolution, so
            # both recur identically at every remaining size. Record this size as FAILED
            # (never silently dropped), then stop -- burning ~10 GPU-minutes per
            # remaining size to relearn the same fact is exactly the waste this prevents.
            print(f"ERROR: size {size} FAILED: {exc}", file=sys.stderr)
            print(
                "ERROR: this is a FORMAT-level failure, not a size-specific one -- it "
                "will recur identically at every remaining size. Aborting the sweep "
                "instead of burning ~10 GPU-minutes per size to relearn the same fact. "
                "The remedy is named in the error above.",
                file=sys.stderr,
            )
            rows.append({"size": size, "status": "FAILED", "error": str(exc)})
            for skipped_size in sizes[idx + 1:]:
                rows.append({
                    "size": skipped_size, "status": "SKIPPED",
                    "error": f"sweep aborted after format-level failure at size {size}",
                })
            aborted = True
            break
        except Exception as exc:  # noqa: BLE001 — one bad size must not kill the sweep
            print(f"ERROR: size {size} FAILED: {exc}", file=sys.stderr)
            rows.append({"size": size, "status": "FAILED", "error": str(exc)})

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    report.write_text(build_report_table(rows), encoding="utf-8")
    print(f"\nWrote {report}")

    # Non-zero exit if the sweep was aborted, or if EVERY attempted size failed —
    # in both cases the sweep produced nothing usable (or incomplete) to trust blindly.
    return 1 if aborted or all(r["status"] != "ok" for r in rows) else 0


def main() -> int:
    p = argparse.ArgumentParser(description="INT8 export sweep + deploy bundle (non-destructive).")
    p.add_argument("--weights",
                   default="runs/detect/rpc_real_blend_b0_full/weights/best.pt")
    p.add_argument("--calib", default="dataset_real_blend.yml",
                   help="Calibration + cluttered-eval yaml. MUST be the real cluttered set.")
    p.add_argument("--single", default="eval_single_item.yml",
                   help="Single-item eval yaml.")
    p.add_argument("--sizes", default="640,448,320")
    p.add_argument("--export-format", default="edgetpu",
                   choices=["edgetpu", "tflite", "saved_model"],
                   help="Ultralytics export format. Leave this alone: 'edgetpu' is the "
                        "ONLY format for which Ultralytics normalizes the boxes, and "
                        "without that the INT8 class scores collapse and the model "
                        "detects nothing (mAP 0.000). The others are for diagnosis only.")
    p.add_argument("--bundle", type=int, default=None,
                   help="Assemble deploy/ from the already-exported model of this size.")
    p.add_argument("--dry", action="store_true", help="Parse args and exit (test hook).")
    args = p.parse_args()
    if args.dry:
        return 0

    weights = Path(args.weights)
    calib = Path(args.calib)
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        return 1
    if not calib.exists():
        print(f"ERROR: calibration yaml not found: {calib}", file=sys.stderr)
        return 1

    # --bundle needs --calib (for the class names) but NOT --single or --sizes, so
    # those two are gated on the sweep path below rather than demanded up front.
    try:
        names = load_class_names(calib)
        verify_class_count(names)
    except (KeyError, ValueError) as exc:
        print(f"ERROR: bad calibration yaml {calib}: {exc}", file=sys.stderr)
        return 1

    if args.bundle is not None:
        tflite = PROJECT_ROOT / "export" / f"rpc_coarse17_int8_{args.bundle}.tflite"
        if not tflite.exists():
            print(f"ERROR: no exported model at {tflite}; run the sweep first.", file=sys.stderr)
            return 1
        write_bundle(tflite, names, PROJECT_ROOT / "deploy")
        print(f"Bundled {tflite.name} + classes.txt -> deploy/")
        return 0

    single = Path(args.single)
    if not single.exists():
        print(f"ERROR: single-item eval yaml not found: {single}", file=sys.stderr)
        return 1

    try:
        sizes = parse_sizes(args.sizes)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return _run_sweep(args, sizes)


if __name__ == "__main__":
    raise SystemExit(main())
