"""
webcam_demo.py — Live overhead-camera demo for the coarse-17 INT8 detector.

Usage:
    python webcam_demo.py --camera 0        # uses the single model bundled in deploy/

POINT THE CAMERA DOWN at products on a plain surface. This model was trained on
top-down RPC checkout scenes; at eye level it is out-of-domain and will underperform.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
EXPECTED_CLASSES = 17


def resolve_model_path(deploy_dir) -> Path:
    """Find THE bundled model in deploy/ — exactly one .tflite must be present.

    Never hardcode a filename: a user who bundled 448 would otherwise silently run a
    leftover 320 model. Zero or several matches are both errors, not guesses.
    """
    deploy_dir = Path(deploy_dir)
    models = sorted(deploy_dir.glob("*.tflite"))
    if not models:
        raise FileNotFoundError(
            f"no model in {deploy_dir}; run: python export_int8.py --bundle SIZE")
    if len(models) > 1:
        listed = ", ".join(m.name for m in models)
        raise RuntimeError(
            f"{len(models)} models in {deploy_dir} ({listed}) — cannot tell which one you "
            f"want. Pass --model explicitly, or re-bundle to leave exactly one.")
    return models[0]


def load_classes(path) -> list[str]:
    """Read classes.txt — one class name per line, in class-index order."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"classes file not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def list_available_cameras(max_index: int = 5) -> list[int]:
    """Probe camera indices 0..max_index-1 and return the ones that actually open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


def open_camera(index: int):
    """Open a camera, or fail loudly listing the indices that DO work.

    Silently opening the wrong camera is the classic failure here — virtual cams
    (OBS, Teams) and external webcams shift the indices around.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        available = list_available_cameras()
        raise RuntimeError(
            f"could not open camera index {index}. Working camera indices: "
            f"{available if available else 'NONE FOUND'}")
    return cap


def detect_frame(model, frame, conf: float = 0.25) -> list[dict]:
    """Run the detector on one BGR frame. Pure w.r.t. the camera, so it is testable."""
    results = model.predict(frame, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append({
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "box": (x1, y1, x2, y2),
            })
    return detections


def draw_detections(frame, detections: list[dict], names: list[str]):
    """Draw boxes + '<class> <conf>' labels onto the frame (mutates and returns it)."""
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cls = d["cls"]
        # An unguarded names[cls] raises a bare IndexError deep in the draw path.
        # Name the actual diagnosis instead: a short/stale classes.txt was bundled.
        if not 0 <= cls < len(names):
            raise ValueError(
                f"class id {cls} outside the {len(names)} bundled class names — "
                f"stale classes.txt?")
        label = f"{names[cls]} {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def run_loop(model, cap, names, conf: float) -> int:
    """Read frames, detect, draw, show FPS. 'q' quits. Returns a process exit code.

    Inference is guarded because the DEFAULT cold-laptop path can fail here, not
    earlier: Ultralytics builds the LiteRT backend lazily on the FIRST predict(), and
    LiteRTBackend.load_model() runs check_requirements("ai-edge-litert>=2.1.4") — a
    runtime pip install. No network or no Windows wheel and it raises, mid-loop. An
    unhandled traceback out of here is what a user following the README would see.
    """
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            # A dead camera (unplugged / stream lost mid-session) is a FAILED
            # session, not a clean exit -- falling through to `return 0` here would
            # report success to the shell for a run that produced nothing further.
            print("ERROR: camera dropped (unplugged or stream lost)", file=sys.stderr)
            return 1

        try:
            detections = detect_frame(model, frame, conf=conf)
        except Exception as exc:  # noqa: BLE001 — any inference failure must exit cleanly
            print(f"ERROR: inference failed: {exc}", file=sys.stderr)
            print("       (if this mentions ai-edge-litert, run: "
                  "pip install ai-edge-litert)", file=sys.stderr)
            return 1

        try:
            frame = draw_detections(frame, detections, names)
        except ValueError as exc:  # stale/short classes.txt vs. the model's class count
            print(f"ERROR: cannot label detections: {exc}", file=sys.stderr)
            return 1

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"{fps:5.1f} FPS  |  {len(detections)} objects",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "POINT CAMERA DOWN at a plain surface",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow("EdgeObjectDetector — coarse-17 (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Live overhead-camera demo (coarse-17 INT8).")
    p.add_argument("--model", default=None,
                   help="Path to the .tflite model. Default: the single model bundled "
                        "in deploy/ (never a hardcoded size — that is how you end up "
                        "running a stale export).")
    p.add_argument("--classes", default=str(HERE / "classes.txt"))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    if args.model is None:
        try:
            model_path = resolve_model_path(HERE)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        model_path = Path(args.model)

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    try:
        names = load_classes(args.classes)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if len(names) != EXPECTED_CLASSES:
        print(f"ERROR: expected {EXPECTED_CLASSES} classes, got {len(names)} — "
              f"wrong classes.txt bundled? Every box would be mislabeled.", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ERROR: ultralytics not installed ({exc}). Run: pip install ultralytics "
              f"opencv-python", file=sys.stderr)
        return 1

    # Load the model BEFORE touching the camera: a bad model (corrupt/truncated
    # export -- the shakiest part of the INT8 toolchain) then fails before any
    # hardware is opened, so there is nothing to leak.
    try:
        model = YOLO(str(model_path), task="detect")
    except Exception as exc:
        print(f"ERROR: failed to load model {model_path}: {exc}", file=sys.stderr)
        return 1

    try:
        cap = open_camera(args.camera)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Running {model_path.name} on camera {args.camera}. Press 'q' to quit.")
    print("POINT THE CAMERA DOWN at products on a plain surface.")

    rc = 1   # if run_loop raises, the finally below still frees the camera; don't exit 0
    try:
        rc = run_loop(model, cap, names, args.conf)
    finally:
        # Always release the camera and tear the window down, even if run_loop threw.
        cap.release()
        cv2.destroyAllWindows()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
