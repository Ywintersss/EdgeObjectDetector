"""Step 0 on the board. Run this FIRST, over SSH, before anything else.

Its only job is to tell you which foundation is missing, in ten seconds, from a script
whose output is unambiguous -- instead of fifteen minutes later, from a confusing failure
deep inside the detect loop.

    python3 probe_board.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check(label: str, fn) -> bool:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 -- a probe reports every failure, never raises
        print(f"  FAIL  {label}: {exc}")
        return False
    print(f"  ok    {label}: {detail}")
    return True


def _numpy():
    import numpy
    return f"numpy {numpy.__version__}"


def _opencv():
    import cv2
    return f"opencv {cv2.__version__}"


def _model():
    models = sorted(HERE.glob("*_edgetpu.tflite"))
    if not models:
        raise FileNotFoundError(
            f"no *_edgetpu.tflite in {HERE} -- did the git clone bring the model? "
            f"(check .gitignore negations)")
    return f"{models[0].name} ({models[0].stat().st_size / 1e6:.1f} MB)"


def _classes():
    path = HERE / "classes.txt"
    names = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(names) != 17:
        raise ValueError(f"expected 17 class names, found {len(names)}")
    return f"17 classes, first={names[0]}"


def _delegate():
    import interpreter as I
    model = sorted(HERE.glob("*_edgetpu.tflite"))[0]
    it = I.make_interpreter(model, use_tpu=True)
    shape = it.get_input_details()[0]["shape"]
    return f"Edge TPU delegate bound, input {list(shape)}"


def _camera():
    import cv2
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                "/dev/video0 did not open. Is the webcam in the USB-C OTG port with an "
                "OTG adapter? If it enumerated but reads fail, the port may not source "
                "enough current -- try a POWERED OTG hub.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("camera opened but returned no frame (likely a power "
                               "problem -- try a powered OTG hub)")
        return f"captured a real {frame.shape[1]}x{frame.shape[0]} frame"
    finally:
        cap.release()


def main() -> int:
    print("Coral Dev Board Mini -- probing the foundations\n")
    results = [
        check("numpy", _numpy),
        check("opencv (sudo apt-get install python3-opencv)", _opencv),
        check("model present", _model),
        check("classes.txt", _classes),
        check("Edge TPU delegate", _delegate),
        check("USB webcam", _camera),
    ]
    failed = results.count(False)
    print()
    if failed:
        print(f"{failed} check(s) FAILED -- fix these before running detect.py.")
        return 1
    print("All checks passed. Run:  python3 detect.py --display stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
