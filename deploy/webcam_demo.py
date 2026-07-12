"""
webcam_demo.py — Live overhead-camera demo for the coarse-17 INT8 detector.

Usage:
    python webcam_demo.py --model rpc_coarse17_int8_320.tflite --camera 0

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
        label = f"{names[d['cls']]} {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame
