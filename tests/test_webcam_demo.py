import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import webcam_demo as W  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_ROOT / "runs/detect/rpc_real_blend_b0_full/weights/best.pt"
REAL_IMAGES = PROJECT_ROOT / "dataset_real/images/real_eval"


def test_load_classes_reads_one_per_line(tmp_path):
    p = tmp_path / "classes.txt"
    p.write_text("alcohol\ncandy\ncanned_food\n", encoding="utf-8")
    assert W.load_classes(p) == ["alcohol", "candy", "canned_food"]


def test_load_classes_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        W.load_classes(tmp_path / "nope.txt")


def test_open_camera_invalid_index_raises_with_enumeration():
    # index 999 cannot exist; the error must TELL the user what does exist,
    # because silently grabbing the wrong camera is the classic failure here.
    with pytest.raises(RuntimeError, match="[Cc]amera"):
        W.open_camera(999)


@pytest.mark.skipif(not (WEIGHTS.exists() and REAL_IMAGES.is_dir()),
                    reason="trained weights or real_eval images not present")
def test_detect_frame_finds_objects_in_a_real_cluttered_image():
    from ultralytics import YOLO

    img_path = next(REAL_IMAGES.glob("*.jpg"))
    frame = cv2.imread(str(img_path))
    assert frame is not None

    model = YOLO(str(WEIGHTS))
    dets = W.detect_frame(model, frame, conf=0.25)

    assert len(dets) >= 1, "model found nothing in a real cluttered checkout scene"
    h, w = frame.shape[:2]
    for d in dets:
        assert 0 <= d["cls"] <= 16, f"class index {d['cls']} outside coarse-17 range"
        assert 0.0 <= d["conf"] <= 1.0
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, "box outside image bounds"


@pytest.mark.skipif(not (WEIGHTS.exists() and REAL_IMAGES.is_dir()),
                    reason="trained weights or real_eval images not present")
def test_draw_detections_returns_same_shape_frame():
    from ultralytics import YOLO

    img_path = next(REAL_IMAGES.glob("*.jpg"))
    frame = cv2.imread(str(img_path))
    model = YOLO(str(WEIGHTS))
    dets = W.detect_frame(model, frame, conf=0.25)

    out = W.draw_detections(frame.copy(), dets, ["c"] * 17)
    assert out.shape == frame.shape


def test_main_missing_model_returns_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv",
                        ["webcam_demo.py", "--model", str(tmp_path / "nope.tflite")])
    rc = W.main()
    assert rc == 1                          # never silently continue without a model
    assert "ERROR" in capsys.readouterr().err
