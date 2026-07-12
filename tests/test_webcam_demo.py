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


def test_draw_detections_rejects_out_of_range_class_id(blank_canvas):
    # A short/stale classes.txt would otherwise blow up with a bare IndexError deep
    # in the draw path -- undiagnosable. The error must name the actual diagnosis.
    dets = [{"cls": 17, "conf": 0.9, "box": (1, 1, 20, 20)}]
    with pytest.raises(ValueError, match="stale classes.txt"):
        W.draw_detections(blank_canvas, dets, ["c"] * 17)


class _FakeCap:
    """Minimal cv2.VideoCapture stand-in: always yields one gray frame."""

    def __init__(self):
        import numpy as np
        self._frame = np.full((32, 32, 3), 127, dtype="uint8")

    def read(self):
        return True, self._frame.copy()

    def release(self):
        pass


def test_run_loop_returns_nonzero_when_inference_raises(capsys):
    # THE default cold-laptop path: LiteRTBackend is built lazily on the FIRST
    # predict(), where Ultralytics tries to pip-install ai-edge-litert. With no
    # network / no wheel, that raises -- and an unhandled traceback out of run_loop()
    # is what the user sees. It must be a clean error + non-zero exit instead.
    class _BoomModel:
        def predict(self, *a, **kw):
            raise RuntimeError("check_requirements: ai-edge-litert install failed")

    rc = W.run_loop(_BoomModel(), _FakeCap(), ["c"] * 17, conf=0.25)

    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "inference failed" in err
    assert "ai-edge-litert" in err          # the underlying cause is surfaced, not eaten


def test_run_loop_returns_nonzero_when_a_class_id_is_out_of_range(capsys):
    # A model emitting a class id past the bundled names (stale classes.txt) must
    # also exit cleanly, not blow a raw ValueError traceback out of the loop.
    import numpy as np

    class _BadClassModel:
        def predict(self, *a, **kw):
            class _B:
                cls = [99]                                    # past the 17 bundled names
                conf = [0.9]
                xyxy = np.array([[1, 1, 20, 20]])             # .tolist() like a real Boxes

            class _R:
                boxes = [_B()]

            return [_R()]

    rc = W.run_loop(_BadClassModel(), _FakeCap(), ["c"] * 17, conf=0.25)

    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "stale classes.txt" in err


class _DroppedCap:
    """cv2.VideoCapture stand-in that immediately reports a dead camera."""

    def read(self):
        return False, None

    def release(self):
        pass


class _OneFrameThenQuitCap:
    """Yields one real frame, then acts as if the user pressed 'q'."""

    def __init__(self):
        import numpy as np
        self._frame = np.full((32, 32, 3), 127, dtype="uint8")

    def read(self):
        return True, self._frame.copy()

    def release(self):
        pass


class _NoopModel:
    def predict(self, *a, **kw):
        return []


def test_run_loop_returns_nonzero_on_dropped_camera(capsys):
    # cap.read() -> (False, None) means the camera died mid-session (unplugged /
    # stream lost). Falling through to `return 0` would report SUCCESS to the shell
    # for a session that actually failed -- a silent failure the project forbids.
    rc = W.run_loop(_NoopModel(), _DroppedCap(), ["c"] * 17, conf=0.25)

    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "WARNING" not in err


def test_run_loop_returns_zero_on_clean_quit(monkeypatch):
    # A user-initiated 'q' quit is NOT a failure and must still exit 0, distinct
    # from a dropped camera. imshow is stubbed out -- this is a unit test with a
    # fake capture, not the GUI webcam demo itself, so no window should pop up.
    monkeypatch.setattr(W.cv2, "imshow", lambda *_a, **_kw: None)
    monkeypatch.setattr(W.cv2, "waitKey", lambda *_a, **_kw: ord("q"))

    rc = W.run_loop(_NoopModel(), _OneFrameThenQuitCap(), ["c"] * 17, conf=0.25)

    assert rc == 0


def test_resolve_model_path_finds_the_single_bundled_model(tmp_path):
    m = tmp_path / "rpc_coarse17_int8_448.tflite"
    m.write_bytes(b"TFL3")
    assert W.resolve_model_path(tmp_path) == m


def test_resolve_model_path_errors_when_no_model_bundled(tmp_path):
    with pytest.raises(FileNotFoundError, match="--bundle"):
        W.resolve_model_path(tmp_path)


def test_resolve_model_path_errors_when_several_models_present(tmp_path):
    (tmp_path / "rpc_coarse17_int8_320.tflite").write_bytes(b"TFL3")
    (tmp_path / "rpc_coarse17_int8_448.tflite").write_bytes(b"TFL3")
    with pytest.raises(RuntimeError) as exc:
        W.resolve_model_path(tmp_path)
    msg = str(exc.value)
    # must LIST them -- otherwise the user cannot tell which one would have run
    assert "rpc_coarse17_int8_320.tflite" in msg
    assert "rpc_coarse17_int8_448.tflite" in msg


def test_main_missing_model_returns_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv",
                        ["webcam_demo.py", "--model", str(tmp_path / "nope.tflite")])
    rc = W.main()
    assert rc == 1                          # never silently continue without a model
    assert "ERROR" in capsys.readouterr().err


def test_main_invalid_model_fails_before_opening_camera(monkeypatch, tmp_path, capsys):
    # A model file that EXISTS but is not a valid model. main() must fail cleanly on
    # the model load, and it must do so BEFORE touching the camera -- --camera 999
    # can never open, so if the ordering regresses to camera-first, this test would
    # surface a camera error instead of a model-load error.
    #
    # NOTE on fixture choice: a corrupt *.tflite* does NOT raise inside YOLO(...) --
    # verified empirically against the installed ultralytics. With task="detect"
    # passed explicitly (as webcam_demo.py always does), Model._load() skips content
    # inspection for non-.pt weights entirely; the LiteRT interpreter is only built
    # lazily on the first predict() call, which already happens inside main()'s
    # try/finally around run_loop(), so that specific case was never actually a leak.
    # A corrupt *.pt* DOES raise synchronously (torch.load -> UnpicklingError) and so
    # is the fixture that actually exercises the YOLO(...) construction guard this
    # fix adds.
    bad_model = tmp_path / "bad.pt"
    bad_model.write_bytes(b"not a real model")

    classes_path = tmp_path / "classes.txt"
    classes_path.write_text(
        "\n".join([
            "alcohol", "candy", "canned_food", "chocolate", "dessert", "dried_food",
            "dried_fruit", "drink", "gum", "instant_drink", "instant_noodles", "milk",
            "personal_hygiene", "puffed_food", "seasoner", "stationery", "tissue",
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", [
        "webcam_demo.py",
        "--model", str(bad_model),
        "--classes", str(classes_path),
        "--camera", "999",
    ])
    rc = W.main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err
    # Must be a MODEL load failure, not a camera-open failure -- camera 999 can never
    # open, so a camera-first ordering would ALSO return 1 with "ERROR" in stderr,
    # which would make this test pass for the wrong reason.
    assert "camera" not in err.lower(), (
        f"camera was touched before the (bad) model load failed: {err!r}")
    assert "model" in err.lower()
