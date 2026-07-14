import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coral"))
import detect as DET  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CPU_MODEL = PROJECT_ROOT / "deploy" / "rpc_coarse17_int8_320.tflite"
REAL_IMAGES = PROJECT_ROOT / "dataset_real" / "images" / "real_eval"
CLASSES = PROJECT_ROOT / "deploy" / "classes.txt"


def test_stage_timer_records_medians_per_stage():
    timer = DET.StageTimer()
    for _ in range(3):
        with timer.time("invoke"):
            pass
        with timer.time("decode"):
            pass
    stats = timer.stats()
    assert set(stats) == {"invoke", "decode"}
    assert stats["invoke"]["n"] == 3
    assert stats["invoke"]["median_ms"] >= 0.0
    assert stats["invoke"]["p90_ms"] >= stats["invoke"]["median_ms"]


def test_stage_timer_report_names_every_stage():
    timer = DET.StageTimer()
    with timer.time("capture"):
        pass
    assert "capture" in timer.report()


def test_load_classes_reads_seventeen_names():
    names = DET.load_classes(CLASSES)
    assert len(names) == 17
    assert names[0] == "alcohol"


def test_draw_detections_rejects_a_class_id_outside_the_names():
    # A short/stale classes.txt would otherwise raise a bare IndexError deep in the draw
    # path. Name the real diagnosis instead.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="class id"):
        DET.draw_detections(frame, [{"cls": 99, "conf": 0.9, "box": (1, 1, 9, 9)}],
                            ["alcohol"])


@pytest.mark.skipif(not CPU_MODEL.exists() or not REAL_IMAGES.is_dir(),
                    reason="needs the bundled CPU model and real_eval images")
def test_detector_finds_objects_in_a_real_scene_and_times_every_stage():
    image = sorted(REAL_IMAGES.glob("*.jpg"))[0]
    frame = cv2.imread(str(image))
    detector = DET.Detector(CPU_MODEL, use_tpu=False)

    detections, timings = detector.detect(frame, conf=0.25, iou=0.45)

    assert len(detections) > 0, "a real cluttered scene must yield detections"
    assert all(0.0 < d["conf"] <= 1.0 for d in detections)
    # Every stage the report claims to measure must actually be measured.
    assert {"preprocess", "invoke", "decode"} <= set(timings)


@pytest.mark.skipif(not CPU_MODEL.exists() or not REAL_IMAGES.is_dir(),
                    reason="needs the bundled CPU model and real_eval images")
def test_benchmark_invoke_discards_the_warmup_call():
    image = sorted(REAL_IMAGES.glob("*.jpg"))[0]
    frame = cv2.imread(str(image))
    detector = DET.Detector(CPU_MODEL, use_tpu=False)

    result = DET.benchmark_invoke(detector, frame, runs=5)

    assert result["n"] == 4          # 5 runs, first one dropped as warm-up
    assert result["median_ms"] > 0.0
