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
CLASSES = PROJECT_ROOT / "coral" / "classes.txt"


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


def test_percentile_p90_uses_nearest_rank_not_naive_int_truncation():
    # Nearest-rank p90 (ceil(0.9*n) - 1, 0-based) is unambiguous for 1..N ms samples.
    # n=10 is the case the old `int(n * 0.9)` formula got wrong: it picked index 9 (the
    # maximum, 10.0) instead of the correct index 8 (the 9th-of-10 value, 9.0).
    assert DET._percentile([float(x) for x in range(1, 11)], 0.9) == 9.0    # n=10
    assert DET._percentile([float(x) for x in range(1, 4)], 0.9) == 3.0     # n=3
    assert DET._percentile([float(x) for x in range(1, 21)], 0.9) == 18.0   # n=20


def test_stage_timer_p90_at_ten_samples_is_not_the_maximum():
    # Same boundary, exercised through the public StageTimer.stats() path that the
    # loop and the report actually use.
    timer = DET.StageTimer()
    for ms in range(1, 11):
        timer.record("invoke", float(ms))
    stats = timer.stats()
    assert stats["invoke"]["n"] == 10
    assert stats["invoke"]["p90_ms"] == 9.0


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


def test_frame_grabber_keeps_only_the_latest_frame_no_backlog():
    """The anti-lag guarantee: a consumer slower than the camera must receive the NEWEST
    frame, never a stale queued one. FrameGrabber drains on a background thread, so read()
    must return the most recent frame the camera produced -- not the first one queued."""
    import threading
    import time

    class _CountingCap:
        """Emits frames tagged 0,1,2,... then holds on the last one."""
        def __init__(self, n):
            self.n = n
            self._i = 0
            self.produced_all = threading.Event()

        def read(self):
            if self._i < self.n:
                frame = np.full((4, 4, 3), self._i % 256, dtype=np.uint8)
                self._i += 1
                if self._i >= self.n:
                    self.produced_all.set()
                return True, frame
            time.sleep(0.005)                          # pace the thread; hold the last frame
            return True, np.full((4, 4, 3), (self.n - 1) % 256, dtype=np.uint8)

        def release(self):
            pass

    cap = _CountingCap(n=20)
    grabber = DET.FrameGrabber(cap).start()
    try:
        assert cap.produced_all.wait(1.0), "the fake camera never finished producing"
        time.sleep(0.02)                               # let the grabber store the last frame
        ok, frame = grabber.read()
        assert ok
        assert int(frame[0, 0, 0]) == 19, (            # latest (n-1), NOT frame 0
            "FrameGrabber handed out a stale frame -- exactly the backlog it exists to kill")
    finally:
        grabber.release()


def test_frame_grabber_release_stops_the_thread_and_frees_the_camera():
    import time

    class _IdleCap:
        released = False

        def read(self):
            time.sleep(0.002)
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    cap = _IdleCap()
    grabber = DET.FrameGrabber(cap).start()
    grabber.release()
    assert not grabber._thread.is_alive(), "release() must join the background thread"
    assert cap.released, "release() must release the underlying capture"


def test_run_loop_times_all_six_stages_including_draw():
    """draw_detections was untimed until recently; nothing guards it from silently
    regressing. Assert that all six stages are recorded, with draw explicitly named."""

    class _FakeDetector:
        """Minimal detector: detect() returns one detection and timing dict."""
        def detect(self, frame, conf, iou):
            detections = [{"cls": 0, "conf": 0.9, "box": (10, 10, 50, 50)}]
            timings = {"preprocess": 1.0, "invoke": 2.0, "decode": 0.5}
            return detections, timings

    class _LoopCap:
        """Yields the same frame repeatedly."""
        def __init__(self):
            self._frame = np.zeros((120, 160, 3), dtype=np.uint8)

        def read(self):
            return True, self._frame.copy()

        def release(self):
            pass

    class _QuitAfterTwoSink:
        """Publishes frames and signals quit after two frames."""
        def __init__(self):
            self._count = 0
            self.published_frames = []

        def publish(self, frame):
            self._count += 1
            self.published_frames.append(frame.copy())

        def should_quit(self):
            return self._count >= 2

        def close(self):
            pass

    names = ["class_0", "class_1"]
    detector = _FakeDetector()
    cap = _LoopCap()
    sink = _QuitAfterTwoSink()
    timer = DET.StageTimer()

    rc = DET.run_loop(detector, cap, names, conf=0.25, iou=0.45,
                      built_sinks=[sink], timer=timer)

    assert rc == 0, "run_loop should return 0 on clean quit via should_quit()"
    assert len(sink.published_frames) == 2, "both frames should have been published"

    stats = timer.stats()
    required_stages = {"capture", "preprocess", "invoke", "decode", "draw", "sink"}
    assert required_stages <= set(stats), (
        f"Missing stages in timer.stats(). "
        f"Expected: {required_stages}. "
        f"Got: {set(stats)}. "
        f"draw_detections must be timed, or its A35 cost vanishes from the accounting "
        f"and the reported FPS over-states the board.")
