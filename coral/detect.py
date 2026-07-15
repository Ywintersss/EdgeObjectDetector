"""Live overhead detector for the Coral Dev Board Mini.

    python3 detect.py                          # TPU + webcam + MJPEG stream on :8080
    python3 detect.py --display both           # ...and an HDMI window if Mendel allows one
    python3 detect.py --cpu --model ../deploy/rpc_coarse17_int8_320.tflite \
        --image shot.jpg                       # desktop dry-run: no TPU, no camera
        # The plain CPU model is not part of the board bundle -- it is produced on a
        # dev machine with `python export_int8.py --bundle 320`.

WHY THE STAGE TIMINGS EXIST. A single FPS number would mislead you. Four things can each
cap it independently, and they call for opposite responses:
  1. 18 of the model's ops fall back off the TPU onto the Cortex-A35 (96.2% coverage).
  2. The numpy decode chews 2100x21 values on that same modest CPU.
  3. JPEG-encoding each frame for the stream costs more A35 time.
  4. The USB webcam on a bandwidth-limited OTG port may simply not deliver frames faster.
"22 FPS" is consistent with all four and distinguishes none of them. So we time six
stages separately (capture, preprocess, invoke, decode, draw, sink), and separately
micro-benchmark invoke() alone.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

import decode as D
import interpreter as I
import sinks as S

HERE = Path(__file__).resolve().parent
EXPECTED_CLASSES = 17


def _percentile(ordered_samples: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an already-sorted list (e.g. fraction=0.9 -> p90).

    Nearest rank, 1-based: rank = ceil(fraction * n), 0-based index = rank - 1. Unlike
    the naive `int(n * fraction)`, this does NOT pick one rank too high when n is exactly
    divisible by 1/fraction -- e.g. at n=10, int(10*0.9)=9 (0-based) selects the maximum
    and mislabels it p90; ceil(0.9*10)-1=8 selects the correct 9th-of-10 value.
    """
    n = len(ordered_samples)
    if n == 0:
        raise ValueError("cannot take a percentile of zero samples")
    if n == 1:
        return ordered_samples[0]
    index = min(max(math.ceil(fraction * n) - 1, 0), n - 1)
    return ordered_samples[index]


class StageTimer:
    """Per-stage wall-clock samples -> median and p90. The whole point of this phase."""

    def __init__(self):
        self._samples: dict[str, list[float]] = {}

    @contextmanager
    def time(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - start) * 1000.0)

    def record(self, stage: str, elapsed_ms: float) -> None:
        """Add an already-measured sample. Detector.detect() times its own stages and
        returns them, so the loop folds those in here rather than re-timing them."""
        self._samples.setdefault(stage, []).append(elapsed_ms)

    def stats(self) -> dict:
        out = {}
        for stage, samples in self._samples.items():
            ordered = sorted(samples)
            out[stage] = {
                "median_ms": statistics.median(ordered),
                "p90_ms": _percentile(ordered, 0.9),
                "n": len(ordered),
            }
        return out

    def report(self) -> str:
        lines = [f"{'stage':<12} {'median':>9} {'p90':>9} {'n':>6}"]
        for stage, s in self.stats().items():
            lines.append(f"{stage:<12} {s['median_ms']:>8.1f}ms {s['p90_ms']:>8.1f}ms "
                         f"{s['n']:>6}")
        return "\n".join(lines)


def load_classes(path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"classes file not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def resolve_model(model_dir, use_tpu: bool) -> Path:
    """Pick the right artifact for the runtime -- never let them be confused.

    The *_edgetpu.tflite REQUIRES a TPU delegate; running it on CPU is not slow, it is
    impossible. And running the plain model with a delegate bound silently wastes the TPU.
    """
    directory = Path(model_dir)
    suffix = "_edgetpu.tflite" if use_tpu else ".tflite"
    candidates = [p for p in sorted(directory.glob("*.tflite"))
                  if p.name.endswith("_edgetpu.tflite") == use_tpu]
    if not candidates:
        raise FileNotFoundError(
            f"no {'*' + suffix} model in {directory} "
            f"(looked for {'an Edge TPU' if use_tpu else 'a plain CPU'} model)")
    if len(candidates) > 1:
        listed = ", ".join(p.name for p in candidates)
        raise RuntimeError(f"{len(candidates)} candidate models in {directory} ({listed})"
                           f" -- pass --model explicitly")
    return candidates[0]


class Detector:
    """model + interpreter, wrapped so the frame->detections path is one call."""

    def __init__(self, model_path, use_tpu: bool):
        self.interpreter = I.make_interpreter(model_path, use_tpu=use_tpu)
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.size = int(self.input["shape"][1])

    def invoke(self, quantized) -> np.ndarray:
        self.interpreter.set_tensor(self.input["index"], quantized)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output["index"])

    def detect(self, frame, conf: float, iou: float):
        """-> (detections, {stage: ms}). Timings are returned, not printed, so callers
        (the loop AND the tests) decide what to do with them."""
        timings = {}

        start = time.perf_counter()
        padded, ratio, pad = D.letterbox(frame, self.size)
        in_scale, in_zp = self.input["quantization"]
        quantized = D.quantize_input(padded, in_scale, in_zp, dtype=self.input["dtype"])
        timings["preprocess"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        raw = self.invoke(quantized)
        timings["invoke"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        detections = D.decode(raw, self.output["quantization"], ratio, pad, self.size,
                              conf_threshold=conf, iou_threshold=iou)
        timings["decode"] = (time.perf_counter() - start) * 1000.0

        return detections, timings


def draw_detections(frame, detections, names):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cls = d["cls"]
        if not 0 <= cls < len(names):
            raise ValueError(
                f"class id {cls} outside the {len(names)} bundled class names -- "
                f"stale classes.txt?")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{names[cls]} {d['conf']:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def benchmark_invoke(detector: Detector, frame, runs: int = 50) -> dict:
    """Median/p90 ms of invoke() ALONE -- the only figure comparable to Coral's published
    latencies. The FIRST call is discarded: it includes delegate warm-up and would drag
    the median toward a number you will never see again."""
    padded, _, _ = D.letterbox(frame, detector.size)
    in_scale, in_zp = detector.input["quantization"]
    quantized = D.quantize_input(padded, in_scale, in_zp, dtype=detector.input["dtype"])

    samples = []
    for i in range(runs):
        start = time.perf_counter()
        detector.invoke(quantized)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if i > 0:                       # drop the warm-up
            samples.append(elapsed_ms)

    ordered = sorted(samples)
    return {"median_ms": statistics.median(ordered), "p90_ms": _percentile(ordered, 0.9),
            "n": len(ordered)}


def run_still(detector, image_path, names, conf, iou) -> int:
    """Desktop dry-run: one image, no camera. Proves the whole path off-board."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: could not read image {image_path}", file=sys.stderr)
        return 1
    detections, timings = detector.detect(frame, conf, iou)
    for d in detections:
        print(f"  {names[d['cls']]:<18} conf {d['conf']:.2f}  box {d['box']}")
    print(f"detections: {len(detections)}")
    print("  ".join(f"{stage} {ms:.1f}ms" for stage, ms in timings.items()))
    bench = benchmark_invoke(detector, frame)
    print(f"invoke-only: median {bench['median_ms']:.1f}ms  p90 {bench['p90_ms']:.1f}ms "
          f"(n={bench['n']}, warm-up discarded)")
    return 0


class FrameGrabber:
    """Read the camera on a background thread, keeping ONLY the most recent frame.

    On the Mini the detect loop runs several times slower than the 30 fps camera, and cv2's
    VideoCapture hands back the OLDEST queued frame -- so a slow consumer falls further and
    further behind and the displayed latency grows without bound (you watch the past).
    Draining the queue here and always returning the latest frame pins latency at ~one
    frame, constant, however slow the loop is. Nothing useful is lost: the frames it skips
    are exactly the stale ones the loop could never have caught up on.

    It exposes the same read()/release() surface as cv2.VideoCapture, so the loop is
    unchanged -- it just receives a grabber instead of the raw capture.
    """

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._ok = False
        self._frame = None
        self._running = False
        self._thread = None

    def start(self):
        # Prime one frame synchronously so the very first read() can never lose the startup
        # race and return None (which the loop would read as a dropped camera).
        self._ok, self._frame = self._cap.read()
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def _pump(self):
        while self._running:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok, self._frame = ok, frame
            if not ok:                       # camera dropped -- stop pulling
                break

    def read(self):
        """-> (ok, frame): the newest frame the background thread has captured."""
        with self._lock:
            return self._ok, self._frame

    def release(self):
        """Stop the background thread, then release the underlying capture."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()


def run_loop(detector, cap, names, conf, iou, built_sinks, timer) -> int:
    while True:
        with timer.time("capture"):
            ok, frame = cap.read()
        if not ok:
            print("ERROR: camera dropped (unplugged or stream lost)", file=sys.stderr)
            return 1

        detections, timings = detector.detect(frame, conf, iou)
        for stage, ms in timings.items():
            timer.record(stage, ms)

        try:
            with timer.time("draw"):
                frame = draw_detections(frame, detections, names)
        except ValueError as exc:
            print(f"ERROR: cannot label detections: {exc}", file=sys.stderr)
            return 1

        try:
            with timer.time("sink"):
                for sink in built_sinks:
                    sink.publish(frame)
        except RuntimeError as exc:
            print(f"ERROR: could not publish frame to a sink: {exc}", file=sys.stderr)
            return 1

        if any(getattr(s, "should_quit", lambda: False)() for s in built_sinks):
            return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Coral Edge TPU overhead detector.")
    p.add_argument("--model", default=None, help="default: the right model in coral/")
    p.add_argument("--classes", default=str(HERE / "classes.txt"))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=320,
                   help="capture width (default 320). The Coral Mini's MUSB OTG port has a "
                        "hard USB isochronous-bandwidth limit; 320x240 is what reliably fits.")
    p.add_argument("--height", type=int, default=240, help="capture height (default 240)")
    p.add_argument("--fourcc", default="YUYV",
                   help="capture pixel format (default YUYV). MJPG and larger frames exceed "
                        "the Mini's MUSB bandwidth and fail STREAMON with ENOSPC (errno 28).")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--display", default="stream", choices=["stream", "hdmi", "both"])
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--cpu", action="store_true",
                   help="run the plain CPU model -- desktop dry-runs only, NOT the board")
    p.add_argument("--image", default=None,
                   help="run on one still image instead of a camera, then exit")
    args = p.parse_args()

    use_tpu = not args.cpu
    try:
        model_path = Path(args.model) if args.model else resolve_model(HERE, use_tpu)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        names = load_classes(args.classes)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if len(names) != EXPECTED_CLASSES:
        print(f"ERROR: expected {EXPECTED_CLASSES} classes, got {len(names)} -- wrong "
              f"classes.txt? Every box would be mislabeled.", file=sys.stderr)
        return 1

    # Load the model BEFORE opening any hardware: a bad model then fails with nothing
    # else open, so there is nothing to leak.
    try:
        detector = Detector(model_path, use_tpu=use_tpu)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {model_path.name} on {'Edge TPU' if use_tpu else 'CPU'}.")

    if args.image:
        return run_still(detector, args.image, names, args.conf, args.iou)

    # Build the sinks BEFORE opening the camera: MjpegSink binds a socket synchronously
    # and a stale detect.py still holding the port raises OSError. If that happened
    # after the camera was opened, the webcam would stay locked until physically
    # replugged -- painful on a headless board reached only over Wi-Fi.
    try:
        built = S.build_sinks(args.display, args.port)
    except OSError as exc:
        print(f"ERROR: could not start display/stream on port {args.port}: {exc}",
              file=sys.stderr)
        return 1
    for sink in built:
        if isinstance(sink, S.MjpegSink):
            print(f"Streaming at http://<board-ip>:{sink.port}/  (Ctrl-C to stop)")

    cap = cv2.VideoCapture(args.camera)
    # Request the capture format BEFORE the first read. On the Coral Mini the MUSB OTG
    # controller cannot reserve the isochronous USB bandwidth for MJPG or for frames much
    # above 320x240 -- STREAMON then fails with ENOSPC. YUYV 320x240 is the largest stream
    # it will host, and the 320x320-input model downsamples anyway, so nothing is lost.
    # Order matters: fourcc first, then the frame size.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        cap.release()
        for sink in built:
            sink.close()
        print(f"ERROR: could not open camera {args.camera}. Run probe_board.py to see "
              f"whether the OTG webcam enumerated at all.", file=sys.stderr)
        return 1
    # isOpened() is not enough on the Mini: the device opens, then STREAMON fails if the
    # requested format overruns the MUSB bandwidth. Prove we can pull one real frame before
    # committing to the loop -- otherwise the failure surfaces as a confusing mid-loop drop.
    ok, _ = cap.read()
    if not ok:
        cap.release()
        for sink in built:
            sink.close()
        print(f"ERROR: camera {args.camera} opened but could not stream at "
              f"{args.fourcc} {args.width}x{args.height}. The Mini's MUSB OTG port has a "
              f"hard USB-bandwidth ceiling -- MJPG and resolutions above ~320x240 fail here "
              f"with ENOSPC. Lower --width/--height, or check the uvcvideo FIX_BANDWIDTH "
              f"quirk (see coral/README.md).", file=sys.stderr)
        return 1
    print(f"POINT THE CAMERA DOWN at products on a plain surface. "
          f"({args.fourcc} {args.width}x{args.height})")

    # Pull frames on a background thread so a loop slower than the camera never accumulates
    # a queue of stale frames -- without this, the displayed latency grows without bound on
    # the Mini (see FrameGrabber). The grabber owns the camera from here on.
    frames = FrameGrabber(cap).start()

    timer = StageTimer()
    rc = 1
    try:
        rc = run_loop(detector, frames, names, args.conf, args.iou, built, timer)
    except KeyboardInterrupt:
        rc = 0
    finally:
        frames.release()          # stops the grabber thread AND releases the camera
        for sink in built:
            sink.close()
        if timer.stats():
            print("\n" + timer.report())
            total = sum(s["median_ms"] for s in timer.stats().values())
            print(f"\nend-to-end: {1000.0 / total:.1f} FPS "
                  f"({total:.1f}ms per frame, summed medians)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
