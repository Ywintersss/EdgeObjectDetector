# Coral Dev Board Mini Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the compiled `rpc_coarse17_int8_320_edgetpu.tflite` on a Coral Dev Board Mini with a live overhead USB webcam, and measure per-stage timings that identify the real bottleneck.

**Architecture:** Ultralytics cannot run on the board (it pulls in torch; the board is a quad-core Cortex-A35). So YOLO post-processing — letterbox, input quantization, box decode, NMS — is reimplemented in pure numpy in `coral/`. That decode is the highest-risk part of the port because it fails *silently*, so it is validated on the desktop against a known-good oracle (Ultralytics driving `deploy/rpc_coarse17_int8_320.tflite`, the CPU twin of the exact model the board runs) **before the board is ever involved**. Only `interpreter.py` differs between desktop and board, so the hardware crossing changes exactly one variable.

**Tech Stack:** Python, numpy, OpenCV. `tflite_runtime` / `ai_edge_litert` for inference. `pycoral`'s `libedgetpu` delegate on the board. Ultralytics appears **only in the oracle test**, never in shipped `coral/` code. pytest.

## Global Constraints

- **Nothing in `coral/` may import `ultralytics` or `torch`.** They cannot be installed on the board. The single exception is `tests/test_coral_oracle.py`, which runs on the desktop only.
- **`coral/` modules are flat siblings, not a package** (no `__init__.py`) — the board runs `python3 detect.py` from inside `coral/`, and `detect.py` does `import decode`. This mirrors the existing `deploy/` convention.
- **Desktop commands use `python`, not `python3`** (project convention). **On the board it is `python3`** — Mendel Linux. Do not confuse the two.
- **Model signature is fixed and verified** (do not re-derive it):
  - Input: `[1, 320, 320, 3]`, `int8`, quantization `(0.003921568859368563, -128)`
  - Output: `[1, 21, 2100]`, `int8`, quantization `(0.005536458920687437, -128)`
  - Output channels **0–3** = box `cx, cy, w, h`, **normalized 0..1**. Channels **4–20** = the 17 class scores (already sigmoid'd by the detect head). Verified empirically on a real image.
- **Never hardcode quantization params in shipped code** — read them from the interpreter's tensor details at runtime. The constants above are for writing tests, not for the runtime path.
- **No silent failures** (project rule). Every error path names the actual diagnosis.
- Class count is **17**; names live in `coral/classes.txt` in class-index order.
- Tests import subdirectory modules via the existing pattern in `tests/test_webcam_demo.py`:
  `sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coral"))`

---

## File Structure

| File | Responsibility |
|---|---|
| `coral/decode.py` | Pure numpy: letterbox, input quantization, NMS, output decode. No TPU, no camera, no Ultralytics. |
| `coral/interpreter.py` | Load a `.tflite`; bind the Edge TPU delegate or don't. **The only desktop/board difference.** |
| `coral/sinks.py` | MJPEG HTTP sink (must work); HDMI `imshow` sink (best-effort). |
| `coral/detect.py` | The runner: camera → preprocess → invoke → decode → sinks, with per-stage timing. |
| `coral/probe_board.py` | Step 0 on the board: prove delegate + OpenCV + `/dev/video0` before anything else. |
| `coral/README.md` | Board setup and run instructions. |
| `coral/classes.txt` | 17 class names (copied from `deploy/`). Committed. |
| `coral/rpc_coarse17_int8_320_edgetpu.tflite` | The compiled model. Committed (3.9 MB). |
| `tests/test_coral_decode.py` | Unit tests for every function in `decode.py`. |
| `tests/test_coral_interpreter.py` | CPU load works; missing TPU delegate raises a *named* error. |
| `tests/test_coral_oracle.py` | **The oracle.** Our pipeline vs Ultralytics on a real image. |
| `tests/test_coral_sinks.py` | MJPEG serves a real frame; HDMI degrades instead of crashing. |
| `tests/test_coral_detect.py` | Stage timer stats; end-to-end still-image run on the CPU model. |
| `.gitignore` | Modify: negations so the deployment artifacts are committed. |

---

### Task 1: The decode — pure numpy post-processing

**Files:**
- Create: `coral/decode.py`
- Test: `tests/test_coral_decode.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `NUM_CLASSES: int = 17`, `NUM_BOX_CHANNELS: int = 4`
  - `letterbox(frame: np.ndarray, size: int, pad_value: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]` → `(padded_bgr, ratio, (dw, dh))`
  - `quantize_input(letterboxed_bgr: np.ndarray, scale: float, zero_point: int, dtype=np.int8) -> np.ndarray` → NHWC batch of 1
  - `nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]` → kept indices, highest score first
  - `decode(output: np.ndarray, quant: tuple[float, int], ratio: float, pad: tuple[int, int], size: int, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> list[dict]` → each dict is `{"cls": int, "conf": float, "box": (x1, y1, x2, y2)}`, sorted by descending confidence

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_coral_decode.py
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coral"))
import decode as D  # noqa: E402

# The REAL signature of rpc_coarse17_int8_320_edgetpu.tflite, measured. Tests are
# written against the real thing, not an invented one.
OUT_QUANT = (0.005536458920687437, -128)
SIZE = 320


def _empty_output():
    """A [1, 21, 2100] int8 tensor that dequantizes to all zeros."""
    return np.full((1, 21, 2100), -128, dtype=np.int8)


def _quantize(value: float) -> int:
    scale, zero_point = OUT_QUANT
    return int(np.clip(round(value / scale) + zero_point, -128, 127))


def test_letterbox_pads_to_square_without_distorting():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)   # 2:1 landscape
    padded, ratio, (dw, dh) = D.letterbox(frame, SIZE)
    assert padded.shape == (SIZE, SIZE, 3)
    assert ratio == pytest.approx(SIZE / 200)          # limited by the WIDE side
    assert dw == 0                                     # fills the width
    assert dh == (SIZE - 160) // 2                     # 100 * 1.6 = 160 tall -> padded
    # The pad must be the fill value, not black -- black is a legitimate image colour.
    assert (padded[0, 0] == 114).all()


def test_letterbox_leaves_a_square_frame_unpadded():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    padded, ratio, (dw, dh) = D.letterbox(frame, SIZE)
    assert (dw, dh) == (0, 0)
    assert ratio == pytest.approx(0.5)


def test_quantize_input_maps_white_to_the_top_of_the_int8_range():
    white = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    q = D.quantize_input(white, scale=0.003921568859368563, zero_point=-128)
    assert q.shape == (1, SIZE, SIZE, 3)
    assert q.dtype == np.int8
    assert (q == 127).all()          # 1.0 / (1/255) + (-128) = 127


def test_quantize_input_maps_black_to_the_zero_point():
    black = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    q = D.quantize_input(black, scale=0.003921568859368563, zero_point=-128)
    assert (q == -128).all()


def test_quantize_input_converts_bgr_to_rgb():
    # A pure-BLUE BGR pixel (255, 0, 0) must land in the LAST (red-position) channel
    # once converted to RGB. Getting this backwards silently wrecks accuracy.
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    frame[..., 0] = 255                                  # BGR blue
    q = D.quantize_input(frame, scale=0.003921568859368563, zero_point=-128)
    assert (q[0, :, :, 2] == 127).all()                  # RGB channel 2 == blue
    assert (q[0, :, :, 0] == -128).all()


def test_nms_suppresses_a_heavily_overlapping_box():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert D.nms(boxes, scores, iou_threshold=0.45) == [0]      # keeps the stronger


def test_nms_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert sorted(D.nms(boxes, scores, iou_threshold=0.45)) == [0, 1]


def test_nms_returns_indices_highest_score_first():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.2, 0.95], dtype=np.float32)
    assert D.nms(boxes, scores, iou_threshold=0.45)[0] == 1


def test_decode_returns_nothing_when_every_score_is_below_threshold():
    assert D.decode(_empty_output(), OUT_QUANT, ratio=1.0, pad=(0, 0), size=SIZE) == []


def test_decode_recovers_a_planted_box_in_original_frame_coordinates():
    # Plant ONE anchor: a centred box, half the image wide/tall, class 7 at conf ~0.9.
    out = _empty_output()
    anchor = 1234
    for channel, value in ((0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)):   # cx, cy, w, h
        out[0, channel, anchor] = _quantize(value)
    out[0, 4 + 7, anchor] = _quantize(0.9)                            # class 7

    dets = D.decode(out, OUT_QUANT, ratio=1.0, pad=(0, 0), size=SIZE)

    assert len(dets) == 1
    assert dets[0]["cls"] == 7
    assert dets[0]["conf"] == pytest.approx(0.9, abs=0.01)   # int8 grid is ~0.0055 coarse
    x1, y1, x2, y2 = dets[0]["box"]
    # centre (160,160), 160x160 -> corners (80,80)-(240,240)
    assert (x1, y1, x2, y2) == pytest.approx((80, 80, 240, 240), abs=2)


def test_decode_undoes_the_letterbox_padding():
    # Same planted box, but pretend the frame was letterboxed: half-scale with 40px of
    # top/bottom padding. The box must come back in ORIGINAL frame coordinates.
    out = _empty_output()
    anchor = 99
    for channel, value in ((0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)):
        out[0, channel, anchor] = _quantize(value)
    out[0, 4 + 2, anchor] = _quantize(0.8)

    dets = D.decode(out, OUT_QUANT, ratio=0.5, pad=(0, 40), size=SIZE)

    x1, y1, x2, y2 = dets[0]["box"]
    # x: (160 -+ 80 - 0)   / 0.5 -> 160 .. 480
    # y: (160 -+ 80 - 40)  / 0.5 -> 80  .. 400
    assert (x1, x2) == pytest.approx((160, 480), abs=2)
    assert (y1, y2) == pytest.approx((80, 400), abs=2)


def test_decode_suppresses_duplicates_of_the_same_class():
    # Two nearly identical boxes, same class -> NMS collapses them to one.
    out = _empty_output()
    for anchor, conf in ((10, 0.9), (11, 0.7)):
        for channel, value in ((0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)):
            out[0, channel, anchor] = _quantize(value)
        out[0, 4 + 3, anchor] = _quantize(conf)

    dets = D.decode(out, OUT_QUANT, ratio=1.0, pad=(0, 0), size=SIZE)
    assert len(dets) == 1
    assert dets[0]["conf"] == pytest.approx(0.9, abs=0.01)


def test_decode_keeps_overlapping_boxes_of_DIFFERENT_classes():
    # NMS must be class-wise. A drink sitting on a box of chocolate overlaps heavily,
    # and suppressing one because of the other would be a real detection loss.
    out = _empty_output()
    for anchor, cls in ((20, 3), (21, 9)):
        for channel, value in ((0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)):
            out[0, channel, anchor] = _quantize(value)
        out[0, 4 + cls, anchor] = _quantize(0.9)

    dets = D.decode(out, OUT_QUANT, ratio=1.0, pad=(0, 0), size=SIZE)
    assert sorted(d["cls"] for d in dets) == [3, 9]
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `python -m pytest tests/test_coral_decode.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'decode'`.

- [ ] **Step 3: Write `coral/decode.py`**

```python
"""Pure-numpy YOLO post-processing for the coarse-17 Edge TPU detector.

No TPU, no camera, no Ultralytics -- which is the point. This is the highest-risk code
in the Coral port (a transposed axis produces plausible-looking garbage, not an
exception), so it is written to run on the DESKTOP, where it can be tested against the
CPU twin of the exact model the board runs.

The output tensor is [1, 21, 2100] int8:
    channels 0..3    box: cx, cy, w, h -- NORMALIZED 0..1
    channels 4..20   the 17 class scores (the detect head already applied sigmoid)

Boxes are normalized ONLY because the export went through format="edgetpu", the one
format for which Ultralytics rescales them out of pixel units. That is the Phase 1 fix
(see export_int8.py). Any other export format emits PIXEL-unit boxes and this decode
would be wrong -- which is exactly why export_int8.py refuses to produce one.
"""

import cv2
import numpy as np

NUM_CLASSES = 17
NUM_BOX_CHANNELS = 4
PAD_VALUE = 114          # Ultralytics' letterbox fill; the oracle test depends on it


def letterbox(frame, size, pad_value=PAD_VALUE):
    """Resize preserving aspect ratio, pad to (size, size). -> (padded, ratio, (dw, dh)).

    Returns the ratio and padding because decode() must UNDO them to put boxes back into
    original frame coordinates.
    """
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"cannot letterbox an empty frame of shape {frame.shape}")
    ratio = min(size / h, size / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=frame.dtype)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, ratio, (dw, dh)


def quantize_input(letterboxed_bgr, scale, zero_point, dtype=np.int8):
    """BGR uint8 frame -> the model's quantized NHWC input batch.

    scale/zero_point MUST come from the interpreter's input details at runtime, never
    from a constant: they are a property of the exported artifact, not of the code.
    """
    rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    info = np.iinfo(dtype)
    q = np.round(rgb / scale) + zero_point
    return np.clip(q, info.min, info.max).astype(dtype)[None]


def nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression on xyxy boxes. -> kept indices, best score first."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_best = (boxes[best, 2] - boxes[best, 0]) * (boxes[best, 3] - boxes[best, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = np.maximum(area_best + area_rest - inter, 1e-9)
        order = rest[(inter / union) < iou_threshold]
    return keep


def decode(output, quant, ratio, pad, size, conf_threshold=0.25, iou_threshold=0.45):
    """[1, 21, 2100] int8 -> detections in ORIGINAL frame coordinates.

    Each detection: {"cls": int, "conf": float, "box": (x1, y1, x2, y2)}.
    """
    expected = NUM_BOX_CHANNELS + NUM_CLASSES
    if output.ndim != 3 or output.shape[1] != expected:
        # Shape drift is the silent killer here -- name it rather than index blindly.
        raise ValueError(
            f"expected an output of shape [1, {expected}, N], got {tuple(output.shape)}; "
            f"the model does not match this decoder")

    scale, zero_point = quant
    t = (output.astype(np.float32) - zero_point) * scale
    t = t[0].T                                        # [anchors, 21]

    boxes_n = t[:, :NUM_BOX_CHANNELS]
    scores = t[:, NUM_BOX_CHANNELS:]
    classes = scores.argmax(axis=1)
    confs = scores.max(axis=1)

    hit = confs >= conf_threshold
    if not hit.any():
        return []
    boxes_n, classes, confs = boxes_n[hit], classes[hit], confs[hit]

    # normalized cxcywh -> letterboxed pixels -> xyxy
    cx, cy, w, h = (boxes_n[:, i] * size for i in range(4))
    dw, dh = pad
    x1 = (cx - w / 2 - dw) / ratio
    y1 = (cy - h / 2 - dh) / ratio
    x2 = (cx + w / 2 - dw) / ratio
    y2 = (cy + h / 2 - dh) / ratio
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    detections = []
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        for k in nms(xyxy[idx], confs[idx], iou_threshold):
            j = idx[k]
            detections.append({
                "cls": int(cls),
                "conf": float(confs[j]),
                "box": tuple(int(round(v)) for v in xyxy[j]),
            })
    detections.sort(key=lambda d: -d["conf"])
    return detections
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_coral_decode.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add coral/decode.py tests/test_coral_decode.py
git commit -m "feat(coral): pure-numpy YOLO decode, letterbox, quantize and class-wise NMS

Co-Authored-By: Claude <claude@anthropic.com>"
```

---

### Task 2: The interpreter — the one thing that differs between desktop and board

**Files:**
- Create: `coral/interpreter.py`
- Test: `tests/test_coral_interpreter.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EDGETPU_LIB: dict[str, str]` — platform → shared-library name
  - `make_interpreter(model_path: str | Path, use_tpu: bool)` → an allocated interpreter exposing `get_input_details()`, `get_output_details()`, `set_tensor()`, `invoke()`, `get_tensor()`
  - Raises `RuntimeError` (named, never silent) if the model is missing, no TFLite runtime is installed, or the Edge TPU delegate cannot be bound.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_coral_interpreter.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coral"))
import interpreter as I  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CPU_MODEL = PROJECT_ROOT / "deploy" / "rpc_coarse17_int8_320.tflite"

pytestmark = pytest.mark.skipif(
    not CPU_MODEL.exists(),
    reason=f"{CPU_MODEL} missing -- run: python export_int8.py --bundle 320")


def test_make_interpreter_loads_the_cpu_model_with_the_expected_signature():
    it = I.make_interpreter(CPU_MODEL, use_tpu=False)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    assert list(inp["shape"]) == [1, 320, 320, 3]
    assert list(out["shape"]) == [1, 21, 2100]
    # Fully-integer both ends -- the property the Edge TPU actually requires.
    assert inp["dtype"].__name__ == "int8"
    assert out["dtype"].__name__ == "int8"


def test_make_interpreter_allocates_so_invoke_works_immediately():
    import numpy as np
    it = I.make_interpreter(CPU_MODEL, use_tpu=False)
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], np.zeros((1, 320, 320, 3), dtype=np.int8))
    it.invoke()   # would raise if allocate_tensors() had not been called
    assert it.get_tensor(it.get_output_details()[0]["index"]).shape == (1, 21, 2100)


def test_missing_model_is_named_not_swallowed(tmp_path):
    with pytest.raises(RuntimeError, match="model not found"):
        I.make_interpreter(tmp_path / "nope.tflite", use_tpu=False)


def test_requesting_the_tpu_without_one_fails_loudly_and_says_why():
    # There is no Edge TPU on the desktop. The failure must NAME the delegate, so that
    # on the board a real delegate problem is distinguishable from a missing model.
    with pytest.raises(RuntimeError, match="Edge TPU delegate"):
        I.make_interpreter(CPU_MODEL, use_tpu=True)
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `python -m pytest tests/test_coral_interpreter.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'interpreter'`.

- [ ] **Step 3: Write `coral/interpreter.py`**

```python
"""Load a .tflite model, with or without the Edge TPU delegate.

This is the ONLY module that differs between the desktop and the Coral. Everything else
-- preprocessing, decode, NMS, drawing -- is byte-identical on both. That is deliberate:
it means the desktop oracle test exercises the same code the board runs, and the hardware
crossing changes exactly one variable. If the board misbehaves, it is here or it is the
hardware; it is not the decode.

Runtime preference order: tflite_runtime (what PyCoral installs on Mendel) -> ai_edge_litert
(the desktop runtime) -> tensorflow.lite. Whichever is present wins.
"""

import platform
from pathlib import Path

# libedgetpu ships under a different name on each OS. Only the Linux one matters for the
# board; the others exist so a wrong-platform attempt fails with a clear message.
EDGETPU_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}


def _load_runtime():
    """-> (Interpreter, load_delegate) from whichever TFLite runtime is installed."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        pass
    try:
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError as exc:
        raise RuntimeError(
            "no TFLite runtime found. On the Coral board: "
            "sudo apt-get install python3-tflite-runtime python3-pycoral. "
            "On a desktop: pip install ai-edge-litert") from exc


def make_interpreter(model_path, use_tpu):
    """Return an ALLOCATED interpreter for model_path.

    use_tpu=True binds the Edge TPU delegate and requires the *_edgetpu.tflite model.
    use_tpu=False runs the plain CPU model -- which is how the desktop oracle test drives
    the very same code path the board uses.
    """
    path = Path(model_path)
    if not path.is_file():
        raise RuntimeError(f"model not found: {path}")

    Interpreter, load_delegate = _load_runtime()

    if not use_tpu:
        it = Interpreter(model_path=str(path))
        it.allocate_tensors()
        return it

    lib = EDGETPU_LIB.get(platform.system())
    if lib is None:
        raise RuntimeError(f"no Edge TPU delegate is available for {platform.system()}")
    try:
        delegate = load_delegate(lib)
    except (ValueError, OSError) as exc:
        # The single most common board failure. Do NOT fall back to CPU silently -- that
        # would report a plausible FPS number for a run that never touched the TPU, which
        # is the whole thing we are here to measure.
        raise RuntimeError(
            f"could not load the Edge TPU delegate ({lib}): {exc}. Is the board a Coral, "
            f"is libedgetpu installed, and is the device free? Refusing to fall back to "
            f"CPU -- that would silently invalidate the benchmark.") from exc

    it = Interpreter(model_path=str(path), experimental_delegates=[delegate])
    it.allocate_tensors()
    return it
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_coral_interpreter.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add coral/interpreter.py tests/test_coral_interpreter.py
git commit -m "feat(coral): tflite loader that binds the Edge TPU delegate or fails loudly

Never falls back to CPU silently -- a silent fallback would report a plausible
FPS for a run that never touched the TPU.

Co-Authored-By: Claude <claude@anthropic.com>"
```

---

### Task 3: The oracle — prove the decode against Ultralytics before touching hardware

This is the task the whole approach exists for. Everything before it is scaffolding; everything after it is plumbing.

**Files:**
- Create: `tests/test_coral_oracle.py`

**Interfaces:**
- Consumes: `decode.letterbox`, `decode.quantize_input`, `decode.decode`, `interpreter.make_interpreter` (Tasks 1–2).
- Produces: nothing (test-only).

**Note:** this is the one file permitted to import `ultralytics`. It runs on the desktop only.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coral_oracle.py
"""THE ORACLE.

Our hand-written pipeline (letterbox -> quantize -> invoke -> decode -> NMS) must
reproduce what Ultralytics produces from the SAME model on the SAME image.

Why this test carries the phase: the decode fails SILENTLY. A transposed axis, a missing
dequantize, or a class index off by four all yield plausible-looking boxes rather than an
exception. On the board, over SSH, with no reference, every such bug would present as
"the Coral is wrong" and cost hours. Here it costs one second and points straight at the
line. Phase 1 was nearly lost to precisely this failure mode (INT8 score collapse looked
perfectly healthy by every surface check).

Tolerance, not equality: Ultralytics' letterbox rounding and NMS tie-breaks are not
bit-identical to ours. The test exists to catch STRUCTURAL errors -- the ones that
actually happen -- not to pin float noise.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coral"))
import decode as D          # noqa: E402
import interpreter as I     # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CPU_MODEL = PROJECT_ROOT / "deploy" / "rpc_coarse17_int8_320.tflite"
REAL_IMAGES = PROJECT_ROOT / "dataset_real" / "images" / "real_eval"

CONF = 0.25
IOU = 0.45
SIZE = 320

pytest.importorskip("ultralytics")
pytestmark = pytest.mark.skipif(
    not CPU_MODEL.exists() or not REAL_IMAGES.is_dir(),
    reason="needs deploy/rpc_coarse17_int8_320.tflite and dataset_real/images/real_eval")


def _real_image() -> Path:
    images = sorted(REAL_IMAGES.glob("*.jpg"))
    if not images:
        pytest.skip(f"no images in {REAL_IMAGES}")
    return images[0]


def _ours(image_path: Path) -> list[dict]:
    """Our pipeline -- the exact code the board will run, minus the TPU delegate."""
    frame = cv2.imread(str(image_path))
    it = I.make_interpreter(CPU_MODEL, use_tpu=False)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]

    padded, ratio, pad = D.letterbox(frame, SIZE)
    in_scale, in_zp = inp["quantization"]
    it.set_tensor(inp["index"],
                  D.quantize_input(padded, in_scale, in_zp, dtype=inp["dtype"]))
    it.invoke()
    return D.decode(it.get_tensor(out["index"]), out["quantization"],
                    ratio, pad, SIZE, conf_threshold=CONF, iou_threshold=IOU)


def _ultralytics(image_path: Path) -> list[dict]:
    """The oracle: the same .tflite, driven by the library we trust."""
    from ultralytics import YOLO
    model = YOLO(str(CPU_MODEL), task="detect")
    results = model.predict(str(image_path), conf=CONF, iou=IOU, imgsz=SIZE, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "box": tuple(float(v) for v in box.xyxy[0].tolist()),
            })
    detections.sort(key=lambda d: -d["conf"])
    return detections


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    union = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - inter
    return inter / union if union > 0 else 0.0


def test_our_decode_matches_ultralytics_on_a_real_image():
    image = _real_image()
    ours = _ours(image)
    theirs = _ultralytics(image)

    # The image is a real cluttered checkout scene -- a pipeline that finds NOTHING is
    # broken, and would otherwise "match" a broken oracle vacuously.
    assert len(theirs) > 0, f"the oracle itself found nothing in {image.name}"
    assert len(ours) == len(theirs), (
        f"we found {len(ours)} detections, Ultralytics found {len(theirs)}")

    for mine, ref in zip(ours, theirs):
        assert mine["cls"] == ref["cls"], f"class mismatch: {mine} vs {ref}"
        assert mine["conf"] == pytest.approx(ref["conf"], abs=0.02)
        overlap = _iou(mine["box"], ref["box"])
        assert overlap > 0.95, (
            f"box disagreement (IoU {overlap:.3f}): {mine['box']} vs {ref['box']}")


def test_our_pipeline_produces_continuous_confidences_not_a_collapsed_grid():
    # The Phase 1 regression, guarded at the far end of the pipeline: the broken INT8
    # export could only ever emit 0 or 1.77. Real confidences vary continuously.
    ours = _ours(_real_image())
    assert len(ours) > 0
    assert all(0.0 < d["conf"] <= 1.0 for d in ours)
    assert len({round(d["conf"], 3) for d in ours}) > 1
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `python -m pytest tests/test_coral_oracle.py -v`
Expected: **This is the real test of the port.** If the decode is right it passes immediately. If it fails, the failure message names the defect — a class mismatch means a channel-offset bug, a low IoU means a letterbox/scale bug, a count mismatch means an NMS or threshold bug. **Do not proceed to Task 4 until this is green.** Debug `decode.py`, not the test.

- [ ] **Step 3: Fix `coral/decode.py` if the oracle disagrees**

No code given here on purpose — the fix depends on what the assertion reports. Read the failure, form one hypothesis, change one thing, re-run. If boxes are wildly out of frame, print the raw dequantized channel ranges and compare against the known-good profile: **channels 0–1 span roughly 0.01–0.98 (centres), channels 2–3 stay small (extents), channels 4–20 are sparse and mostly zero.**

- [ ] **Step 4: Run the full suite — nothing else may have broken**

Run: `python -m pytest -q`
Expected: all tests pass (112 existing + the new ones).

- [ ] **Step 5: Commit**

```bash
git add tests/test_coral_oracle.py
git commit -m "test(coral): pin the hand-written decode against Ultralytics on the CPU twin

The decode fails silently -- a transposed axis yields plausible boxes, not an
exception. This catches that on the desktop in one second instead of over SSH.

Co-Authored-By: Claude <claude@anthropic.com>"
```

---

### Task 4: The sinks — MJPEG (must work) and HDMI (best-effort)

**Files:**
- Create: `coral/sinks.py`
- Test: `tests/test_coral_sinks.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `MjpegSink(port: int = 8080)` with `.publish(frame: np.ndarray) -> None`, `.url: str`, `.close() -> None`
  - `HdmiSink()` with `.publish(frame) -> None`, `.should_quit() -> bool`, `.close() -> None`, `.available: bool`
  - `build_sinks(display: str, port: int) -> list` for `display in {"stream", "hdmi", "both"}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_coral_sinks.py
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coral"))
import sinks as S  # noqa: E402


@pytest.fixture
def frame():
    return np.full((120, 160, 3), 64, dtype=np.uint8)


def test_mjpeg_sink_serves_a_published_frame_as_a_jpeg(frame):
    sink = S.MjpegSink(port=0)          # port 0 -> OS picks a free one, so tests never clash
    try:
        sink.publish(frame)
        # Connect over the LOOPBACK address, not the 0.0.0.0 the sink advertises: 0.0.0.0
        # is a bind address, and connecting to it is not portable.
        with urllib.request.urlopen(f"http://127.0.0.1:{sink.port}/", timeout=5) as response:
            assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
            chunk = response.read(4096)
        # JPEG SOI marker -- proves a real encoded image went out, not an empty stream.
        assert b"\xff\xd8" in chunk
    finally:
        sink.close()


def test_mjpeg_sink_url_names_the_bound_port(frame):
    sink = S.MjpegSink(port=0)
    try:
        assert sink.url.startswith("http://")
        assert str(sink.port) in sink.url
    finally:
        sink.close()


def test_hdmi_sink_degrades_instead_of_crashing_when_no_display(monkeypatch, frame):
    # Mendel runs Wayland; OpenCV's imshow is built against GTK/X11 and may refuse to open
    # a window. That must cost us the WINDOW, never the RUN -- the MJPEG stream is still
    # carrying the session.
    import cv2

    def explode(*args, **kwargs):
        raise cv2.error("no display")

    monkeypatch.setattr(cv2, "imshow", explode)
    sink = S.HdmiSink()
    sink.publish(frame)             # must not raise
    assert sink.available is False
    sink.publish(frame)             # still must not raise once disabled
    sink.close()


def test_build_sinks_both_returns_a_stream_and_a_display():
    built = S.build_sinks("both", port=0)
    try:
        kinds = {type(s).__name__ for s in built}
        assert kinds == {"MjpegSink", "HdmiSink"}
    finally:
        for s in built:
            s.close()


def test_build_sinks_rejects_an_unknown_display():
    with pytest.raises(ValueError, match="unknown display"):
        S.build_sinks("hologram", port=0)
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `python -m pytest tests/test_coral_sinks.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'sinks'`.

- [ ] **Step 3: Write `coral/sinks.py`**

```python
"""Where annotated frames go. A sink, not a mode -- the loop produces one frame and hands
it to every enabled sink, so --display both is not a special case.

MjpegSink is the one that MUST work: it is a socket, it needs no display server, and it
works while the camera is mounted overhead and the board is headless on Wi-Fi.

HdmiSink is best-effort ON PURPOSE. Mendel runs a Wayland compositor and OpenCV's imshow
is built against GTK/X11, so the window may simply refuse to open. Losing the window must
never lose the run.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"


class MjpegSink:
    """Serves the latest published frame as multipart/x-mixed-replace on :port."""

    def __init__(self, port: int = 8080):
        self._frame = None
        self._lock = threading.Lock()
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.end_headers()
                try:
                    while True:
                        jpeg = sink._latest_jpeg()
                        if jpeg is None:
                            continue
                        self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass        # the browser tab closed; that is not an error

            def log_message(self, *args):
                pass            # do not spam the console with one line per frame

        self._server = ThreadingHTTPServer(("", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://0.0.0.0:{self.port}/"

    def _latest_jpeg(self):
        with self._lock:
            return self._frame

    def publish(self, frame) -> None:
        # Encode once, here, rather than once per connected client. On a Cortex-A35 this
        # encode is a real cost -- it is one of the five stages the runner times.
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("cv2.imencode failed to encode the annotated frame")
        with self._lock:
            self._frame = buf.tobytes()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class HdmiSink:
    """cv2.imshow, if the board will give us a window. Disables itself if not."""

    WINDOW = "EdgeObjectDetector - coarse-17 on Edge TPU"

    def __init__(self):
        self.available = True
        self._quit = False

    def publish(self, frame) -> None:
        if not self.available:
            return
        try:
            cv2.imshow(self.WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._quit = True
        except cv2.error as exc:
            # Say it ONCE, then stop trying. Not silent -- but not fatal either.
            print(f"WARNING: HDMI display unavailable ({exc}); continuing with the stream "
                  f"only. This is expected on Mendel's Wayland compositor.")
            self.available = False

    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        if self.available:
            cv2.destroyAllWindows()


def build_sinks(display: str, port: int) -> list:
    """display in {stream, hdmi, both} -> the sinks to feed."""
    if display == "stream":
        return [MjpegSink(port)]
    if display == "hdmi":
        return [HdmiSink()]
    if display == "both":
        return [MjpegSink(port), HdmiSink()]
    raise ValueError(f"unknown display {display!r}; expected stream, hdmi or both")
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_coral_sinks.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add coral/sinks.py tests/test_coral_sinks.py
git commit -m "feat(coral): MJPEG and best-effort HDMI sinks

MJPEG must work (a socket, no display server). HDMI may not open a window under
Mendel's Wayland compositor -- losing the window must not lose the run.

Co-Authored-By: Claude <claude@anthropic.com>"
```

---

### Task 5: The runner — five timed stages, and a still-image mode so it is testable off-board

**Files:**
- Create: `coral/detect.py`
- Test: `tests/test_coral_detect.py`

**Interfaces:**
- Consumes: `decode` (Task 1), `interpreter` (Task 2), `sinks` (Task 4).
- Produces:
  - `StageTimer()` with `.time(stage: str)` (context manager), `.record(stage: str, elapsed_ms: float) -> None`, `.stats() -> dict[str, dict[str, float]]` (per stage: `median_ms`, `p90_ms`, `n`), `.report() -> str`
  - `load_classes(path) -> list[str]`
  - `Detector(model_path, use_tpu)` with `.detect(frame, conf, iou) -> tuple[list[dict], dict[str, float]]`
  - `draw_detections(frame, detections, names) -> np.ndarray`
  - `benchmark_invoke(detector, frame, runs: int = 50) -> dict` — median/p90 ms of `invoke()` alone, **discarding the first call** (delegate warm-up)
  - `main() -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_coral_detect.py
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
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `python -m pytest tests/test_coral_detect.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'detect'`.

- [ ] **Step 3: Write `coral/detect.py`**

```python
"""Live overhead detector for the Coral Dev Board Mini.

    python3 detect.py                          # TPU + webcam + MJPEG stream on :8080
    python3 detect.py --display both           # ...and an HDMI window if Mendel allows one
    python3 detect.py --cpu --image shot.jpg   # desktop dry-run: no TPU, no camera

WHY THE STAGE TIMINGS EXIST. A single FPS number would mislead you. Four things can each
cap it independently, and they call for opposite responses:
  1. 18 of the model's ops fall back off the TPU onto the Cortex-A35 (96.2% coverage).
  2. The numpy decode chews 2100x21 values on that same modest CPU.
  3. JPEG-encoding each frame for the stream costs more A35 time.
  4. The USB webcam on a bandwidth-limited OTG port may simply not deliver frames faster.
"22 FPS" is consistent with all four and distinguishes none of them. So we time five
stages separately, and separately micro-benchmark invoke() alone.
"""

import argparse
import statistics
import sys
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
IMGSZ = 320


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
            p90_index = min(int(len(ordered) * 0.9), len(ordered) - 1)
            out[stage] = {
                "median_ms": statistics.median(ordered),
                "p90_ms": ordered[p90_index],
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
    p90_index = min(int(len(ordered) * 0.9), len(ordered) - 1)
    return {"median_ms": statistics.median(ordered), "p90_ms": ordered[p90_index],
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
            frame = draw_detections(frame, detections, names)
        except ValueError as exc:
            print(f"ERROR: cannot label detections: {exc}", file=sys.stderr)
            return 1

        with timer.time("sink"):
            for sink in built_sinks:
                sink.publish(frame)

        if any(getattr(s, "should_quit", lambda: False)() for s in built_sinks):
            return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Coral Edge TPU overhead detector.")
    p.add_argument("--model", default=None, help="default: the right model in coral/")
    p.add_argument("--classes", default=str(HERE / "classes.txt"))
    p.add_argument("--camera", type=int, default=0)
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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        print(f"ERROR: could not open camera {args.camera}. Run probe_board.py to see "
              f"whether the OTG webcam enumerated at all.", file=sys.stderr)
        return 1

    built = S.build_sinks(args.display, args.port)
    for sink in built:
        if isinstance(sink, S.MjpegSink):
            print(f"Streaming at http://<board-ip>:{sink.port}/  (Ctrl-C to stop)")
    print("POINT THE CAMERA DOWN at products on a plain surface.")

    timer = StageTimer()
    rc = 1
    try:
        rc = run_loop(detector, cap, names, args.conf, args.iou, built, timer)
    except KeyboardInterrupt:
        rc = 0
    finally:
        cap.release()
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
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_coral_detect.py -v`
Expected: 6 passed.

- [ ] **Step 5: Prove the whole path end-to-end on the desktop, off-board**

Run:
```bash
python coral/detect.py --cpu --model deploy/rpc_coarse17_int8_320.tflite \
  --classes deploy/classes.txt \
  --image dataset_real/images/real_eval/20180824-13-35-55-2.jpg
```
Expected: several detections with **continuous** confidences (e.g. `drink conf 0.94`), a per-stage timing line, and an invoke-only median. If confidences are only ever `0.00` or `1.00`, stop — that is the Phase 1 score collapse and the wrong model is bundled.

- [ ] **Step 6: Commit**

```bash
git add coral/detect.py tests/test_coral_detect.py
git commit -m "feat(coral): board runner with five timed stages and a still-image dry-run

A single FPS number cannot distinguish a TPU bottleneck from CPU-fallback ops,
numpy decode, JPEG encode, or a bandwidth-limited USB camera. So time all five.

Co-Authored-By: Claude <claude@anthropic.com>"
```

---

### Task 6: The bundle — artifacts, probe, README, and a repo the board can actually clone

**Files:**
- Create: `coral/probe_board.py`, `coral/README.md`
- Copy: `export/rpc_coarse17_int8_320_edgetpu.tflite` → `coral/`, `deploy/classes.txt` → `coral/`
- Modify: `.gitignore` (add negations after line 52)

**Interfaces:**
- Consumes: `interpreter.make_interpreter` (Task 2).
- Produces: a `coral/` directory that is complete after `git clone`.

- [ ] **Step 1: Un-ignore the deployment artifacts**

Append to `.gitignore` (the existing `*.tflite` on line 24 and `deploy/classes.txt` on line 52 would otherwise hand the board a repo with **no model and no class names**):

```
# The COMPILED deployment artifacts are the release, not a rebuildable intermediate:
# the board gets them in one `git clone`. 3.9 MB -- far inside GitHub's limits, no LFS.
!coral/*_edgetpu.tflite
!coral/classes.txt
```

- [ ] **Step 2: Copy the artifacts in and verify git will actually track them**

```bash
cp export/rpc_coarse17_int8_320_edgetpu.tflite coral/
cp deploy/classes.txt coral/
git check-ignore -v coral/rpc_coarse17_int8_320_edgetpu.tflite coral/classes.txt
```
Expected: **no output and exit code 1** — meaning neither file is ignored any more. If it prints a matching rule, the negation is in the wrong place; move it after the rule it must override.

- [ ] **Step 3: Write `coral/probe_board.py`**

```python
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
```

- [ ] **Step 4: Write `coral/README.md`**

````markdown
# EdgeObjectDetector — Coral Dev Board Mini

Runs the coarse-17 detector on the Edge TPU from a live overhead USB webcam.

**No Ultralytics, no torch.** Post-processing is hand-written numpy (`decode.py`), pinned
against Ultralytics on the desktop by `tests/test_coral_oracle.py`.

## ⚠️ Do this BEFORE you plug the camera in

The webcam takes the **USB-C OTG port** — which is the port `mdt` talks over. The moment
it is plugged in, `mdt shell` is gone.

**So: bring up Wi-Fi and confirm you can SSH in over it while `mdt` still works.**

```bash
mdt shell                       # while USB is still yours
nmtui                           # join Wi-Fi
ip addr show wlan0              # note the IP
```

Then, from your laptop, **prove SSH works over Wi-Fi** (`ssh mendel@<board-ip>`) before
unplugging USB. Skip this and you lock yourself out of the board.

## Setup (on the board)

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-pycoral
git clone --depth 1 <your-repo-url> EdgeObjectDetector
cd EdgeObjectDetector/coral
python3 probe_board.py
```

`probe_board.py` checks numpy, OpenCV, the model, `classes.txt`, the Edge TPU delegate,
and that the webcam yields a real frame. Fix whatever it reports before going further.

## Run

```bash
python3 detect.py --display stream          # then open http://<board-ip>:8080 on your laptop
python3 detect.py --display both            # ...plus an HDMI window, if Wayland allows one
```

**POINT THE CAMERA DOWN** at products on a plain surface. The model was trained on
top-down RPC checkout scenes; at eye level it is out-of-domain.

Ctrl-C prints the per-stage timing report.

## Reading the timings

Five stages are timed separately because a single FPS number cannot tell you which of
these is the bottleneck:

- **capture** — the USB webcam on a bandwidth-limited OTG port
- **preprocess** — letterbox + quantize, on the Cortex-A35
- **invoke** — the TPU… *plus* the 18 ops (of 475) that fall back onto the A35
- **decode** — 2100×21 of numpy, on the A35
- **sink** — JPEG-encoding each frame for the stream, on the A35

Four of those five are CPU. If `invoke` is small and the rest dominate, the TPU is doing
its job and the A35 is the wall — which is the expected shape of the result.
````

- [ ] **Step 5: Verify the bundle is self-contained the way the board will see it**

Run:
```bash
git add -A coral/ .gitignore
git status --short coral/
```
Expected: **six** files staged — `decode.py`, `interpreter.py`, `sinks.py`, `detect.py`, `probe_board.py`, `README.md` — **plus** `classes.txt` and `rpc_coarse17_int8_320_edgetpu.tflite`. If the `.tflite` is absent, Step 1's negation did not take and the board would clone an empty bundle.

- [ ] **Step 6: Run the full suite one more time**

Run: `python -m pytest -q`
Expected: all pass. No test may have regressed.

- [ ] **Step 7: Commit and push**

```bash
git commit -m "feat(coral): bundle the compiled model, board probe and README

The compiled artifact is the release, not a rebuildable intermediate: the board
gets it in one git clone, so it is committed (3.9 MB) via narrow .gitignore
negations rather than by loosening the rule.

Co-Authored-By: Claude <claude@anthropic.com>"
git push -u origin main
```

---

### Task 7: On the board — the field run

Not code. This is the runbook, and the ordering is load-bearing.

**Files:**
- Create: `coral/results.md`

- [ ] **Step 1: Wi-Fi first, camera second**

Follow `coral/README.md`. **Confirm `ssh mendel@<board-ip>` works over Wi-Fi while `mdt` is still available.** Only then unplug USB and attach the webcam through the OTG adapter.

- [ ] **Step 2: Clone and probe**

```bash
ssh mendel@<board-ip>
sudo apt-get update && sudo apt-get install -y python3-opencv python3-pycoral
git clone --depth 1 <your-repo-url> EdgeObjectDetector
cd EdgeObjectDetector/coral
python3 probe_board.py
```
Expected: six `ok` lines. If **USB webcam** fails, try a powered OTG hub. If **Edge TPU delegate** fails, `python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"` should list a device.

- [ ] **Step 3: Run it**

```bash
python3 detect.py --display stream
```
Open `http://<board-ip>:8080` on the laptop. Point the camera down at real products.
Expected: correct boxes with continuous confidences.

- [ ] **Step 4: Record what the hardware actually did**

Ctrl-C, then paste the stage report into `coral/results.md`, along with the invoke-only
median from a still-image run:

```bash
python3 detect.py --image /path/to/a/captured/frame.jpg
```

Record: per-stage median/p90, end-to-end FPS, invoke-only median, and **which stage
dominates**. That last one is the finding — not the FPS number.

- [ ] **Step 5: Commit the results**

```bash
git add coral/results.md
git commit -m "docs(coral): measured Edge TPU stage timings on the Dev Board Mini

Co-Authored-By: Claude <claude@anthropic.com>"
```

---

## Success Criteria (from the spec)

1. `probe_board.py` passes on the board. → Task 7 Step 2
2. The desktop oracle test is green. → **Task 3**
3. The board runs the live overhead camera and draws correct boxes. → Task 7 Step 3
4. A per-stage breakdown is recorded, with TPU invoke latency isolated. → Task 7 Step 4

**Not** a criterion: any particular FPS figure.
