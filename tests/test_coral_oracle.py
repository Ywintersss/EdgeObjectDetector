"""THE ORACLE.

Our hand-written pipeline (letterbox -> quantize -> invoke -> decode -> NMS) must
reproduce what Ultralytics produces from the SAME model on the SAME images.

Why this test carries the phase: the decode fails SILENTLY. A transposed axis, a missing
dequantize, or a class index off by four all yield plausible-looking boxes rather than an
exception. On the board, over SSH, with no reference, every such bug would present as
"the Coral is wrong" and cost hours. Here it costs a second and points straight at the
line. Phase 1 was nearly lost to precisely this failure mode (an INT8 score collapse
looked perfectly healthy by every surface check and scored mAP 0.000).

WHAT WE ASSERT, AND WHY NOT MORE
--------------------------------
Two pipelines over the same int8 model cannot be expected to agree box-for-box:

* The output tensor is int8 with scale 0.005536. One step in a box channel is
  0.005536 * 320 / 0.1745 ~= 10 PIXELS in an 1834px frame. Nothing finer is expressible.
* Duplicate anchors routinely TIE on that coarse confidence grid. NMS then keeps whichever
  the sort happened to put first -- Ultralytics' stable sort keeps one, our argsort keeps
  the other. Both are correct detections of the same object, one grid step apart.

So a per-box IoU floor of 0.95 is not a correctness bar; it is a demand for precision the
quantized model does not have. Measured over 30 real images / 195 detections (with the
inputs verified bit-identical): counts matched 30/30, zero unmatched detections, MEDIAN
IoU 0.9958, min 0.790, and 100% above 0.75.

The assertions below are chosen against those measurements:

* counts equal + every detection matched to a same-class counterpart -- the STRUCTURAL
  guarantee, and the thing that actually breaks when the decode is wrong.
* median IoU > 0.95 -- guards the decode MATH (measured 0.9958; a systematic coordinate
  error moves the median immediately).
* every box IoU > 0.70 -- guards against a structurally broken box. A transposed axis or a
  missed un-letterboxing scores near 0 here, nowhere near 0.70.

Detections are paired by GREEDY IoU MATCHING within a class, never by sorted-confidence
position: confidences tie on the int8 grid, so a positional zip silently compares box 3
against box 7 and reports a phantom failure.
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
# 5 images keeps the test fast. The thresholds below were calibrated on 30 (195
# detections) -- do not read them as derived from a 5-image run.
N_IMAGES = 5

MIN_MEDIAN_IOU = 0.95    # measured 0.9958 over 195 detections
MIN_BOX_IOU = 0.70       # HEADROOM, not the measured value: worst observed was 0.790.
                         # A transposed axis or a skipped un-letterboxing scores ~0.0,
                         # so the floor has room to spare and still catches them.

pytest.importorskip("ultralytics")
pytestmark = pytest.mark.skipif(
    not CPU_MODEL.exists() or not REAL_IMAGES.is_dir(),
    reason="needs deploy/rpc_coarse17_int8_320.tflite and dataset_real/images/real_eval")


def _real_images() -> list[Path]:
    images = sorted(REAL_IMAGES.glob("*.jpg"))[:N_IMAGES]
    if not images:
        pytest.skip(f"no images in {REAL_IMAGES}")
    return images


def _ours(image_path: Path) -> list[dict]:
    """Our pipeline -- the exact code the board will run, minus the TPU delegate."""
    frame = cv2.imread(str(image_path))
    interp = I.make_interpreter(CPU_MODEL, use_tpu=False)
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    padded, ratio, pad = D.letterbox(frame, SIZE)
    in_scale, in_zp = inp["quantization"]
    interp.set_tensor(inp["index"],
                      D.quantize_input(padded, in_scale, in_zp, dtype=inp["dtype"]))
    interp.invoke()
    return D.decode(interp.get_tensor(out["index"]), out["quantization"],
                    ratio, pad, SIZE, conf_threshold=CONF, iou_threshold=IOU)


def _ultralytics(image_path: Path) -> list[dict]:
    """The oracle: the same .tflite, driven by the library we trust."""
    from ultralytics import YOLO
    model = YOLO(str(CPU_MODEL), task="detect")
    result = model.predict(str(image_path), conf=CONF, iou=IOU, imgsz=SIZE,
                           verbose=False)[0]
    return [{"cls": int(b.cls[0]), "conf": float(b.conf[0]),
             "box": tuple(float(v) for v in b.xyxy[0].tolist())} for b in result.boxes]


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match(ours: list[dict], theirs: list[dict]):
    """Pair each of ours with its best same-class counterpart. -> (ious, n_unmatched).

    Greedy, and never positional: confidences tie on the int8 grid, so zipping by sorted
    confidence compares mismatched boxes and invents failures that are not there.
    """
    pool = list(theirs)
    ious, unmatched = [], 0
    for mine in ours:
        best_iou, best_j = 0.0, None
        for j, ref in enumerate(pool):
            if ref["cls"] != mine["cls"]:
                continue
            overlap = _iou(mine["box"], ref["box"])
            if overlap > best_iou:
                best_iou, best_j = overlap, j
        if best_j is None:
            unmatched += 1          # no same-class counterpart at all
            continue
        pool.pop(best_j)
        ious.append(best_iou)
    return ious, unmatched + len(pool)


def test_our_decode_matches_ultralytics_on_real_images():
    all_ious = []
    for image in _real_images():
        ours, theirs = _ours(image), _ultralytics(image)

        # A pipeline that finds NOTHING would otherwise "agree" with a broken oracle
        # vacuously. These are real cluttered checkout scenes; they contain products.
        assert theirs, f"the oracle itself found nothing in {image.name}"
        assert len(ours) == len(theirs), (
            f"{image.name}: we found {len(ours)} detections, Ultralytics found "
            f"{len(theirs)}")

        ious, unmatched = _match(ours, theirs)
        assert unmatched == 0, (
            f"{image.name}: {unmatched} detection(s) had no same-class counterpart -- "
            f"a class-channel bug, not a rounding difference")
        assert min(ious) > MIN_BOX_IOU, (
            f"{image.name}: worst box IoU {min(ious):.3f} <= {MIN_BOX_IOU} -- that is a "
            f"structurally wrong box (transposed axis / un-letterboxing), not quantization")
        all_ious.extend(ious)

    median = float(np.median(all_ious))
    assert median > MIN_MEDIAN_IOU, (
        f"median IoU {median:.4f} <= {MIN_MEDIAN_IOU} across {len(all_ious)} detections "
        f"-- the decode has a systematic coordinate error")


def test_our_input_tensor_is_bit_identical_to_ultralytics():
    """The strongest guarantee in the suite: we feed the model the EXACT same bytes.

    With this pinned, the model's output is deterministic and therefore identical too, so
    any downstream disagreement can only come from our decode -- which is what makes the
    box assertions above meaningful rather than a wash of two compounding differences.

    This is also what caught the real bug: we ROUNDED where Ultralytics TRUNCATES, which
    silently shifted whole boxes by a full output quantization step (~10px).
    """
    from ultralytics import YOLO

    image = _real_images()[0]
    model = YOLO(str(CPU_MODEL), task="detect")
    model.predict(str(image), conf=CONF, iou=IOU, imgsz=SIZE, verbose=False)

    interp = model.predictor.model.interpreter
    captured = {}
    original_set_tensor = interp.set_tensor

    def spy(index, value):
        captured["x"] = np.array(value)
        return original_set_tensor(index, value)

    interp.set_tensor = spy
    try:
        model.predict(str(image), conf=CONF, iou=IOU, imgsz=SIZE, verbose=False)
    finally:
        interp.set_tensor = original_set_tensor

    theirs = captured["x"]
    frame = cv2.imread(str(image))
    inp = interp.get_input_details()[0]
    in_scale, in_zp = inp["quantization"]
    padded, _, _ = D.letterbox(frame, SIZE)
    ours = D.quantize_input(padded, in_scale, in_zp, dtype=inp["dtype"])

    assert ours.shape == theirs.shape
    differing = int((ours != theirs).sum())
    assert differing == 0, (
        f"{differing} of {ours.size} input bytes differ from Ultralytics. Our "
        f"preprocessing has drifted from the pipeline the model's accuracy was measured "
        f"through.")


def test_our_pipeline_produces_continuous_confidences_not_a_collapsed_grid():
    # The Phase 1 regression, guarded at the far end of the pipeline: the broken INT8
    # export could only ever emit 0 or 1.77. Real confidences vary continuously.
    ours = _ours(_real_images()[0])
    assert len(ours) > 0
    assert all(0.0 < d["conf"] <= 1.0 for d in ours)
    assert len({round(d["conf"], 3) for d in ours}) > 1
