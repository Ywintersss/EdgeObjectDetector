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

from __future__ import annotations

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

    TRUNCATES rather than rounds, matching Ultralytics byte-for-byte
    (nn/backends/litert.py: `im = (im / scale + zero_point).astype(dtype)`; .astype
    truncates). Rounding is arguably the more correct quantizer, and it was tried -- but
    the model's scale is a hair above 1/255, so `pixel / 255 / scale` lands just under the
    integer and the two disagree on ~0.1% of pixels. That was enough to shift whole boxes
    by a full output quantization step (~10 px in a 1834 px frame). Every accuracy number
    we have for this model -- mAP@50 0.978 -- was measured through Ultralytics' pipeline,
    so the board must reproduce THAT pipeline, not a variant we never evaluated. Verified:
    this yields a bit-identical input tensor (0 of 307,200 pixels differ).
    """
    if letterboxed_bgr.dtype != np.uint8:
        raise ValueError(
            f"quantize_input expects a BGR uint8 frame, got dtype {letterboxed_bgr.dtype} "
            f"-- this looks like an already-normalized float frame; dividing it by 255 "
            f"again would silently produce a near-zero input with no error")
    rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    info = np.iinfo(dtype)
    # Clip before the cast: an out-of-range float would WRAP under .astype (Ultralytics
    # does not clip; it is safe there only because its inputs are already in range).
    return np.clip(rgb / scale + zero_point, info.min, info.max).astype(dtype)[None]


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
