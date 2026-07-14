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


def test_quantize_input_truncates_like_ultralytics_rather_than_rounding():
    # White lands on 126, NOT 127 -- and that is deliberate. The model's scale is a hair
    # ABOVE 1/255, so 1.0 / scale = 254.99998, + (-128) = 126.99998. Ultralytics casts
    # with .astype (which truncates) -> 126. Rounding would give 127.
    # We match Ultralytics byte-for-byte because every accuracy number we have for this
    # model was measured through ITS pipeline. See quantize_input's docstring.
    white = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    q = D.quantize_input(white, scale=0.003921568859368563, zero_point=-128)
    assert q.shape == (1, SIZE, SIZE, 3)
    assert q.dtype == np.int8
    assert (q == 126).all()


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
    assert (q[0, :, :, 2] == 126).all()                  # RGB channel 2 == blue (126: truncated)
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


def test_decode_maps_each_box_channel_to_the_right_axis():
    # Plant ONE anchor: an ASYMMETRIC, OFF-CENTRE box with distinct channel values.
    # Every existing test uses a centred square (cx == cy == 0.5, w == h == 0.5),
    # so a cx/cy or w/h channel swap would have passed all of them silently.
    # This test uses cx=0.25, cy=0.75, w=0.6, h=0.2 (all different) to catch any
    # permutation of the four box channels.
    out = _empty_output()
    anchor = 500
    for channel, value in ((0, 0.25), (1, 0.75), (2, 0.6), (3, 0.2)):  # cx, cy, w, h
        out[0, channel, anchor] = _quantize(value)
    out[0, 4 + 11, anchor] = _quantize(0.85)  # class 11

    dets = D.decode(out, OUT_QUANT, ratio=1.0, pad=(0, 0), size=SIZE)

    assert len(dets) == 1
    assert dets[0]["cls"] == 11
    assert dets[0]["conf"] == pytest.approx(0.85, abs=0.01)
    x1, y1, x2, y2 = dets[0]["box"]
    # cx=0.25, cy=0.75, w=0.6, h=0.2 (normalized) with size=320:
    #   cx_px = 80, cy_px = 240, w_px = 192, h_px = 64
    #   x1 = cx_px - w_px/2 = 80 - 96 = -16
    #   y1 = cy_px - h_px/2 = 240 - 32 = 208
    #   x2 = cx_px + w_px/2 = 80 + 96 = 176
    #   y2 = cy_px + h_px/2 = 240 + 32 = 272
    assert (x1, y1, x2, y2) == pytest.approx((-16, 208, 176, 272), abs=2)
