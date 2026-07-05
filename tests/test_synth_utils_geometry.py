import numpy as np
from synth_utils import denormalize_box, normalize_box, mask_to_bbox


def test_denormalize_box_center():
    # centered box covering half the width/height of a 100x200 image
    assert denormalize_box((0.5, 0.5, 0.5, 0.5), 100, 200) == (25, 50, 75, 150)


def test_denormalize_box_clips_to_bounds():
    # box spilling past edges is clipped, never negative or > size
    assert denormalize_box((0.0, 0.0, 0.4, 0.4), 100, 100) == (0, 0, 20, 20)


def test_normalize_round_trips():
    box = (0.3, 0.6, 0.2, 0.4)
    x1, y1, x2, y2 = denormalize_box(box, 640, 640)
    back = normalize_box(x1, y1, x2, y2, 640, 640)
    assert all(abs(a - b) < 1e-3 for a, b in zip(box, back))


def test_mask_to_bbox_tight():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:8] = 1
    assert mask_to_bbox(mask) == (3, 2, 8, 5)


def test_mask_to_bbox_empty_returns_none():
    assert mask_to_bbox(np.zeros((10, 10), dtype=np.uint8)) is None
