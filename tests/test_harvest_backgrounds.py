import random
import numpy as np
from harvest_backgrounds import sample_empty_patches


def _overlaps(px, box):
    x, y, s = px
    bx1, by1, bx2, by2 = box
    return not (x + s <= bx1 or x >= bx2 or y + s <= by1 or y >= by2)


def test_patches_avoid_boxes():
    img = np.full((200, 200, 3), 120, dtype=np.uint8)
    boxes = [(0, 0, 100, 200)]  # left half occupied
    rng = random.Random(0)
    patches = sample_empty_patches(img, boxes, patch=20, max_patches=10, rng=rng)
    assert len(patches) > 0
    assert all(p.shape == (20, 20, 3) for p in patches)


def test_no_patches_when_fully_covered():
    img = np.full((50, 50, 3), 120, dtype=np.uint8)
    boxes = [(0, 0, 50, 50)]
    rng = random.Random(0)
    assert sample_empty_patches(img, boxes, patch=20, max_patches=5, rng=rng) == []
