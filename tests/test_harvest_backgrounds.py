import random
import numpy as np
from harvest_backgrounds import sample_empty_patches, _patch_free


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


def test_patch_free_rejects_overlap_on_every_side():
    boxes = [(40, 40, 60, 60)]
    s = 10

    # Fully clear of the box.
    assert _patch_free(0, 0, s, boxes) is True, "patch at (0,0) does not touch the box; should be free"

    # Overlapping from the LEFT: patch spans x=[35,45), y=[45,55) -> x-overlap [40,45), y-overlap [45,55).
    assert _patch_free(35, 45, s, boxes) is False, "patch overlapping box from the left must be rejected"

    # Overlapping from the RIGHT: patch spans x=[55,65), y=[45,55) -> x-overlap [55,60), y-overlap [45,55).
    assert _patch_free(55, 45, s, boxes) is False, "patch overlapping box from the right must be rejected"

    # Overlapping from ABOVE: patch spans x=[45,55), y=[35,45) -> x-overlap [45,55), y-overlap [40,45).
    assert _patch_free(45, 35, s, boxes) is False, "patch overlapping box from above must be rejected"

    # Overlapping from BELOW: patch spans x=[45,55), y=[55,65) -> x-overlap [45,55), y-overlap [55,60).
    assert _patch_free(45, 55, s, boxes) is False, "patch overlapping box from below must be rejected"

    # Edge-adjacent (not overlapping): patch spans x=[30,40), y=[45,55); x+s == 40 == box x1,
    # half-open interval means touching edges do not count as overlap.
    assert _patch_free(30, 45, s, boxes) is True, "edge-adjacent patch (half-open, no overlap) should be free"
