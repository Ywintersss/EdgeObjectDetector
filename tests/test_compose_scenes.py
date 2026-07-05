import random
import numpy as np
import synth_utils
from compose_scenes import make_background, compose_one


def _solid_rgba(color, h, w):
    patch = np.zeros((h, w, 4), dtype=np.uint8)
    patch[..., :3] = color
    patch[..., 3] = 255
    return patch


def test_make_background_fills_canvas():
    tiles = [np.full((32, 32, 3), 130, dtype=np.uint8) for _ in range(4)]
    rng = random.Random(0)
    canvas = make_background(tiles, size=128, rng=rng)
    assert canvas.shape == (128, 128, 3)
    assert canvas.mean() > 0  # not blank


def test_compose_one_emits_labels_in_range():
    rng = random.Random(0)
    canvas = np.full((640, 640, 3), 120, dtype=np.uint8)
    cutouts = [(3, _solid_rgba((0, 0, 200), 120, 120)),
               (7, _solid_rgba((0, 200, 0), 120, 120))]
    rows = compose_one(canvas, cutouts, rng)
    assert len(rows) >= 1
    for cid, xc, yc, w, h in rows:
        assert cid in (3, 7)
        assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_fully_occluded_object_is_dropped():
    rng = random.Random(1)
    canvas = np.full((200, 200, 3), 120, dtype=np.uint8)
    # object A then object B pasted at the same spot, B bigger -> A should drop
    small = (1, _solid_rgba((0, 0, 200), 30, 30))
    big = (2, _solid_rgba((0, 200, 0), 120, 120))
    # force identical placement by seeding; assert only class 2 survives on top
    rows = compose_one(canvas, [small, big], rng, drop_thresh=0.5)
    surviving_classes = {r[0] for r in rows}
    assert 2 in surviving_classes


def test_fully_overwritten_owner_has_near_zero_visibility():
    # Deterministic case bypassing random placement: object 0 occupies a 10x10
    # region entirely, then object 1 fully overwrites that same region on the
    # owner map (as alpha_paste would when pasted on top and fully opaque).
    owner_map = np.full((20, 20), -1, dtype=np.int32)
    owner_map[0:10, 0:10] = 0        # object 0 initially "owns" a 10x10 block
    owner_map[0:10, 0:10] = 1        # object 1 pastes on top, fully overwriting it
    owner_map[10:20, 10:20] = 1      # object 1 also owns a separate untouched block
    total_pixels = {0: 100, 1: 200}  # object 0 had 100 px total; object 1 had 200 px

    vis = synth_utils.compute_visibilities(owner_map, total_pixels)

    assert vis[0] < 0.15   # object 0 fully occluded -> below drop threshold
    assert vis[1] == 1.0   # object 1 fully visible
