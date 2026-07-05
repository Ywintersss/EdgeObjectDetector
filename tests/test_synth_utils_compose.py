import random
import numpy as np
from synth_utils import (rotate_rgba, random_placement, alpha_paste,
                         compute_visibilities)


def test_alpha_paste_draws_and_owns(blank_canvas, red_rgba):
    owner = np.full(blank_canvas.shape[:2], -1, dtype=np.int32)
    alpha_paste(blank_canvas, red_rgba, x=10, y=5, owner_map=owner, owner_id=0)
    # pasted 20x10 red block: red channel maxed inside, owner stamped
    assert blank_canvas[5, 10, 2] == 255
    assert (owner == 0).sum() == 20 * 10
    # outside the paste region untouched
    assert owner[0, 0] == -1


def test_alpha_paste_clips_at_border(blank_canvas, red_rgba):
    owner = np.full(blank_canvas.shape[:2], -1, dtype=np.int32)
    # paste near bottom-right corner so it overhangs; must not raise
    alpha_paste(blank_canvas, red_rgba, x=95, y=95, owner_map=owner, owner_id=1)
    assert (owner == 1).sum() == 5 * 5  # only the on-canvas 5x5 corner drawn


def test_later_paste_overwrites_owner(blank_canvas, red_rgba):
    owner = np.full(blank_canvas.shape[:2], -1, dtype=np.int32)
    alpha_paste(blank_canvas, red_rgba, 10, 10, owner, 0)
    alpha_paste(blank_canvas, red_rgba, 12, 10, owner, 1)  # overlaps object 0
    vis = compute_visibilities(owner, {0: 200, 1: 200})
    assert vis[1] == 1.0            # object 1 fully on top
    assert vis[0] < 1.0            # object 0 partially covered


def test_rotate_rgba_expands_and_preserves_content(red_rgba):
    out = rotate_rgba(red_rgba, 90.0)
    # 90-degree rotation swaps dimensions (allow +/-1 for rounding)
    assert abs(out.shape[0] - 10) <= 1 and abs(out.shape[1] - 20) <= 1
    assert out[..., 3].max() == 255  # some opaque product pixels survive


def test_random_placement_keeps_center_on_canvas():
    rng = random.Random(0)
    for _ in range(50):
        x, y = random_placement((640, 640), (100, 80), rng)
        cx, cy = x + 50, y + 40
        assert 0 <= cx <= 640 and 0 <= cy <= 640
