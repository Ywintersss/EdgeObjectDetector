import numpy as np
from extract_cutouts import grabcut_cutout


def test_grabcut_extracts_foreground_block():
    # gray background with a bright square "product" in the middle
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    img[25:55, 25:55] = (200, 180, 160)
    out = grabcut_cutout(img, box_px=(20, 20, 60, 60), iters=3)
    assert out is not None
    assert out.shape[2] == 4                 # BGRA
    assert out[..., 3].max() == 255          # has opaque product pixels
    # cropped roughly to the product, not the whole 80x80 frame
    assert out.shape[0] <= 60 and out.shape[1] <= 60


def test_grabcut_returns_none_on_empty_box():
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    assert grabcut_cutout(img, box_px=(10, 10, 10, 10)) is None  # zero-area box
