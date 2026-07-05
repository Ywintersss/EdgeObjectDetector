"""Shared pytest fixtures: tiny in-memory images so tests are fast and real."""
import numpy as np
import pytest


@pytest.fixture
def blank_canvas():
    """A 100x100 BGR gray canvas."""
    return np.full((100, 100, 3), 127, dtype=np.uint8)


@pytest.fixture
def red_rgba():
    """A 20x10 fully-opaque red RGBA patch (BGRA order to match cv2)."""
    patch = np.zeros((20, 10, 4), dtype=np.uint8)
    patch[..., 2] = 255   # red channel (BGR)
    patch[..., 3] = 255   # alpha
    return patch
