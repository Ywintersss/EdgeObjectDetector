"""synth_utils.py — pure helpers for synthetic checkout-scene generation.

Kept free of file I/O (except the yaml writer) so every function is unit-testable
with tiny in-memory numpy arrays.
"""
from __future__ import annotations

import numpy as np


def denormalize_box(box: tuple[float, float, float, float],
                    img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """YOLO normalized (xc,yc,w,h) -> pixel (x1,y1,x2,y2), clipped to image."""
    xc, yc, w, h = box
    x1 = int(round((xc - w / 2) * img_w))
    y1 = int(round((yc - h / 2) * img_h))
    x2 = int(round((xc + w / 2) * img_w))
    y2 = int(round((yc + h / 2) * img_h))
    x1 = max(0, min(img_w, x1)); x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h, y1)); y2 = max(0, min(img_h, y2))
    return x1, y1, x2, y2


def normalize_box(x1: int, y1: int, x2: int, y2: int,
                  img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Pixel corners -> YOLO normalized (xc,yc,w,h)."""
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight bbox (x1,y1,x2,y2) of nonzero pixels; x2,y2 exclusive. None if empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
