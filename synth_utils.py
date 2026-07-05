"""synth_utils.py — pure helpers for synthetic checkout-scene generation.

Kept free of file I/O (except the yaml writer) so every function is unit-testable
with tiny in-memory numpy arrays.
"""
from __future__ import annotations

import random

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


class ClassBalancedSampler:
    """Draw (class_id, item) picks, always preferring the least-used class.

    Keeps per-class draw counts so that across a whole generation run every class
    appears a similar number of times, even when the cut-out library is skewed.
    """

    def __init__(self, class_to_items: dict[int, list[str]], seed: int = 0):
        if not class_to_items:
            raise ValueError("class_to_items must not be empty")
        self._pool = {c: list(items) for c, items in class_to_items.items() if items}
        if not self._pool:
            raise ValueError("every class had an empty item list")
        self._counts = {c: 0 for c in self._pool}
        self._rng = random.Random(seed)

    def sample(self, k: int) -> list[tuple[int, str]]:
        picks: list[tuple[int, str]] = []
        for _ in range(k):
            min_count = min(self._counts.values())
            candidates = [c for c, n in self._counts.items() if n == min_count]
            cls = self._rng.choice(candidates)
            item = self._rng.choice(self._pool[cls])
            self._counts[cls] += 1
            picks.append((cls, item))
        return picks
