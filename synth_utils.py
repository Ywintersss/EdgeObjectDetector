"""synth_utils.py — pure helpers for synthetic checkout-scene generation.

Kept free of file I/O (except the yaml writer) so every function is unit-testable
with tiny in-memory numpy arrays.
"""
from __future__ import annotations

import random

import cv2
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


def rotate_rgba(rgba: np.ndarray, angle_deg: float,
                rng: "random.Random | None" = None) -> np.ndarray:
    """Rotate an RGBA cut-out about its center, expanding so nothing is clipped."""
    h, w = rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    mat[0, 2] += (new_w / 2.0) - center[0]
    mat[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(rgba, mat, (new_w, new_h),
                          flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def random_placement(canvas_wh: tuple[int, int], obj_wh: tuple[int, int],
                     rng: "random.Random") -> tuple[int, int]:
    """Random top-left (x,y) keeping the object's center on-canvas (edges may overhang)."""
    cw, ch = canvas_wh
    ow, oh = obj_wh
    # center must be within [0, cw] x [0, ch]  ->  x in [-ow/2, cw-ow/2]
    x = rng.randint(-ow // 2, cw - ow // 2)
    y = rng.randint(-oh // 2, ch - oh // 2)
    return x, y


def alpha_paste(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int,
                owner_map: np.ndarray, owner_id: int) -> None:
    """Alpha-blend rgba (H,W,4 BGRA) onto canvas (BGR) at (x,y); stamp owner_map."""
    ch, cw = canvas.shape[:2]
    oh, ow = rgba.shape[:2]
    # intersection of the paste rect with the canvas
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + ow), min(ch, y + oh)
    if x1 >= x2 or y1 >= y2:
        return
    # corresponding region inside the rgba patch
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    patch = rgba[sy1:sy2, sx1:sx2]
    alpha = (patch[..., 3:4].astype(np.float32)) / 255.0
    fg = patch[..., :3].astype(np.float32)
    bg = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (alpha * fg + (1 - alpha) * bg).astype(np.uint8)
    opaque = patch[..., 3] > 0
    owner_slice = owner_map[y1:y2, x1:x2]
    owner_slice[opaque] = owner_id


def compute_visibilities(owner_map: np.ndarray,
                         total_pixels: dict[int, int]) -> dict[int, float]:
    """visible/total per owner_id, where visible = pixels still owned at the end."""
    ids, counts = np.unique(owner_map, return_counts=True)
    visible = {int(i): int(c) for i, c in zip(ids, counts) if i >= 0}
    return {oid: visible.get(oid, 0) / total for oid, total in total_pixels.items()
            if total > 0}


def write_synth_yaml(dst_root, names_block_lines: list[str], out_path) -> None:
    """Write a dataset yaml (UTF-8, ASCII hyphen) for the synthetic tree."""
    from pathlib import Path
    dst_root = Path(dst_root)
    header = [
        "# dataset_synth.yaml - synthetic checkout scenes + real val",
        f"path: {dst_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "nc: 200",
    ]
    text = "\n".join(header + list(names_block_lines)) + "\n"
    Path(out_path).write_text(text, encoding="utf-8")
