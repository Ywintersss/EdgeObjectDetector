"""harvest_backgrounds.py — sample empty checkout-tray patches from val images.

Only product-FREE tray texture is sampled (zero bbox overlap), so no val product
pixels or labels ever enter training. These tiles become the canvas background for
synthetic scenes.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from synth_utils import denormalize_box

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _patch_free(x, y, s, boxes_px):
    for bx1, by1, bx2, by2 in boxes_px:
        if not (x + s <= bx1 or x >= bx2 or y + s <= by1 or y >= by2):
            return False
    return True


def sample_empty_patches(img, boxes_px, patch, max_patches, rng):
    """Up to max_patches `patch`x`patch` BGR crops not overlapping any box."""
    h, w = img.shape[:2]
    if w < patch or h < patch:
        return []
    out = []
    attempts = 0
    while len(out) < max_patches and attempts < max_patches * 30:
        attempts += 1
        x = rng.randint(0, w - patch)
        y = rng.randint(0, h - patch)
        if _patch_free(x, y, patch, boxes_px):
            out.append(img[y:y + patch, x:x + patch].copy())
    return out


def _read_boxes_px(label_path: Path, w: int, h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        box = tuple(float(v) for v in parts[1:])
        boxes.append(denormalize_box(box, w, h))
    return boxes


def parse_args():
    p = argparse.ArgumentParser(description="Harvest empty tray background patches.")
    p.add_argument("--src", default="dataset/images/val")
    p.add_argument("--labels", default="dataset/labels/val")
    p.add_argument("--out", default="backgrounds")
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--count", type=int, default=4000, help="Total patches to collect.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    root = PROJECT_ROOT
    src = (root / args.src) if not os.path.isabs(args.src) else Path(args.src)
    lbl = (root / args.labels) if not os.path.isabs(args.labels) else Path(args.labels)
    out = (root / args.out) if not os.path.isabs(args.out) else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"ERROR: no val images at {src}", file=sys.stderr)
        return 1
    per_image = max(1, args.count // len(files))
    saved = 0
    for f in files:
        if saved >= args.count:
            break
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = _read_boxes_px(lbl / (f.stem + ".txt"), w, h)
        for patch in sample_empty_patches(img, boxes, args.patch, per_image, rng):
            cv2.imwrite(str(out / f"bg_{saved:06d}.png"), patch)
            saved += 1
            if saved >= args.count:
                break
    print(f"Done: saved {saved} background patches -> {out}")
    if saved == 0:
        print("ERROR: harvested zero patches.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
