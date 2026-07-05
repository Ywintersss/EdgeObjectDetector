"""compose_scenes.py — synthesize labeled multi-object checkout scenes.

Pastes GrabCut cut-outs (class-balanced) onto harvested tray backgrounds, records
a YOLO box per visible product, and mixes in a fraction of original single-item
images. Output is a ready-to-train dataset_synth/ tree + dataset_synth.yml.
"""
import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from synth_utils import (ClassBalancedSampler, alpha_paste, compute_visibilities,
                         mask_to_bbox, normalize_box, random_placement,
                         rotate_rgba, write_synth_yaml)

PROJECT_ROOT = Path(__file__).resolve().parent


def make_background(bg_tiles, size, rng):
    """Tile random background patches into a size x size canvas, blur the seams."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if not bg_tiles:
        canvas[:] = 128
        return canvas
    th, tw = bg_tiles[0].shape[:2]
    for y in range(0, size, th):
        for x in range(0, size, tw):
            tile = rng.choice(bg_tiles)
            hh = min(th, size - y)
            ww = min(tw, size - x)
            canvas[y:y + hh, x:x + ww] = tile[:hh, :ww]
    return cv2.GaussianBlur(canvas, (5, 5), 0)


def _scale_rgba(bgra, target_side, rng, min_scale, max_scale, canvas_size):
    """Scale a cut-out so its longest side is a random fraction of the canvas."""
    frac = rng.uniform(min_scale, max_scale)
    target = max(8, int(canvas_size * frac))
    h, w = bgra.shape[:2]
    s = target / max(h, w)
    new = cv2.resize(bgra, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA)
    return new


def compose_one(canvas, cutouts, rng, min_scale=0.15, max_scale=0.40,
                drop_thresh=0.15):
    """Paste cut-outs; return YOLO rows for objects still >= drop_thresh visible."""
    size = canvas.shape[0]
    owner = np.full((size, size), -1, dtype=np.int32)
    placed = []          # (owner_id, class_id, x1,y1,x2,y2, total_pixels)
    total_pixels = {}
    for oid, (cid, bgra) in enumerate(cutouts):
        scaled = _scale_rgba(bgra, None, rng, min_scale, max_scale, size)
        angle = rng.uniform(-25, 25)
        rot = rotate_rgba(scaled, angle)
        oh, ow = rot.shape[:2]
        x, y = random_placement((size, size), (ow, oh), rng)
        alpha_mask = rot[..., 3] > 0
        total = int(alpha_mask.sum())
        if total == 0:
            continue
        alpha_paste(canvas, rot, x, y, owner, oid)
        # box from the alpha mask position on the canvas, clipped
        local_bbox = mask_to_bbox(alpha_mask)
        lx1, ly1, lx2, ly2 = local_bbox
        gx1 = max(0, x + lx1); gy1 = max(0, y + ly1)
        gx2 = min(size, x + lx2); gy2 = min(size, y + ly2)
        if gx2 <= gx1 or gy2 <= gy1:
            continue
        placed.append((oid, cid, gx1, gy1, gx2, gy2))
        total_pixels[oid] = total
    vis = compute_visibilities(owner, total_pixels)
    rows = []
    for oid, cid, x1, y1, x2, y2 in placed:
        if vis.get(oid, 0.0) < drop_thresh:
            continue
        xc, yc, w, h = normalize_box(x1, y1, x2, y2, size, size)
        rows.append((cid, xc, yc, w, h))
    return rows


def _load_cutout_index(cutouts_root: Path):
    """Return {class_id: [png paths]} from cutouts/<class_id>/*.png."""
    index = defaultdict(list)
    for cls_dir in sorted(cutouts_root.iterdir()):
        if cls_dir.is_dir() and cls_dir.name.isdigit():
            for png in cls_dir.glob("*.png"):
                index[int(cls_dir.name)].append(str(png))
    return {c: v for c, v in index.items() if v}


def _load_names_block(dataset_yml: Path):
    """Return the `names:` block lines from the existing dataset.yml."""
    lines = dataset_yml.read_text(encoding="utf-8", errors="ignore").splitlines()
    block, capture = [], False
    for line in lines:
        if line.startswith("names:"):
            capture = True
        if capture:
            block.append(line)
    return block


def main():
    args = parse_args()
    root = PROJECT_ROOT
    cutouts_root = (root / args.cutouts)
    bg_root = (root / args.backgrounds)
    out_root = (root / args.out)
    rng = random.Random(args.seed)

    index = _load_cutout_index(cutouts_root)
    if not index:
        print(f"ERROR: no cut-outs under {cutouts_root}", file=sys.stderr)
        return 1
    bg_tiles = [cv2.imread(str(p)) for p in sorted(bg_root.glob("*.png"))]
    bg_tiles = [t for t in bg_tiles if t is not None]
    if not bg_tiles:
        print(f"ERROR: no backgrounds under {bg_root}", file=sys.stderr)
        return 1

    img_out = out_root / "images" / "train"
    lbl_out = out_root / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    sampler = ClassBalancedSampler(index, seed=args.seed)
    for i in range(args.num):
        k = rng.randint(args.min_objs, args.max_objs)
        picks = sampler.sample(k)
        cutouts = []
        for cid, path in picks:
            bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if bgra is not None and bgra.ndim == 3 and bgra.shape[2] == 4:
                cutouts.append((cid, bgra))
        if not cutouts:
            continue
        canvas = make_background(bg_tiles, args.size, rng)
        rows = compose_one(canvas, cutouts, rng)
        if not rows:
            continue
        stem = f"synth_{i:06d}"
        cv2.imwrite(str(img_out / f"{stem}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        (lbl_out / f"{stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in rows)
            + "\n")
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{args.num} scenes")

    _mix_single_items(root, out_root, args.single_item_frac, rng)
    _link_val(root, out_root)
    names_block = _load_names_block(root / "dataset.yml")
    write_synth_yaml(out_root, names_block, root / "dataset_synth.yml")
    print(f"Done. Synthetic dataset at {out_root}; yaml at dataset_synth.yml")
    return 0


def _mix_single_items(root, out_root, frac, rng):
    """Copy a fraction of original single-item train images+labels into synth train."""
    if frac <= 0:
        return
    src_img = root / "dataset_640" / "images" / "train"
    src_lbl = root / "dataset_640" / "labels" / "train"
    imgs = sorted(src_img.glob("*.jpg"))
    n = int(len(imgs) * frac)
    for f in rng.sample(imgs, min(n, len(imgs))):
        lbl = src_lbl / (f.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copy2(f, out_root / "images" / "train" / f.name)
        shutil.copy2(lbl, out_root / "labels" / "train" / lbl.name)
    print(f"  mixed in {n} single-item images (frac={frac})")


def _link_val(root, out_root):
    """Point synth val at the real resized val by copying (Windows-safe, no symlink)."""
    for sub in ("images/val", "labels/val", "images/test"):
        src = root / "dataset_640" / sub
        dst = out_root / sub
        if dst.exists() or not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)
    print("  copied real val/test into synth tree")


def parse_args():
    p = argparse.ArgumentParser(description="Compose synthetic checkout scenes.")
    p.add_argument("--cutouts", default="cutouts")
    p.add_argument("--backgrounds", default="backgrounds")
    p.add_argument("--out", default="dataset_synth")
    p.add_argument("--num", type=int, default=20000)
    p.add_argument("--min-objs", type=int, default=3)
    p.add_argument("--max-objs", type=int, default=15)
    p.add_argument("--single-item-frac", type=float, default=0.1)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
