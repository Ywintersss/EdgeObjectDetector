"""extract_cutouts.py — GrabCut RGBA cut-outs of each isolated RPC product.

For every single-item training image (exactly one YOLO box), GrabCut is seeded
from that box to segment the product off the white turntable, and the result is
saved as a cropped BGRA PNG under cutouts/<class_id>/.
"""
import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from synth_utils import denormalize_box, mask_to_bbox

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def grabcut_cutout(img: np.ndarray, box_px, iters: int = 5):
    """Segment the product inside box_px via GrabCut; return cropped BGRA or None."""
    x1, y1, x2, y2 = box_px
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    bbox = mask_to_bbox(fg)
    if bbox is None:
        return None
    bx1, by1, bx2, by2 = bbox
    bgr = img[by1:by2, bx1:bx2]
    alpha = fg[by1:by2, bx1:bx2]
    bgra = np.dstack([bgr, alpha])
    return bgra


def _read_single_box(label_path: Path):
    """Return the (class_id, (xc,yc,w,h)) of the first box, or None."""
    try:
        line = label_path.read_text().splitlines()[0].split()
    except (IndexError, OSError):
        return None
    if len(line) != 5:
        return None
    cid = int(float(line[0]))
    box = tuple(float(v) for v in line[1:])
    return cid, box


def _worker(task):
    """Process one image -> write one cut-out PNG. Returns (status, class_id|msg)."""
    img_path, label_path, out_root = task
    try:
        parsed = _read_single_box(Path(label_path))
        if parsed is None:
            return ("error", f"no box: {label_path}")
        cid, box = parsed
        img = cv2.imread(img_path)
        if img is None:
            return ("error", f"unreadable: {img_path}")
        h, w = img.shape[:2]
        box_px = denormalize_box(box, w, h)
        cut = grabcut_cutout(img, box_px)
        if cut is None:
            return ("error", f"empty mask: {img_path}")
        out_dir = Path(out_root) / str(cid)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(img_path).stem + ".png")
        cv2.imwrite(str(out_path), cut)
        return ("ok", cid)
    except Exception as exc:  # never abort the pool on one bad file
        return ("error", f"{img_path}: {exc}")


def build_tasks(src_root: Path, out_root: Path, per_class: int, limit):
    """Enumerate (img, label, out_root) tasks, capping per-class camera angles."""
    img_dir = src_root / "images" / "train"
    lbl_dir = src_root / "labels" / "train"
    seen = defaultdict(int)
    tasks = []
    files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if limit:
        files = files[:limit]
    for f in files:
        label = lbl_dir / (f.stem + ".txt")
        if not label.exists():
            continue
        parsed = _read_single_box(label)
        if parsed is None:
            continue
        cid = parsed[0]
        if seen[cid] >= per_class:
            continue
        seen[cid] += 1
        tasks.append((str(f), str(label), str(out_root)))
    return tasks


def parse_args():
    p = argparse.ArgumentParser(description="Extract GrabCut RGBA product cut-outs.")
    p.add_argument("--src", default="dataset", help="Source dataset root.")
    p.add_argument("--out", default="cutouts", help="Cut-out output root.")
    p.add_argument("--per-class", type=int, default=40, help="Max camera angles/class.")
    p.add_argument("--workers", type=int, default=0, help="Processes (0=auto).")
    p.add_argument("--limit", type=int, default=None, help="Cap images (for trials).")
    return p.parse_args()


def main():
    args = parse_args()
    src_root = (PROJECT_ROOT / args.src) if not os.path.isabs(args.src) else Path(args.src)
    out_root = (PROJECT_ROOT / args.out) if not os.path.isabs(args.out) else Path(args.out)
    workers = args.workers or max(1, (os.cpu_count() or 4))
    tasks = build_tasks(src_root, out_root, args.per_class, args.limit)
    if not tasks:
        print(f"ERROR: no tasks built from {src_root}", file=sys.stderr)
        return 1
    print(f"Extracting {len(tasks)} cut-outs -> {out_root} (workers={workers})")
    ok = 0
    per_class = defaultdict(int)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            status, payload = fut.result()
            if status == "ok":
                ok += 1
                per_class[payload] += 1
            else:
                print(f"  WARN {payload}", file=sys.stderr)
            if i % 2000 == 0:
                print(f"  {i}/{len(tasks)} done")
    classes_covered = len(per_class)
    print(f"Done: {ok} cut-outs across {classes_covered}/200 classes.")
    if classes_covered < 200:
        print(f"WARNING: {200 - classes_covered} classes have no cut-outs.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
