"""
build_real_coarse.py — Non-destructively turn the real RPC test scenes into a
coarse-17 training set, blend with the existing synthetic + studio coarse sets,
and emit ready-to-train yamls. Does NOT start training.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from preprocessing import coco_to_yolo
from remap_to_coarse import build_coarse_mapping, remap_label_line


def scene_key(filename: str) -> str:
    """RPC scene id = filename stem with the trailing '-<cameraId>' removed.

    '20181025-15-09-20-161.jpg' -> '20181025-15-09-20' (groups all camera/burst
    frames of one physical tray so they never split across train/eval).
    """
    stem = Path(filename).stem
    return stem.rsplit("-", 1)[0]


def split_by_scene(basenames, ratios=(0.75, 0.125, 0.125), seed=0):
    """Partition basenames into real_ft / real_eval / reserve BY SCENE (no leakage).

    All frames sharing a scene_key are assigned to the same split. Deterministic
    under a fixed seed.
    """
    groups = defaultdict(list)
    for n in basenames:
        groups[scene_key(n)].append(n)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_ft = int(len(keys) * ratios[0])
    n_eval = int(len(keys) * ratios[1])
    buckets = {
        "real_ft": keys[:n_ft],
        "real_eval": keys[n_ft:n_ft + n_eval],
        "reserve": keys[n_ft + n_eval:],
    }
    return {name: [f for k in ks for f in groups[k]] for name, ks in buckets.items()}


def subsample(basenames, n, seed=0):
    """Deterministic size-n sample (or all if n >= len)."""
    pool = sorted(basenames)
    if n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)


def hardlink(src, dst) -> None:
    """Per-file hardlink src->dst (mkdir parents). Symlink fallback across volumes.

    Never links a whole directory: every path component stays a real dir so
    Ultralytics' Path.resolve() cannot collapse it back to the source labels.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def materialize_split(basenames, src_images, src_labels, out_images, out_labels,
                      old2new) -> tuple[int, int]:
    """Hardlink each basename's image and write its coarse-remapped label.

    A basename with no matching *.txt is skipped (image-only frames are ignored).
    Returns (n_images_linked, n_labels_written).
    """
    src_images, src_labels = Path(src_images), Path(src_labels)
    out_images, out_labels = Path(out_images), Path(out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    n_img = n_lbl = 0
    for base in basenames:
        stem = Path(base).stem
        lbl = src_labels / f"{stem}.txt"
        if not lbl.exists():
            continue
        img = src_images / f"{stem}.jpg"
        if img.exists():
            hardlink(img, out_images / img.name)
            n_img += 1
        lines = lbl.read_text(encoding="utf-8").splitlines()
        out = "\n".join(remap_label_line(ln, old2new) for ln in lines)
        if lines:
            out += "\n"
        (out_labels / f"{stem}.txt").write_text(out, encoding="utf-8")
        n_lbl += 1
    return n_img, n_lbl


def names_from_json(json_path) -> list[str]:
    """Canonical 200 class names in the SAME order coco_to_yolo 0-indexes them."""
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return [c["name"] for c in doc["categories"]]


def _yaml_names_block(coarse_names):
    return {i: n for i, n in enumerate(coarse_names)}


def write_blend_yaml(out_path, real_ft_images, synth_coarse_images, studio_images,
                     real_eval_images, coarse_names) -> None:
    """Blend yaml: multi-path train (real + synth-coarse + studio-coarse), val=real_eval."""
    doc = {
        "train": [Path(real_ft_images).resolve().as_posix(),
                  Path(synth_coarse_images).resolve().as_posix(),
                  Path(studio_images).resolve().as_posix()],
        "val": Path(real_eval_images).resolve().as_posix(),
        "nc": len(coarse_names),
        "names": _yaml_names_block(coarse_names),
    }
    Path(out_path).write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def write_single_item_yaml(out_path, studio_eval_images, studio_root, coarse_names) -> None:
    """Single-item eval yaml: val = held-out studio slice (coarse-17)."""
    doc = {
        "path": Path(studio_root).resolve().as_posix(),
        "train": Path(studio_eval_images).resolve().as_posix(),  # unused; YOLO needs a key
        "val": Path(studio_eval_images).resolve().as_posix(),
        "nc": len(coarse_names),
        "names": _yaml_names_block(coarse_names),
    }
    Path(out_path).write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def verify_no_leak(image_dirs) -> None:
    """Assert no scene_key is shared across any two image dirs (train/eval/reserve)."""
    keyset = {}
    for name, d in image_dirs.items():
        keyset[name] = {scene_key(p.name) for p in Path(d).glob("*.jpg")}
    names = list(keyset)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = keyset[names[i]] & keyset[names[j]]
            assert not overlap, f"scene leak {names[i]}<->{names[j]}: {sorted(overlap)[:3]}"
