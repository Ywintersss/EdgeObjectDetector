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
