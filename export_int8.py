"""
export_int8.py — Export the trained coarse-17 YOLO11n to INT8 TFLite at several
input sizes, measure the accuracy cost of each, and assemble a laptop deploy bundle.

Usage:
    python export_int8.py --sizes 320                  # validate the toolchain first (fast)
    python export_int8.py --sizes 640,448,320          # full sweep -> export/report.md
    python export_int8.py --bundle 320                 # assemble deploy/ for the laptop

INT8 calibration reads REAL cluttered images (dataset_real_blend.yml). Calibrating on
the wrong distribution silently costs accuracy that looks like "quantization is lossy".
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

# FP32 baseline measured on the held-out splits (see the design spec).
BASELINE = {"cluttered_map50": 0.995, "cluttered_map": 0.879,
            "single_map50": 0.995, "single_map": 0.945}

EXPECTED_CLASSES = 17


def parse_sizes(text: str) -> list[int]:
    """Parse '640,448,320' -> [640, 448, 320]. Raises ValueError on anything not a positive int."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"no sizes parsed from {text!r}")
    sizes = []
    for p in parts:
        try:
            n = int(p)
        except ValueError as exc:
            raise ValueError(f"invalid size {p!r} in {text!r}") from exc
        if n <= 0:
            raise ValueError(f"size must be positive, got {n}")
        sizes.append(n)
    return sizes


def load_class_names(yaml_path) -> list[str]:
    """Class names ordered by CLASS INDEX (not dict insertion order).

    Ultralytics writes names as {0: 'alcohol', 1: 'candy', ...}. Iterating that dict in
    insertion order would mislabel every detection while crashing nothing — so we sort
    explicitly by index.
    """
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"dataset yaml not found: {p}")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    names = doc["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)
