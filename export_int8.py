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


def gate_verdict(baseline_map: float, model_map: float) -> str:
    """Classify the INT8 accuracy drop per the spec's quantization gate.

    Drop is in PERCENTAGE POINTS of mAP@50-95: <2 clean, 2-5 acceptable, >5 RED FLAG.
    A RED FLAG means the first suspect is the calibration data, not the input size.
    """
    # Round before comparing: raw float64 arithmetic can drift an exact 5.0pt
    # drop to 5.000000000000004, which would wrongly fail the <=5.0 test and
    # trigger a false RED FLAG. Precision to 1e-9 is far tighter than the
    # spec's 0.1pt reporting granularity, so it only absorbs float noise.
    drop_points = round((baseline_map - model_map) * 100.0, 9)
    if drop_points < 2.0:
        return "clean"
    if drop_points <= 5.0:
        return "acceptable"
    return "RED FLAG"


def build_report_table(rows: list[dict]) -> str:
    """Render the size-vs-accuracy table. FAILED sizes are shown, never dropped."""
    lines = [
        "# INT8 Export Report",
        "",
        f"FP32 baseline — cluttered mAP@50-95: **{BASELINE['cluttered_map']:.3f}**, "
        f"single-item: **{BASELINE['single_map']:.3f}**",
        "",
        "Latency is desktop CPU, for comparing sizes against each other only — "
        "it is NOT a Coral FPS forecast.",
        "",
        "| Size | Cluttered mAP@50 | Cluttered mAP@50-95 | Δ vs FP32 (pts) | Gate | "
        "Single mAP@50-95 | File | CPU latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -x["size"]):
        if r["status"] != "ok":
            lines.append(
                f"| {r['size']} | FAILED | FAILED | — | FAILED | — | — | {r['error']} |")
            continue
        delta = (r["cluttered_map"] - BASELINE["cluttered_map"]) * 100.0
        verdict = gate_verdict(BASELINE["cluttered_map"], r["cluttered_map"])
        lines.append(
            f"| {r['size']} | {r['cluttered_map50']:.3f} | {r['cluttered_map']:.3f} | "
            f"{delta:+.1f} | {verdict} | {r['single_map']:.3f} | "
            f"{r['bytes'] / 1e6:.1f} MB | {r['latency_ms']:.1f} ms |")
    return "\n".join(lines) + "\n"
