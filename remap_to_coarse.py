"""
remap_to_coarse.py — Non-destructively relabel the synthetic RPC dataset from
200 fine SKUs to 17 coarse categories.

Each RPC class name is '<id>_<category>'. This collapses all SKUs that share a
category into one class, rewriting ONLY the leading class index of each YOLO
label row. The original 200-way labels are never modified; the coarse tree is a
separate, disposable, fully-reversible artifact.

Usage:
    python remap_to_coarse.py --src-yaml dataset_synth.yml --out dataset_synth_coarse --verify
"""

import re

_ID_PREFIX = re.compile(r"^\d+_")


def build_coarse_mapping(names: list[str]) -> tuple[dict[int, int], list[str]]:
    """Map 200 fine SKU indices to their coarse category indices.

    Returns (old2new, coarse_names). coarse_names is the alphabetically sorted
    unique set of category suffixes, giving a stable, deterministic index order.
    """
    coarse_names = sorted({_ID_PREFIX.sub("", n) for n in names})
    cat_index = {c: i for i, c in enumerate(coarse_names)}
    old2new = {i: cat_index[_ID_PREFIX.sub("", n)] for i, n in enumerate(names)}
    return old2new, coarse_names


def remap_label_line(line: str, old2new: dict[int, int]) -> str:
    """Rewrite the class index (token 0) of a YOLO row; leave box coords intact."""
    if not line.strip():
        return line  # blank / whitespace-only lines pass through unchanged
    parts = line.split()
    parts[0] = str(old2new[int(parts[0])])
    return " ".join(parts)
