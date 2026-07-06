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
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

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


def load_names_from_yaml(src_yaml: Path) -> tuple[Path, list[str]]:
    """Read the source dataset yaml -> (absolute dataset path, ordered class names)."""
    doc = yaml.safe_load(Path(src_yaml).read_text(encoding="utf-8"))
    raw = doc["names"]
    if isinstance(raw, dict):
        names = [raw[k] for k in sorted(raw, key=int)]  # dict form: order by index
    else:
        names = list(raw)
    return Path(doc["path"]), names


def remap_tree(src_labels: Path, dst_labels: Path,
               old2new: dict[int, int]) -> tuple[int, int]:
    """Rewrite every *.txt under src_labels into the mirrored path under dst_labels.

    Only class indices change; box coords are untouched. Unreadable/corrupt files
    are skipped and tallied so one bad file never aborts the batch.
    """
    src_labels = Path(src_labels)
    dst_labels = Path(dst_labels)
    written = skipped = 0
    for src in sorted(src_labels.glob("*.txt")):
        try:
            lines = src.read_text(encoding="utf-8").splitlines()
            out = "\n".join(remap_label_line(ln, old2new) for ln in lines)
            if lines:
                out += "\n"  # preserve trailing newline for non-empty files
            dst = dst_labels / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(out, encoding="utf-8")
            written += 1
        except Exception as exc:  # noqa: BLE001 — skip corrupt file, keep going
            print(f"  WARN skipped {src.name}: {exc}", file=sys.stderr)
            skipped += 1
    return written, skipped


def link_images(src_images: Path, dst_images: Path) -> None:
    """Make dst_images resolve to src_images without copying (symlink; junction fallback)."""
    src_images = Path(src_images)
    dst_images = Path(dst_images)
    if dst_images.exists():
        return
    dst_images.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src_images, dst_images, target_is_directory=True)
    except OSError:
        # Windows without Developer Mode: a junction needs no admin rights.
        subprocess.run(["cmd", "/c", "mklink", "/J",
                        str(dst_images), str(src_images)], check=True)


def write_coarse_yaml(out_root: Path, coarse_names: list[str], out_path: Path) -> None:
    """Write a YOLO dataset yaml for the coarse tree (UTF-8, ASCII hyphen)."""
    out_root = Path(out_root).resolve()
    lines = [
        "# dataset_synth_coarse.yaml - 17 coarse categories (derived; non-destructive)",
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(coarse_names)}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(coarse_names)]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify(out_yaml: Path, expected_nc: int) -> None:
    """Sanity-check the generated tree without training: nc, index range, histogram."""
    from collections import Counter
    doc = yaml.safe_load(Path(out_yaml).read_text(encoding="utf-8"))
    assert doc["nc"] == expected_nc, f"nc {doc['nc']} != {expected_nc}"
    names = doc["names"]
    root = Path(doc["path"])
    hist: Counter = Counter()
    max_idx = -1
    for split in ("train", "val"):
        lbl_dir = root / "labels" / split
        for f in lbl_dir.glob("*.txt"):
            for ln in f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    c = int(ln.split()[0])
                    max_idx = max(max_idx, c)
                    hist[c] += 1
    assert max_idx < expected_nc, f"class index {max_idx} >= nc {expected_nc}"
    print(f"VERIFY OK: nc={expected_nc}, max class index={max_idx}")
    print("Per-category box counts:")
    for c in sorted(hist):
        label = names[c] if isinstance(names, dict) else names[c]
        print(f"  {c:2d} {label:20s} {hist[c]}")


def main() -> int:
    p = argparse.ArgumentParser(description="Relabel synthetic RPC data 200->17 (non-destructive).")
    p.add_argument("--src-yaml", default="dataset_synth.yml")
    p.add_argument("--out", default="dataset_synth_coarse")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    src_yaml = Path(args.src_yaml).resolve()
    ds_path, names = load_names_from_yaml(src_yaml)
    old2new, coarse_names = build_coarse_mapping(names)
    print(f"{len(names)} SKUs -> {len(coarse_names)} coarse categories")

    out_root = Path(args.out).resolve()
    # link images (no copy) and remap both label splits
    link_images(ds_path / "images", out_root / "images")
    total_w = total_s = 0
    for split in ("train", "val"):
        w, s = remap_tree(ds_path / "labels" / split, out_root / "labels" / split, old2new)
        print(f"  {split}: {w} labels written, {s} skipped")
        total_w += w
        total_s += s

    out_yaml = Path(f"{out_root.name}.yml").resolve()
    write_coarse_yaml(out_root, coarse_names, out_yaml)
    print(f"Wrote {out_yaml.name} ({total_w} labels, {total_s} skipped)")

    if args.verify:
        _verify(out_yaml, len(coarse_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
