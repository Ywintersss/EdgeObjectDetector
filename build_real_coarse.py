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
    """RPC scene id = (capture date, station id) — groups every burst frame of one tray.

    Filenames are 'YYYYMMDD-HH-MM-SS-<station>'. One physical arrangement is captured as
    a ~3-shot burst seconds apart, so the timestamp (HH-MM-SS) is UNIQUE per image while
    the trailing station id is REUSED across dates. The scene is therefore the
    (date, station) pair: '20180824-15-44-39-474.jpg' -> '20180824-474', and its burst
    siblings '...-44-47-474' / '...-44-56-474' map to the same key.

    NOTE: the earlier version keyed on the timestamp (stem minus the last field), which is
    unique per shot — it scattered each scene's burst frames across train/eval and caused
    100% train->eval leakage. Grouping on (date, station) is what prevents that.
    """
    parts = Path(filename).stem.split("-")
    return f"{parts[0]}-{parts[-1]}"


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
    """Single-item eval yaml: val = held-out studio slice (coarse-17).

    NOTE: this eval slice is sampled from dataset_640/train and is NOT guaranteed
    disjoint from the synthetic cut-out sources in the blend; treat single-item
    numbers as a soft signal, not a clean held-out metric."""
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


def _is_complete(out) -> bool:
    """True once a build finished (sentinel written at the very end of main)."""
    return (Path(out) / ".complete").exists()


def main() -> int:
    p = argparse.ArgumentParser(description="Build real-scene coarse-17 blend (non-destructive).")
    p.add_argument("--test-json", default="instances_test2019.json")
    p.add_argument("--test-images", default="dataset/images/test")
    p.add_argument("--out", default="dataset_real")
    p.add_argument("--studio-images", default="dataset_640/images/train")
    p.add_argument("--studio-labels", default="dataset_640/labels/train")
    p.add_argument("--synth-coarse-images", default="dataset_synth_coarse/images/train")
    p.add_argument("--studio-out", default="studio_coarse")
    p.add_argument("--studio-train-n", type=int, default=10000)
    p.add_argument("--studio-eval-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--dry", action="store_true", help="parse args and exit (test hook)")
    args = p.parse_args()
    if args.dry:
        return 0

    out = Path(args.out).resolve()
    if _is_complete(out):
        print(f"{out} already built (.complete present); remove the tree to rebuild. Skipping.")
        return 0

    # 1) 200-class mapping derived from the json categories (matches coco_to_yolo order)
    names = names_from_json(args.test_json)
    old2new, coarse_names = build_coarse_mapping(names)
    print(f"{len(names)} SKUs -> {len(coarse_names)} coarse categories")

    # 2) convert test json -> flat 200-class labels in a FRESH staging dir
    staging = out / "_staging_labels"
    if staging.exists() and any(staging.glob("*.txt")):
        print(f"ERROR: staging {staging} not empty; remove it and rerun.", file=sys.stderr)
        return 1
    staging.mkdir(parents=True, exist_ok=True)
    coco_to_yolo(str(args.test_json), str(staging))

    # 3) scene-grouped split over basenames that have a label
    basenames = [p.stem for p in staging.glob("*.txt")]
    parts = split_by_scene(basenames, seed=args.seed)
    for name, bases in parts.items():
        ni, nl = materialize_split(bases, args.test_images, staging,
                                   out / "images" / name, out / "labels" / name, old2new)
        print(f"  real {name}: {ni} images, {nl} labels")

    # 4) studio slice (coarse) from dataset_640: disjoint train + held-out eval
    studio_out = Path(args.studio_out).resolve()
    if not studio_out.exists():
        studio_bases = [p.stem for p in Path(args.studio_labels).glob("*.txt")]
        studio_train = subsample(studio_bases, args.studio_train_n, seed=args.seed)
        remaining = sorted(set(studio_bases) - set(studio_train))
        studio_eval = subsample(remaining, args.studio_eval_n, seed=args.seed)
        for name, bases in (("train", studio_train), ("eval", studio_eval)):
            ni, nl = materialize_split(bases, args.studio_images, args.studio_labels,
                                       studio_out / "images" / name,
                                       studio_out / "labels" / name, old2new)
            print(f"  studio {name}: {ni} images, {nl} labels")
    else:
        print(f"{studio_out} exists; reusing.")

    # 5) compose yamls
    write_blend_yaml(Path("dataset_real_blend.yml"),
                     out / "images" / "real_ft",
                     Path(args.synth_coarse_images),
                     studio_out / "images" / "train",
                     out / "images" / "real_eval",
                     coarse_names)
    write_single_item_yaml(Path("eval_single_item.yml"),
                           studio_out / "images" / "eval", studio_out, coarse_names)
    print("Wrote dataset_real_blend.yml + eval_single_item.yml")

    # 6) verify (optional)
    if args.verify:
        verify_no_leak({k: out / "images" / k for k in ("real_ft", "real_eval", "reserve")})
        print("VERIFY OK: no scene leak across real_ft/real_eval/reserve")

    # 7) write completion sentinel (make idempotency check reachable on re-run)
    (out / ".complete").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
