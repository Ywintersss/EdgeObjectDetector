# tests/test_build_real_coarse.py
import json
import sys

import pytest
import yaml

import build_real_coarse as B


def test_scene_key_strips_camera_id():
    assert B.scene_key("20181025-15-09-20-161.jpg") == "20181025-15-09-20"
    assert B.scene_key("20181025-14-40-13-146.jpg") == "20181025-14-40-13"


def test_split_by_scene_no_scene_overlap():
    # two scenes, several camera frames each
    names = [f"20181025-15-09-20-{c}.jpg" for c in range(6)] + \
            [f"20181025-14-40-13-{c}.jpg" for c in range(6)]
    parts = B.split_by_scene(names, seed=0)
    keys = {k: {B.scene_key(n) for n in v} for k, v in parts.items()}
    # no scene key appears in more than one split
    all_pairs = [("real_ft", "real_eval"), ("real_ft", "reserve"), ("real_eval", "reserve")]
    for a, b in all_pairs:
        assert keys[a].isdisjoint(keys[b]), f"scene leak between {a} and {b}"
    # every input basename lands in exactly one split
    assert sum(len(v) for v in parts.values()) == len(names)


def test_split_by_scene_deterministic():
    names = [f"2018-00-00-{s:02d}-{c}.jpg" for s in range(20) for c in range(3)]
    assert B.split_by_scene(names, seed=0) == B.split_by_scene(names, seed=0)


def test_subsample_deterministic_and_sized():
    names = [f"img_{i}.jpg" for i in range(100)]
    a = B.subsample(names, 10, seed=0)
    assert len(a) == 10
    assert a == B.subsample(names, 10, seed=0)
    assert set(a).issubset(set(names))


def test_coco_to_yolo_importable_and_converts(tmp_path):
    from preprocessing import coco_to_yolo
    j = tmp_path / "mini.json"
    j.write_text(json.dumps({
        "categories": [{"id": 1, "name": "1_puffed_food"}, {"id": 2, "name": "2_puffed_food"}],
        "images": [{"id": 10, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [{"image_id": 10, "category_id": 2, "bbox": [10, 20, 30, 40]}],
    }))
    out = tmp_path / "labels"
    coco_to_yolo(str(j), str(out))
    txt = (out / "a.txt").read_text().strip()
    # category_id 2 is the 2nd category -> YOLO index 1; box normalized
    assert txt.split()[0] == "1"
