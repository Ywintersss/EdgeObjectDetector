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


def test_materialize_split_hardlinks_images_and_remaps_labels(tmp_path):
    # source images + 200-class labels for two basenames
    src_img = tmp_path / "src_images"; src_img.mkdir()
    src_lbl = tmp_path / "src_labels"; src_lbl.mkdir()
    for base in ("a", "b"):
        (src_img / f"{base}.jpg").write_bytes(b"JPEGDATA")
        (src_lbl / f"{base}.txt").write_text("5 0.1 0.2 0.3 0.4\n")
    (src_img / "orphan.jpg").write_bytes(b"X")  # image with no label -> skipped
    old2new = {5: 2}

    out_img = tmp_path / "out" / "images" / "real_ft"
    out_lbl = tmp_path / "out" / "labels" / "real_ft"
    n_img, n_lbl = B.materialize_split(["a", "b", "orphan"], src_img, src_lbl,
                                       out_img, out_lbl, old2new)
    assert (n_img, n_lbl) == (2, 2)  # orphan (no label) is skipped
    # label: only class index changed, coords byte-identical
    assert (out_lbl / "a.txt").read_text().strip() == "2 0.1 0.2 0.3 0.4"
    # image present (hardlink or copy) with same bytes
    assert (out_img / "a.jpg").read_bytes() == b"JPEGDATA"
    # out image dir is a REAL directory, not a reparse point
    assert out_img.is_dir() and not out_img.is_symlink()


def test_names_from_json(tmp_path):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"categories": [
        {"id": 1, "name": "1_puffed_food"}, {"id": 2, "name": "5_drink"}],
        "images": [], "annotations": []}))
    assert B.names_from_json(j) == ["1_puffed_food", "5_drink"]


def test_write_blend_yaml_structure(tmp_path):
    out = tmp_path / "blend.yml"
    B.write_blend_yaml(out,
                       real_ft_images=tmp_path / "dataset_real/images/real_ft",
                       synth_coarse_images=tmp_path / "dataset_synth_coarse/images/train",
                       studio_images=tmp_path / "studio_coarse/images/train",
                       real_eval_images=tmp_path / "dataset_real/images/real_eval",
                       coarse_names=["alcohol", "candy"])
    doc = yaml.safe_load(out.read_text())
    assert isinstance(doc["train"], list) and len(doc["train"]) == 3
    assert doc["train"][0].endswith("dataset_real/images/real_ft")
    assert doc["val"].endswith("dataset_real/images/real_eval")
    assert doc["nc"] == 2
    assert doc["names"] == {0: "alcohol", 1: "candy"}


def test_verify_no_leak_raises_on_shared_scene(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"; a.mkdir(); b.mkdir()
    (a / "20181025-15-09-20-1.jpg").write_bytes(b"x")
    (b / "20181025-15-09-20-2.jpg").write_bytes(b"x")   # SAME scene key -> leak
    with pytest.raises(AssertionError):
        B.verify_no_leak({"a": a, "b": b})
