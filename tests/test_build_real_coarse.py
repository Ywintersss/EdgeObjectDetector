# tests/test_build_real_coarse.py
import json
import sys

import pytest
import yaml

import build_real_coarse as B


def test_scene_key_groups_by_date_and_station():
    # RPC filename = YYYYMMDD-HH-MM-SS-<station>. A physical tray is one (date, station)
    # burst captured as ~3 shots seconds apart; the station id is REUSED across dates.
    # The scene key must be (date, station), NOT the timestamp.
    assert B.scene_key("20180824-15-44-39-474.jpg") == "20180824-474"
    # burst siblings of the SAME arrangement (same date+station, different seconds) -> same key
    assert B.scene_key("20180824-15-44-39-474.jpg") == B.scene_key("20180824-15-44-56-474.jpg")
    # different station on the same date -> different scene
    assert B.scene_key("20180824-15-50-57-480.jpg") == "20180824-480"
    # same station on a DIFFERENT date -> different scene (station id is reused daily)
    assert B.scene_key("20180825-15-44-39-474.jpg") == "20180825-474"


def test_split_by_scene_no_scene_overlap():
    # 8 physical scenes = (date, station), each a 3-frame burst (differ only in seconds)
    names = [f"20180824-15-44-{sec:02d}-{station}.jpg"
             for station in range(8)
             for sec in (39, 47, 56)]
    parts = B.split_by_scene(names, seed=0)
    keys = {k: {B.scene_key(n) for n in v} for k, v in parts.items()}
    # no scene key appears in more than one split
    all_pairs = [("real_ft", "real_eval"), ("real_ft", "reserve"), ("real_eval", "reserve")]
    for a, b in all_pairs:
        assert keys[a].isdisjoint(keys[b]), f"scene leak between {a} and {b}"
    # all 3 frames of a scene land in the SAME split (no burst gets split)
    for station in range(8):
        holders = [name for name, v in parts.items()
                   if any(B.scene_key(n) == f"20180824-{station}" for n in v)]
        assert len(holders) == 1, f"scene 20180824-{station} split across {holders}"
    # every input basename lands in exactly one split
    assert sum(len(v) for v in parts.values()) == len(names)


def test_split_by_scene_deterministic():
    names = [f"20180824-15-{sc:02d}-{fr:02d}-{station}.jpg"
             for station in range(20) for sc, fr in ((10, 0), (10, 5), (10, 9))]
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


def test_write_single_item_yaml_structure(tmp_path):
    out = tmp_path / "single_item.yml"
    studio_eval = tmp_path / "studio_coarse/images/eval"
    studio_root = tmp_path / "studio_coarse"
    B.write_single_item_yaml(out,
                             studio_eval_images=studio_eval,
                             studio_root=studio_root,
                             coarse_names=["alcohol", "candy"])
    doc = yaml.safe_load(out.read_text())
    assert doc["val"].endswith("studio_coarse/images/eval")
    assert doc["nc"] == 2
    assert doc["names"] == {0: "alcohol", 1: "candy"}
    assert doc["path"].endswith("studio_coarse")


def test_verify_no_leak_raises_on_shared_scene(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"; a.mkdir(); b.mkdir()
    # same (date, station)=20180824-474, different seconds = burst siblings of ONE scene
    (a / "20180824-15-44-39-474.jpg").write_bytes(b"x")
    (b / "20180824-15-44-56-474.jpg").write_bytes(b"x")   # same scene key -> leak
    with pytest.raises(AssertionError):
        B.verify_no_leak({"a": a, "b": b})


def test_main_parses_args(monkeypatch):
    called = {}
    monkeypatch.setattr(sys, "argv", ["build_real_coarse.py",
                                      "--test-json", "instances_test2019.json",
                                      "--out", "dataset_real", "--dry"])
    # --dry short-circuits before any filesystem work
    rc = B.main()
    assert rc == 0


def test_materialize_split_remaps_multibox_label(tmp_path):
    src_img = tmp_path / "si"; src_img.mkdir()
    src_lbl = tmp_path / "sl"; src_lbl.mkdir()
    (src_img / "m.jpg").write_bytes(b"J")
    (src_lbl / "m.txt").write_text("5 0.1 0.2 0.3 0.4\n7 0.5 0.6 0.1 0.1\n")
    out_img = tmp_path / "o" / "images" / "real_ft"
    out_lbl = tmp_path / "o" / "labels" / "real_ft"
    B.materialize_split(["m"], src_img, src_lbl, out_img, out_lbl, {5: 2, 7: 3})
    lines = (out_lbl / "m.txt").read_text().splitlines()
    assert lines == ["2 0.1 0.2 0.3 0.4", "3 0.5 0.6 0.1 0.1"]


def test_is_complete_sentinel(tmp_path):
    out = tmp_path / "dataset_real"
    out.mkdir()
    assert B._is_complete(out) is False
    (out / ".complete").touch()
    assert B._is_complete(out) is True
