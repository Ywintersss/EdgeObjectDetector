import pytest
from remap_to_coarse import build_coarse_mapping, remap_label_line


def test_build_coarse_mapping_collapses_suffixes():
    names = ["1_puffed_food", "2_puffed_food", "3_tissue", "4_alcohol"]
    old2new, coarse = build_coarse_mapping(names)
    # 3 unique categories, alphabetically ordered -> stable indices
    assert coarse == ["alcohol", "puffed_food", "tissue"]
    assert old2new == {0: 1, 1: 1, 2: 2, 3: 0}


def test_build_coarse_mapping_all_indices_in_range():
    # realistic-ish: two categories
    names = [f"{i+1}_cat_a" for i in range(5)] + [f"{i+6}_cat_b" for i in range(3)]
    old2new, coarse = build_coarse_mapping(names)
    assert len(coarse) == 2
    assert set(old2new) == set(range(8))
    assert all(0 <= v < len(coarse) for v in old2new.values())


def test_remap_label_line_rewrites_only_class_index():
    old2new = {33: 9}
    line = "33 0.051562 0.223438 0.103125 0.271875"
    assert remap_label_line(line, old2new) == "9 0.051562 0.223438 0.103125 0.271875"


def test_remap_label_line_preserves_coords_exactly():
    old2new = {117: 2}
    line = "117 0.216406 0.141406 0.214062 0.282813"
    out = remap_label_line(line, old2new)
    assert out.split()[1:] == line.split()[1:]  # coords untouched


def test_remap_label_line_passes_blank_through():
    assert remap_label_line("", {0: 0}) == ""
    assert remap_label_line("   ", {0: 0}) == "   "


import os
from pathlib import Path
import yaml
from remap_to_coarse import (
    load_names_from_yaml, remap_tree, write_coarse_yaml, build_coarse_mapping,
)


def _write(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_load_names_from_yaml_dict_form(tmp_path):
    y = tmp_path / "d.yml"
    _write(y, "path: " + str(tmp_path).replace("\\", "/") +
           "\ntrain: images/train\nnc: 2\nnames:\n  0: 1_alcohol\n  1: 2_tissue\n")
    ds_path, names = load_names_from_yaml(y)
    assert names == ["1_alcohol", "2_tissue"]
    assert ds_path == tmp_path


def test_remap_tree_rewrites_and_mirrors(tmp_path):
    src = tmp_path / "src" / "labels" / "train"
    dst = tmp_path / "dst" / "labels" / "train"
    _write(src / "a.txt", "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.3 0.3\n")
    _write(src / "b.txt", "1 0.4 0.4 0.1 0.1\n")
    # names: index0 -> cat_a, index1 -> cat_b  => two coarse classes, identity-ish
    old2new = {0: 0, 1: 1}
    written, skipped = remap_tree(src, dst, old2new)
    assert (written, skipped) == (2, 0)
    assert (dst / "a.txt").read_text(encoding="utf-8").splitlines() == \
        ["0 0.5 0.5 0.2 0.2", "1 0.1 0.1 0.3 0.3"]


def test_remap_tree_collapses_two_skus_into_one_class(tmp_path):
    src = tmp_path / "s" / "labels" / "val"
    dst = tmp_path / "d" / "labels" / "val"
    _write(src / "x.txt", "0 0.5 0.5 0.2 0.2\n1 0.6 0.6 0.2 0.2\n")
    old2new = {0: 0, 1: 0}  # both SKUs -> same coarse class 0
    remap_tree(src, dst, old2new)
    classes = [ln.split()[0] for ln in
               (dst / "x.txt").read_text(encoding="utf-8").splitlines()]
    assert classes == ["0", "0"]  # collapsed


def test_remap_tree_skips_unreadable_but_continues(tmp_path, monkeypatch):
    src = tmp_path / "s" / "labels" / "train"
    dst = tmp_path / "d" / "labels" / "train"
    _write(src / "good.txt", "0 0.5 0.5 0.2 0.2\n")
    _write(src / "bad.txt", "0 0.5 0.5 0.2 0.2\n")
    # force a read error on bad.txt only; good.txt must still be written
    orig_read = Path.read_text
    def flaky(self, *a, **k):
        if self.name == "bad.txt":
            raise OSError("unreadable")
        return orig_read(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", flaky)
    written, skipped = remap_tree(src, dst, {0: 0})
    assert written == 1 and skipped == 1
    assert (dst / "good.txt").exists() and not (dst / "bad.txt").exists()


def test_write_coarse_yaml_has_17_style_shape(tmp_path):
    out_root = tmp_path / "dataset_synth_coarse"
    out_path = tmp_path / "dataset_synth_coarse.yml"
    write_coarse_yaml(out_root, ["alcohol", "candy", "tissue"], out_path)
    doc = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert doc["nc"] == 3
    assert doc["train"] == "images/train" and doc["val"] == "images/val"
    assert doc["names"] == {0: "alcohol", 1: "candy", 2: "tissue"}
    assert Path(doc["path"]) == out_root


from remap_to_coarse import link_images


def test_link_images_file_is_reachable_with_matching_content(tmp_path):
    src_images = tmp_path / "dataset_synth" / "images"
    dst_images = tmp_path / "dataset_synth_coarse" / "images"
    _write(src_images / "train" / "x.jpg", "fake-jpeg-bytes")

    link_images(src_images, dst_images)

    linked = dst_images / "train" / "x.jpg"
    assert linked.exists()
    assert linked.read_text(encoding="utf-8") == "fake-jpeg-bytes"


def test_link_images_resolve_does_not_escape_dst_tree(tmp_path):
    """Regression test for the silent-corruption bug.

    If link_images ever goes back to symlinking/junctioning the whole
    images/ directory (instead of per-file hardlinks), dst_images becomes a
    reparse point and Path.resolve() on a file inside it collapses back to
    the ORIGINAL src_images location. Ultralytics does its "/images/" ->
    "/labels/" swap on the *resolved* path, so that regression would make
    training silently read the wrong (200-class) label tree instead of the
    new coarse one. This assertion is the one that would fail on that
    regression: the resolved path of a linked file must stay under the
    resolved dst_images root, not collapse back into src_images.
    """
    src_images = tmp_path / "dataset_synth" / "images"
    dst_images = tmp_path / "dataset_synth_coarse" / "images"
    _write(src_images / "train" / "x.jpg", "fake-jpeg-bytes")

    link_images(src_images, dst_images)

    resolved_linked = (dst_images / "train" / "x.jpg").resolve()
    resolved_dst_root = dst_images.resolve()
    resolved_src_root = src_images.resolve()

    assert resolved_dst_root in resolved_linked.parents
    assert resolved_src_root not in resolved_linked.parents


def test_link_images_hardlinks_share_inode_when_supported(tmp_path):
    src_images = tmp_path / "dataset_synth" / "images"
    dst_images = tmp_path / "dataset_synth_coarse" / "images"
    _write(src_images / "train" / "x.jpg", "fake-jpeg-bytes")

    link_images(src_images, dst_images)

    src_ino = os.stat(src_images / "train" / "x.jpg").st_ino
    dst_ino = os.stat(dst_images / "train" / "x.jpg").st_ino
    if src_ino == 0 or dst_ino == 0:
        pytest.skip("st_ino not supported on this filesystem")
    assert src_ino == dst_ino  # zero-copy hardlink, not a duplicated file


def test_link_images_is_noop_when_dst_already_exists(tmp_path):
    src_images = tmp_path / "dataset_synth" / "images"
    dst_images = tmp_path / "dataset_synth_coarse" / "images"
    _write(src_images / "train" / "x.jpg", "fake-jpeg-bytes")
    dst_images.mkdir(parents=True)  # pre-existing dst -> link_images must no-op, not raise

    link_images(src_images, dst_images)  # should not raise

    assert not (dst_images / "train").exists()
