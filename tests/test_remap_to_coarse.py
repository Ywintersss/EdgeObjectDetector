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
