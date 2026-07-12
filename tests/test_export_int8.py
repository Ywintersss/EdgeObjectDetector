# tests/test_export_int8.py
import pytest
import yaml

import export_int8 as E


def test_parse_sizes_parses_csv():
    assert E.parse_sizes("640,448,320") == [640, 448, 320]
    assert E.parse_sizes("320") == [320]
    assert E.parse_sizes(" 640 , 320 ") == [640, 320]


def test_parse_sizes_rejects_malformed():
    for bad in ("", "640,abc", "640,-1", "640,0"):
        with pytest.raises(ValueError):
            E.parse_sizes(bad)


def test_load_class_names_orders_by_index(tmp_path):
    # names dict deliberately OUT of insertion order -> must come back index-ordered
    y = tmp_path / "d.yml"
    y.write_text(yaml.safe_dump({
        "nc": 3,
        "names": {2: "canned_food", 0: "alcohol", 1: "candy"},
    }), encoding="utf-8")
    assert E.load_class_names(y) == ["alcohol", "candy", "canned_food"]


def test_load_class_names_accepts_list(tmp_path):
    y = tmp_path / "d.yml"
    y.write_text(yaml.safe_dump({"nc": 2, "names": ["alcohol", "candy"]}), encoding="utf-8")
    assert E.load_class_names(y) == ["alcohol", "candy"]


def test_load_class_names_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.load_class_names(tmp_path / "nope.yml")
