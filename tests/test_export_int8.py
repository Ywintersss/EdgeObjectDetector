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


def test_gate_verdict_thresholds():
    # drop measured in percentage points of mAP@50-95 against the 0.879 baseline
    assert E.gate_verdict(0.879, 0.870) == "clean"        # 0.9 pt drop
    assert E.gate_verdict(0.879, 0.860) == "clean"        # 1.9 pt drop
    assert E.gate_verdict(0.879, 0.855) == "acceptable"   # 2.4 pt drop
    assert E.gate_verdict(0.879, 0.830) == "acceptable"   # 4.9 pt drop
    assert E.gate_verdict(0.879, 0.820) == "RED FLAG"     # 5.9 pt drop


def test_gate_verdict_improvement_is_clean():
    # a model scoring ABOVE baseline is obviously not a regression
    assert E.gate_verdict(0.879, 0.890) == "clean"


def test_build_report_table_includes_rows_and_delta():
    rows = [
        {"size": 320, "status": "ok", "cluttered_map50": 0.990, "cluttered_map": 0.860,
         "single_map50": 0.992, "single_map": 0.930, "bytes": 3_100_000, "latency_ms": 41.0},
    ]
    md = E.build_report_table(rows)
    assert "320" in md
    assert "0.860" in md
    assert "clean" in md          # 1.9pt drop -> clean
    assert "-1.9" in md           # delta in percentage points, signed


def test_build_report_table_marks_failed_rows():
    rows = [{"size": 640, "status": "FAILED", "error": "onnx2tf blew up"}]
    md = E.build_report_table(rows)
    assert "FAILED" in md
    assert "onnx2tf blew up" in md
    # a failed size must never be silently omitted
    assert "640" in md
