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


def test_gate_verdict_exact_2_0_point_drop_is_acceptable():
    # Built from real baseline arithmetic (not a pre-rounded literal) so any
    # float drift in (baseline - model) * 100.0 is actually exercised.
    baseline = 0.879
    model_map = baseline - 0.02
    assert E.gate_verdict(baseline, model_map) == "acceptable"


def test_gate_verdict_exact_5_0_point_drop_is_acceptable():
    # 0.879 - 0.05 -> (baseline - model) * 100.0 drifts to 5.000000000000004
    # in raw float64 arithmetic, which must still land in "acceptable", not
    # "RED FLAG".
    baseline = 0.879
    model_map = baseline - 0.05
    assert E.gate_verdict(baseline, model_map) == "acceptable"


def test_build_export_kwargs_forces_int8_and_real_calibration():
    kw = E.build_export_kwargs(320, "dataset_real_blend.yml")
    assert kw["format"] == "tflite"
    assert kw["int8"] is True          # Coral runs ONLY fully-integer models
    assert kw["imgsz"] == 320
    # calibration MUST read the real cluttered deployment distribution
    assert kw["data"] == "dataset_real_blend.yml"


def test_verify_class_count_accepts_17():
    E.verify_class_count(["c"] * 17)   # must not raise


def test_verify_class_count_rejects_wrong_count():
    # guards the silent disaster: wrong model bundled -> every box confidently mislabeled
    with pytest.raises(ValueError, match="17"):
        E.verify_class_count(["c"] * 16)
    with pytest.raises(ValueError, match="17"):
        E.verify_class_count(["c"] * 200)


def test_write_bundle_copies_model_and_writes_classes(tmp_path):
    src = tmp_path / "m.tflite"
    src.write_bytes(b"TFL3")
    deploy = tmp_path / "deploy"
    names = ["alcohol", "candy", "canned_food", "chocolate", "dessert", "dried_food",
             "dried_fruit", "drink", "gum", "instant_drink", "instant_noodles", "milk",
             "personal_hygiene", "puffed_food", "seasoner", "stationery", "tissue"]

    E.write_bundle(src, names, deploy)

    assert (deploy / "m.tflite").read_bytes() == b"TFL3"
    # classes.txt is DERIVED, one name per line, in index order
    assert (deploy / "classes.txt").read_text(encoding="utf-8").splitlines() == names


def test_write_bundle_rejects_wrong_class_count(tmp_path):
    src = tmp_path / "m.tflite"
    src.write_bytes(b"TFL3")
    with pytest.raises(ValueError, match="17"):
        E.write_bundle(src, ["only_one"], tmp_path / "deploy")


def test_main_dry_returns_zero(monkeypatch):
    import sys as _sys
    monkeypatch.setattr(_sys, "argv", ["export_int8.py", "--dry"])
    assert E.main() == 0


def _class_names_17():
    return ["alcohol", "candy", "canned_food", "chocolate", "dessert", "dried_food",
            "dried_fruit", "drink", "gum", "instant_drink", "instant_noodles", "milk",
            "personal_hygiene", "puffed_food", "seasoner", "stationery", "tissue"]


def test_main_missing_single_yaml_fails_fast(monkeypatch, tmp_path, capsys):
    # main() must reject a missing --single yaml BEFORE any expensive export/val
    # work happens, per the spec's "no silent failures" fail-fast requirement.
    import sys as _sys

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    calib = tmp_path / "calib.yml"
    calib.write_text(yaml.safe_dump({"nc": 17, "names": _class_names_17()}), encoding="utf-8")
    missing_single = tmp_path / "eval_single_item.yml"  # deliberately never created

    _sys.argv = [
        "export_int8.py",
        "--weights", str(weights),
        "--calib", str(calib),
        "--single", str(missing_single),
    ]
    monkeypatch.setattr(_sys, "argv", _sys.argv)

    rc = E.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "ERROR" in captured.err
    assert str(missing_single) in captured.err


def test_main_malformed_sizes_fails_fast(monkeypatch, tmp_path, capsys):
    # A bad --sizes value must produce the clean ERROR-to-stderr pattern used
    # everywhere else in main(), not an uncaught ValueError/traceback.
    import sys as _sys

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    calib = tmp_path / "calib.yml"
    calib.write_text(yaml.safe_dump({"nc": 17, "names": _class_names_17()}), encoding="utf-8")
    single = tmp_path / "eval_single_item.yml"
    single.write_text(yaml.safe_dump({"nc": 17, "names": _class_names_17()}), encoding="utf-8")

    monkeypatch.setattr(_sys, "argv", [
        "export_int8.py",
        "--weights", str(weights),
        "--calib", str(calib),
        "--single", str(single),
        "--sizes", "abc",
    ])

    rc = E.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "ERROR" in captured.err
