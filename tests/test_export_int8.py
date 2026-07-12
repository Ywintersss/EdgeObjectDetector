# tests/test_export_int8.py
import numpy as np
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


def test_load_class_names_empty_yaml_raises_value_error(tmp_path):
    # yaml.safe_load on an empty/comment-only file returns None, so doc["names"]
    # would raise a raw, uncaught TypeError. main()'s except (KeyError, ValueError)
    # doesn't catch TypeError, so a traceback would escape instead of the clean
    # ERROR/return-1 pattern used everywhere else.
    y = tmp_path / "empty.yml"
    y.write_text("# just a comment, no content\n", encoding="utf-8")
    with pytest.raises(ValueError, match=str(y).replace("\\", "\\\\")):
        E.load_class_names(y)


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


def test_build_report_table_states_calibration_eval_overlap_caveat():
    # dataset_real_blend.yml's `val:` split IS real_eval, and Ultralytics calibrates
    # from the val split -- so quantization ranges are tuned on the very images the
    # quantization cost is then measured against. The number is optimistically biased
    # and the report must say so, or a reader will treat it as an unbiased estimate.
    md = E.build_report_table([])
    lowered = md.lower()
    assert "calibration" in lowered
    assert "real_eval" in lowered
    assert "lower bound" in lowered or "optimistic" in lowered


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


def test_build_export_kwargs_honors_export_format():
    # The escape hatch: if the default (litert) path emits float32 graph I/O, the
    # user must be able to rerun via saved_model to get *_full_integer_quant.tflite,
    # which is what the Edge TPU compiler actually consumes.
    kw = E.build_export_kwargs(448, "dataset_real_blend.yml", export_format="saved_model")
    assert kw["format"] == "saved_model"
    assert kw["int8"] is True          # still non-negotiable on the alternate path
    assert kw["data"] == "dataset_real_blend.yml"
    assert kw["imgsz"] == 448


# --- fully-integer graph I/O verification -------------------------------------------
# int8=True no longer guarantees an integer-I/O graph: Ultralytics 8.4.83+ rewrites
# format="tflite" to the litert path, which deliberately keeps FP32 graph input/output.
# The Edge TPU cannot consume that, so we must assert on the ARTIFACT, not the kwargs.

def test_assert_integer_io_accepts_int8_and_uint8():
    E._assert_integer_io(np.int8, np.int8, "m.tflite")      # must not raise
    E._assert_integer_io(np.uint8, np.uint8, "m.tflite")    # must not raise
    E._assert_integer_io(np.int8, np.uint8, "m.tflite")     # must not raise


def test_assert_integer_io_rejects_float_input():
    with pytest.raises(ValueError) as exc:
        E._assert_integer_io(np.float32, np.int8, "m.tflite")
    msg = str(exc.value)
    assert "float32" in msg                    # names the ACTUAL dtype seen
    assert "--export-format saved_model" in msg  # names the remedy, at the moment of need


def test_assert_integer_io_rejects_float_output():
    with pytest.raises(ValueError) as exc:
        E._assert_integer_io(np.int8, np.float32, "m.tflite")
    msg = str(exc.value)
    assert "float32" in msg
    assert "--export-format saved_model" in msg


def test_assert_integer_io_rejects_float_both():
    with pytest.raises(ValueError, match="float32"):
        E._assert_integer_io(np.float32, np.float32, "m.tflite")


def test_resolve_export_artifact_passes_through_a_file(tmp_path):
    f = tmp_path / "best_int8.tflite"
    f.write_bytes(b"TFL3")
    assert E._resolve_export_artifact(f) == f


def test_resolve_export_artifact_finds_tflite_inside_a_directory(tmp_path):
    # model.export() can return a *_saved_model/ DIRECTORY; copying that with
    # shutil.copy2 would raise IsADirectoryError.
    d = tmp_path / "best_saved_model"
    d.mkdir()
    (d / "best_float32.tflite").write_bytes(b"nope")
    wanted = d / "best_full_integer_quant.tflite"
    wanted.write_bytes(b"TFL3")

    assert E._resolve_export_artifact(d) == wanted


def test_resolve_export_artifact_directory_without_full_integer_raises(tmp_path):
    d = tmp_path / "best_saved_model"
    d.mkdir()
    (d / "best_float32.tflite").write_bytes(b"nope")
    with pytest.raises(ValueError, match="full_integer_quant"):
        E._resolve_export_artifact(d)


def test_verify_full_integer_rejects_a_non_model_file(tmp_path):
    # Not a mock: a real interpreter is constructed and really rejects this.
    bad = tmp_path / "bad.tflite"
    bad.write_bytes(b"not a flatbuffer")
    with pytest.raises((ValueError, RuntimeError)):
        E.verify_full_integer(bad)


def test_not_fully_integer_error_is_a_value_error():
    # Must stay a ValueError subclass so any existing `except ValueError` handling
    # (e.g. _assert_integer_io's callers) keeps working unmodified.
    assert issubclass(E.NotFullyIntegerError, ValueError)


def test_assert_integer_io_raises_not_fully_integer_error_on_float_io():
    # _run_sweep needs to distinguish "format is wrong" (abort the whole sweep)
    # from "this one size broke" (keep going) -- that requires a distinct type,
    # not just any ValueError.
    with pytest.raises(E.NotFullyIntegerError):
        E._assert_integer_io(np.float32, np.int8, "m.tflite")


# --- sweep abort on a format-level failure -------------------------------------------
# If verify_full_integer fails at one size, it fails for the SAME reason at every
# size -- it is a FORMAT problem, not a resolution problem. Re-running the ~10-minute
# export for the remaining sizes only proves the same fact again at real GPU cost.

def test_run_sweep_aborts_after_first_not_fully_integer_error(monkeypatch, tmp_path, capsys):
    calls = []

    def _fake_export_one_size(weights, size, calib_yaml, out_dir, export_format="tflite"):
        calls.append(size)
        raise E.NotFullyIntegerError(f"{size} is NOT a fully-integer graph")

    monkeypatch.setattr(E, "export_one_size", _fake_export_one_size)
    monkeypatch.setattr(E, "PROJECT_ROOT", tmp_path)

    class _Args:
        weights = "w.pt"
        calib = "calib.yml"
        single = "single.yml"
        export_format = "tflite"

    rc = E._run_sweep(_Args(), [640, 448, 320])

    # export_one_size must be attempted for 640 only -- 448/320 would just burn
    # ~10 more GPU-minutes each to relearn the same format-level fact.
    assert calls == [640]
    assert rc != 0

    err = capsys.readouterr().err
    assert "FORMAT" in err or "format" in err
    assert "--export-format saved_model" in err

    report = (tmp_path / "export" / "report.md").read_text(encoding="utf-8")
    # size 640 must be honestly reported as FAILED -- never silently omitted
    assert "640" in report
    assert "FAILED" in report
    # 448/320 were never attempted -- the report must say so explicitly (SKIPPED),
    # never silently omitted and never implying they passed or were tested.
    assert "448" in report
    assert "320" in report
    assert "SKIPPED" in report


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


def test_write_bundle_removes_stale_tflite_models(tmp_path):
    # Bundling 448 after having bundled 320 must not leave BOTH models in deploy/,
    # or the demo can silently run the size the user did not choose.
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    stale = deploy / "rpc_coarse17_int8_320.tflite"
    stale.write_bytes(b"OLD")

    src = tmp_path / "rpc_coarse17_int8_448.tflite"
    src.write_bytes(b"NEW")
    E.write_bundle(src, _class_names_17(), deploy)

    assert not stale.exists(), "stale model left behind -> demo could run the wrong size"
    assert [p.name for p in deploy.glob("*.tflite")] == ["rpc_coarse17_int8_448.tflite"]


def test_val_tflite_fails_loudly_when_latency_is_missing(monkeypatch):
    # latency is a PUBLISHED column of the deliverable table; a silent 0.0 ms would
    # be a fabricated number in a report someone reads as measured.
    import ultralytics

    class _Box:
        map50 = 0.99
        map = 0.86

    class _Metrics:
        box = _Box()
        speed = {}          # no "inference" key

    class _FakeYOLO:
        def __init__(self, *a, **kw):
            pass

        def val(self, *a, **kw):
            return _Metrics()

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    with pytest.raises(KeyError):
        E.val_tflite("m.tflite", "d.yml", 320)


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


def test_main_bundle_does_not_require_single_yaml_or_sizes(monkeypatch, tmp_path, capsys):
    # --bundle consumes neither eval_single_item.yml nor --sizes, so demanding them
    # blocks a legitimate bundle run for inputs it will never read.
    import sys as _sys

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    calib = tmp_path / "calib.yml"
    calib.write_text(yaml.safe_dump({"nc": 17, "names": _class_names_17()}), encoding="utf-8")

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "rpc_coarse17_int8_320.tflite").write_bytes(b"TFL3")
    deploy_dir = tmp_path / "deploy"
    monkeypatch.setattr(E, "PROJECT_ROOT", tmp_path)

    monkeypatch.setattr(_sys, "argv", [
        "export_int8.py",
        "--weights", str(weights),
        "--calib", str(calib),
        "--single", str(tmp_path / "never_created.yml"),   # missing, and irrelevant
        "--sizes", "abc",                                  # malformed, and irrelevant
        "--bundle", "320",
    ])

    rc = E.main()
    assert rc == 0, capsys.readouterr().err
    assert (deploy_dir / "rpc_coarse17_int8_320.tflite").exists()
    assert (deploy_dir / "classes.txt").exists()


def test_main_empty_calib_yaml_returns_clean_error(monkeypatch, tmp_path, capsys):
    # An empty/comment-only --calib yaml must hit main()'s existing
    # except (KeyError, ValueError) handler and print the clean ERROR pattern --
    # not let a raw TypeError traceback escape.
    import sys as _sys

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    calib = tmp_path / "empty.yml"
    calib.write_text("# nothing here\n", encoding="utf-8")
    single = tmp_path / "single.yml"
    single.write_text(yaml.safe_dump({"nc": 17, "names": _class_names_17()}), encoding="utf-8")

    monkeypatch.setattr(_sys, "argv", [
        "export_int8.py",
        "--weights", str(weights),
        "--calib", str(calib),
        "--single", str(single),
    ])

    rc = E.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "ERROR" in captured.err


def test_main_yaml_without_names_key_returns_clean_error(monkeypatch, tmp_path, capsys):
    # A yaml missing its names: block must produce the ERROR/return-1 pattern used
    # everywhere else in main(), not a raw KeyError traceback.
    import sys as _sys

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    calib = tmp_path / "calib.yml"
    calib.write_text(yaml.safe_dump({"nc": 17}), encoding="utf-8")   # no names:
    single = tmp_path / "single.yml"
    single.write_text(yaml.safe_dump({"nc": 17, "names": _class_names_17()}), encoding="utf-8")

    monkeypatch.setattr(_sys, "argv", [
        "export_int8.py",
        "--weights", str(weights),
        "--calib", str(calib),
        "--single", str(single),
    ])

    rc = E.main()
    assert rc == 1
    assert "ERROR" in capsys.readouterr().err
