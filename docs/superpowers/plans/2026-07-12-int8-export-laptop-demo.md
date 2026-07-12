# INT8 Export + Laptop Overhead-Camera Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the trained YOLO11n coarse-17 detector to INT8 TFLite at 640/448/320, measure the accuracy cost of each, and run the chosen model live on a laptop overhead camera.

**Architecture:** One desktop script (`export_int8.py`) does export + validation + reporting, because INT8 calibration must read the ~10 GB real cluttered dataset that only exists on the desktop. It emits a size-vs-accuracy table and assembles a small self-contained `deploy/` bundle. A second script (`deploy/webcam_demo.py`) runs on the laptop against that bundle and needs no dataset.

**Tech Stack:** Python 3.13, Ultralytics 8.4.x, PyTorch 2.11+cu128, OpenCV (`opencv-python`), PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-07-12-int8-export-laptop-demo-design.md`

## Global Constraints

- Run Python with `python`, never `python3`.
- Model under test: `runs/detect/rpc_real_blend_b0_full/weights/best.pt` (YOLO11n, 17 classes).
- FP32 baseline to compare against: **cluttered mAP@50 = 0.995, mAP@50-95 = 0.879**; single-item mAP@50 = 0.995, mAP@50-95 = 0.945.
- INT8 calibration data MUST be `dataset_real_blend.yml` (real cluttered images). Never calibrate on studio singles.
- Export sizes: **640, 448, 320**.
- Class count is exactly **17**. Coarse-17 names, in index order 0–16:
  `alcohol, candy, canned_food, chocolate, dessert, dried_food, dried_fruit, drink, gum, instant_drink, instant_noodles, milk, personal_hygiene, puffed_food, seasoner, stationery, tissue`
- Quantization gate (mAP@50-95 drop vs 0.879, in percentage points): `< 2` = clean; `2–5` = acceptable; `> 5` = RED FLAG, do not proceed to Coral.
- No silent failures. Every error path prints to stderr and returns a non-zero exit code.
- Code style: snake_case, small pure functions, module docstring with Usage, `main() -> int`, `raise SystemExit(main())` — matching `train.py` and `build_real_coarse.py`.
- Latency measured in this plan is **desktop CPU only** and is a relative size-vs-size indicator. It is NOT a Coral FPS forecast.

## File Structure

| File | Responsibility |
|---|---|
| `export_int8.py` (create) | Desktop: export sweep, validation, report, bundle assembly. |
| `deploy/webcam_demo.py` (create) | Laptop: overhead camera loop + live inference. No dataset dependency. |
| `deploy/README.md` (create) | Laptop run instructions. |
| `tests/test_export_int8.py` (create) | Unit tests for export helpers. |
| `tests/test_webcam_demo.py` (create) | Unit tests for camera/detection helpers. |
| `.gitignore` (modify) | Ignore generated `export/` artifacts and `deploy/classes.txt`. |

Generated (not committed): `export/*.tflite`, `deploy/*.tflite`, `deploy/classes.txt`.
Committed as a record: `export/report.md`.

---

### Task 1: Pure helpers — `parse_sizes` and `load_class_names`

**Files:**
- Create: `export_int8.py`
- Test: `tests/test_export_int8.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `parse_sizes(text: str) -> list[int]`
  - `load_class_names(yaml_path) -> list[str]` — returns names ordered by class index.

`load_class_names` is a correctness guard, not a convenience. Ultralytics yaml `names:` is a dict
keyed by index (`{0: alcohol, 1: candy, ...}`). If we iterate it in insertion order rather than
index order, every detection gets a confidently wrong label and nothing crashes.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_export_int8.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_int8'`

- [ ] **Step 3: Write minimal implementation**

```python
"""
export_int8.py — Export the trained coarse-17 YOLO11n to INT8 TFLite at several
input sizes, measure the accuracy cost of each, and assemble a laptop deploy bundle.

Usage:
    python export_int8.py --sizes 320                  # validate the toolchain first (fast)
    python export_int8.py --sizes 640,448,320          # full sweep -> export/report.md
    python export_int8.py --bundle 320                 # assemble deploy/ for the laptop

INT8 calibration reads REAL cluttered images (dataset_real_blend.yml). Calibrating on
the wrong distribution silently costs accuracy that looks like "quantization is lossy".
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

# FP32 baseline measured on the held-out splits (see the design spec).
BASELINE = {"cluttered_map50": 0.995, "cluttered_map": 0.879,
            "single_map50": 0.995, "single_map": 0.945}

EXPECTED_CLASSES = 17


def parse_sizes(text: str) -> list[int]:
    """Parse '640,448,320' -> [640, 448, 320]. Raises ValueError on anything not a positive int."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"no sizes parsed from {text!r}")
    sizes = []
    for p in parts:
        try:
            n = int(p)
        except ValueError as exc:
            raise ValueError(f"invalid size {p!r} in {text!r}") from exc
        if n <= 0:
            raise ValueError(f"size must be positive, got {n}")
        sizes.append(n)
    return sizes


def load_class_names(yaml_path) -> list[str]:
    """Class names ordered by CLASS INDEX (not dict insertion order).

    Ultralytics writes names as {0: 'alcohol', 1: 'candy', ...}. Iterating that dict in
    insertion order would mislabel every detection while crashing nothing — so we sort
    explicitly by index.
    """
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"dataset yaml not found: {p}")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    names = doc["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_export_int8.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add export_int8.py tests/test_export_int8.py
git commit -m "feat: export_int8 size parsing + index-ordered class names"
```

---

### Task 2: The quantization gate and the report table

**Files:**
- Modify: `export_int8.py`
- Test: `tests/test_export_int8.py`

**Interfaces:**
- Consumes: `BASELINE` from Task 1.
- Produces:
  - `gate_verdict(baseline_map: float, model_map: float) -> str` → `"clean" | "acceptable" | "RED FLAG"`
  - `build_report_table(rows: list[dict]) -> str` → markdown.

A `row` dict has keys: `size` (int), `status` (`"ok"` or `"FAILED"`), and when ok:
`cluttered_map50`, `cluttered_map`, `single_map50`, `single_map` (floats), `bytes` (int),
`latency_ms` (float). When FAILED it carries `error` (str) instead.

This encodes the spec's acceptance criteria as executable code rather than prose in a doc.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_export_int8.py

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_export_int8.py -v`
Expected: FAIL — `AttributeError: module 'export_int8' has no attribute 'gate_verdict'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to export_int8.py

def gate_verdict(baseline_map: float, model_map: float) -> str:
    """Classify the INT8 accuracy drop per the spec's quantization gate.

    Drop is in PERCENTAGE POINTS of mAP@50-95: <2 clean, 2-5 acceptable, >5 RED FLAG.
    A RED FLAG means the first suspect is the calibration data, not the input size.
    """
    drop_points = (baseline_map - model_map) * 100.0
    if drop_points < 2.0:
        return "clean"
    if drop_points <= 5.0:
        return "acceptable"
    return "RED FLAG"


def build_report_table(rows: list[dict]) -> str:
    """Render the size-vs-accuracy table. FAILED sizes are shown, never dropped."""
    lines = [
        "# INT8 Export Report",
        "",
        f"FP32 baseline — cluttered mAP@50-95: **{BASELINE['cluttered_map']:.3f}**, "
        f"single-item: **{BASELINE['single_map']:.3f}**",
        "",
        "Latency is desktop CPU, for comparing sizes against each other only — "
        "it is NOT a Coral FPS forecast.",
        "",
        "| Size | Cluttered mAP@50 | Cluttered mAP@50-95 | Δ vs FP32 (pts) | Gate | "
        "Single mAP@50-95 | File | CPU latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -x["size"]):
        if r["status"] != "ok":
            lines.append(
                f"| {r['size']} | FAILED | FAILED | — | FAILED | — | — | {r.get('error', '')} |")
            continue
        delta = (r["cluttered_map"] - BASELINE["cluttered_map"]) * 100.0
        verdict = gate_verdict(BASELINE["cluttered_map"], r["cluttered_map"])
        lines.append(
            f"| {r['size']} | {r['cluttered_map50']:.3f} | {r['cluttered_map']:.3f} | "
            f"{delta:+.1f} | {verdict} | {r['single_map']:.3f} | "
            f"{r['bytes'] / 1e6:.1f} MB | {r['latency_ms']:.1f} ms |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_export_int8.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add export_int8.py tests/test_export_int8.py
git commit -m "feat: quantization gate + size-vs-accuracy report table"
```

---

### Task 3: Export and validate one size

**Files:**
- Modify: `export_int8.py`
- Test: `tests/test_export_int8.py`

**Interfaces:**
- Consumes: `BASELINE`, `EXPECTED_CLASSES`, `gate_verdict` (Tasks 1–2).
- Produces:
  - `build_export_kwargs(size: int, calib_yaml: str) -> dict`
  - `verify_class_count(names: list[str]) -> None` — raises `ValueError` unless exactly 17.
  - `export_one_size(weights, size, calib_yaml, out_dir) -> Path`
  - `val_tflite(tflite_path, data_yaml, size) -> dict` with keys `map50`, `map`, `latency_ms`.

`build_export_kwargs` and `verify_class_count` are pure and tested. `export_one_size` / `val_tflite`
wrap Ultralytics and are exercised for real in Task 5 — they are thin by design so that almost all
the logic sits in the tested pure functions.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_export_int8.py

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_export_int8.py -v`
Expected: FAIL — `AttributeError: module 'export_int8' has no attribute 'build_export_kwargs'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to export_int8.py

def build_export_kwargs(size: int, calib_yaml: str) -> dict:
    """Ultralytics export kwargs for a fully-integer TFLite model.

    int8=True is non-negotiable: the Edge TPU executes ONLY fully-integer models.
    `data` supplies the calibration images that set the quantization ranges.
    """
    return {"format": "tflite", "int8": True, "imgsz": size, "data": calib_yaml}


def verify_class_count(names: list[str]) -> None:
    """Fail loudly unless the model/labels carry exactly 17 coarse classes.

    A wrong class count means every detection is confidently mislabeled while nothing
    crashes — the worst failure mode, because it looks like success.
    """
    if len(names) != EXPECTED_CLASSES:
        raise ValueError(
            f"expected {EXPECTED_CLASSES} classes, got {len(names)} — wrong model or yaml?")


def export_one_size(weights, size: int, calib_yaml: str, out_dir: Path) -> Path:
    """Export weights -> INT8 TFLite at `size`. Returns the path of the copied artifact."""
    import shutil

    from ultralytics import YOLO

    model = YOLO(str(weights))
    produced = Path(model.export(**build_export_kwargs(size, str(calib_yaml))))

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"rpc_coarse17_int8_{size}.tflite"
    shutil.copy2(produced, dest)
    return dest


def val_tflite(tflite_path, data_yaml, size: int) -> dict:
    """Validate an exported TFLite model. Returns mAP@50, mAP@50-95, and CPU latency (ms)."""
    from ultralytics import YOLO

    model = YOLO(str(tflite_path), task="detect")
    metrics = model.val(data=str(data_yaml), imgsz=size, verbose=False)
    return {
        "map50": float(metrics.box.map50),
        "map": float(metrics.box.map),
        "latency_ms": float(metrics.speed.get("inference", 0.0)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_export_int8.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add export_int8.py tests/test_export_int8.py
git commit -m "feat: INT8 export kwargs, class-count guard, tflite validation"
```

---

### Task 4: CLI, sweep orchestration, and the deploy bundle

**Files:**
- Modify: `export_int8.py`
- Modify: `.gitignore`
- Create: `deploy/README.md`
- Test: `tests/test_export_int8.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `write_bundle(tflite_path, names, deploy_dir) -> None` — writes the `.tflite` + `classes.txt`.
  - `main() -> int`

A failing size records `status="FAILED"` and the sweep continues — one bad size must not destroy a
30-minute run — but it is printed loudly and appears in the table.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_export_int8.py

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_export_int8.py -v`
Expected: FAIL — `AttributeError: module 'export_int8' has no attribute 'write_bundle'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to export_int8.py

def write_bundle(tflite_path, names: list[str], deploy_dir) -> None:
    """Assemble the laptop bundle: the chosen model + a DERIVED classes.txt.

    classes.txt is generated from the yaml names block, never hand-typed — a
    hand-typed ordering is exactly how you get silently mislabeled detections.
    """
    import shutil

    verify_class_count(names)
    deploy_dir = Path(deploy_dir)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite_path, deploy_dir / Path(tflite_path).name)
    (deploy_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def _run_sweep(args) -> int:
    """Export + validate each size, then write export/report.md. Returns an exit code."""
    out_dir = PROJECT_ROOT / "export"
    rows = []
    for size in parse_sizes(args.sizes):
        print(f"\n=== size {size}: exporting INT8 (calibrating on {args.calib}) ===")
        try:
            tflite = export_one_size(args.weights, size, args.calib, out_dir)
            cluttered = val_tflite(tflite, args.calib, size)
            single = val_tflite(tflite, args.single, size)
            rows.append({
                "size": size, "status": "ok",
                "cluttered_map50": cluttered["map50"], "cluttered_map": cluttered["map"],
                "single_map50": single["map50"], "single_map": single["map"],
                "bytes": tflite.stat().st_size, "latency_ms": cluttered["latency_ms"],
            })
            print(f"  size {size}: cluttered mAP@50-95 = {cluttered['map']:.3f} "
                  f"({gate_verdict(BASELINE['cluttered_map'], cluttered['map'])})")
        except Exception as exc:  # noqa: BLE001 — one bad size must not kill the sweep
            print(f"ERROR: size {size} FAILED: {exc}", file=sys.stderr)
            rows.append({"size": size, "status": "FAILED", "error": str(exc)})

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    report.write_text(build_report_table(rows), encoding="utf-8")
    print(f"\nWrote {report}")

    # Non-zero exit if EVERY size failed — the sweep produced nothing usable.
    return 1 if all(r["status"] != "ok" for r in rows) else 0


def main() -> int:
    p = argparse.ArgumentParser(description="INT8 export sweep + deploy bundle (non-destructive).")
    p.add_argument("--weights",
                   default="runs/detect/rpc_real_blend_b0_full/weights/best.pt")
    p.add_argument("--calib", default="dataset_real_blend.yml",
                   help="Calibration + cluttered-eval yaml. MUST be the real cluttered set.")
    p.add_argument("--single", default="eval_single_item.yml",
                   help="Single-item eval yaml.")
    p.add_argument("--sizes", default="640,448,320")
    p.add_argument("--bundle", type=int, default=None,
                   help="Assemble deploy/ from the already-exported model of this size.")
    p.add_argument("--dry", action="store_true", help="Parse args and exit (test hook).")
    args = p.parse_args()
    if args.dry:
        return 0

    weights = Path(args.weights)
    calib = Path(args.calib)
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        return 1
    if not calib.exists():
        print(f"ERROR: calibration yaml not found: {calib}", file=sys.stderr)
        return 1

    names = load_class_names(calib)
    try:
        verify_class_count(names)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.bundle is not None:
        tflite = PROJECT_ROOT / "export" / f"rpc_coarse17_int8_{args.bundle}.tflite"
        if not tflite.exists():
            print(f"ERROR: no exported model at {tflite}; run the sweep first.", file=sys.stderr)
            return 1
        write_bundle(tflite, names, PROJECT_ROOT / "deploy")
        print(f"Bundled {tflite.name} + classes.txt -> deploy/")
        return 0

    return _run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_export_int8.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Ignore generated artifacts**

Append to `.gitignore`:

```gitignore
# --- INT8 export artifacts (rebuildable; report.md is kept as a record) ---
export/*.tflite
deploy/*.tflite
deploy/classes.txt
```

- [ ] **Step 6: Write the laptop instructions**

Create `deploy/README.md`:

```markdown
# EdgeObjectDetector — Laptop Demo Bundle

Self-contained. No dataset and no repo checkout needed.

## Setup

    pip install ultralytics opencv-python

No CUDA required — the INT8 model runs on CPU, which is the point: it approximates
what the Coral Edge TPU will execute.

## Run

    python webcam_demo.py --model rpc_coarse17_int8_320.tflite --camera 0

## POINT THE CAMERA DOWN

This model was trained on RPC checkout scenes, which are **top-down overhead shots of
products lying on a flat, plain surface**.

Held at eye level, pointed at a product in your hand, it is out-of-domain and will
underperform. That is not a bug — it is the training distribution.

Prop the camera **above a table, looking down**, and place products on a plain surface.
```

- [ ] **Step 7: Commit**

```bash
git add export_int8.py tests/test_export_int8.py .gitignore deploy/README.md
git commit -m "feat: export sweep CLI, deploy bundle, laptop README"
```

---

### Task 5: Toolchain gate — actually export at 320

**Files:** none changed unless the toolchain fails.

This is the risky step, isolated on purpose. The Ultralytics INT8 path runs
`onnx` → `onnx2tf` → `tensorflow`, and this environment is **Python 3.13 + Windows** — the
least-travelled combination for that stack. Prove it works at the cheapest size before spending
30 minutes on a full sweep.

- [ ] **Step 1: Run a single-size export end-to-end**

Run: `python export_int8.py --sizes 320`

Expected: Ultralytics installs `onnx2tf` / `tensorflow` on first use, exports, then validates.
Console shows `size 320: cluttered mAP@50-95 = 0.XXX (clean|acceptable|RED FLAG)`.

- [ ] **Step 2: Confirm the report holds real numbers**

Run: `cat export/report.md`
Expected: a table row for 320 with real mAP values — not `FAILED`.

- [ ] **Step 3: If the export FAILED — escalate through the fallbacks, in order**

Do **not** work around it by disabling `int8`. A float model is useless on an Edge TPU.

1. Read the actual error. If it is a missing package, install it and retry.
2. Dependency/ABI failure → create a Python 3.11 venv for export only:
   `py -3.11 -m venv .venv-export && .venv-export\Scripts\pip install ultralytics` then rerun.
3. Still failing → run the export inside **WSL2**. Phase 2 needs WSL2 anyway for
   `edgetpu_compiler`, so this is not wasted work.

Record which path worked in the commit message — Phase 2 will reuse it.

- [ ] **Step 4: Commit the report**

```bash
git add export/report.md
git commit -m "chore: INT8 export toolchain validated at 320"
```

---

### Task 6: Full sweep

**Files:** `export/report.md` (generated).

- [ ] **Step 1: Run the full sweep**

Run: `python export_int8.py --sizes 640,448,320`
Expected: three exports + six validations. Roughly 20–40 minutes.

- [ ] **Step 2: Read the gate**

Run: `cat export/report.md`

Check the **Gate** column against the cluttered mAP@50-95 baseline of 0.879:
- `clean` / `acceptable` → pick a size and continue.
- `RED FLAG` at every size → **stop.** Do not proceed to the bundle or to Coral. The first suspect
  is the calibration data (is `--calib` really the real cluttered set?), not the input size.

- [ ] **Step 3: Commit the report**

```bash
git add export/report.md
git commit -m "chore: INT8 size sweep results (640/448/320)"
```

---

### Task 7: Webcam demo — camera and detection helpers

**Files:**
- Create: `deploy/webcam_demo.py`
- Test: `tests/test_webcam_demo.py`

**Interfaces:**
- Consumes: `deploy/classes.txt`, a `.tflite` from Task 4's bundle.
- Produces:
  - `load_classes(path) -> list[str]`
  - `list_available_cameras(max_index: int = 5) -> list[int]`
  - `open_camera(index: int) -> cv2.VideoCapture` — raises `RuntimeError` naming working indices.
  - `detect_frame(model, frame, conf: float = 0.25) -> list[dict]` — each dict has
    `cls` (int), `conf` (float), `box` (tuple of 4 ints, x1 y1 x2 y2).
  - `draw_detections(frame, detections, names) -> frame`

`detect_frame` is pure with respect to the camera — it takes a numpy frame. That is what makes
inference testable without any hardware, and it is tested against a **real dataset image** and the
**real PyTorch model** (not the tflite, so this test does not depend on Task 5 succeeding).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_webcam_demo.py
import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import webcam_demo as W  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_ROOT / "runs/detect/rpc_real_blend_b0_full/weights/best.pt"
REAL_IMAGES = PROJECT_ROOT / "dataset_real/images/real_eval"


def test_load_classes_reads_one_per_line(tmp_path):
    p = tmp_path / "classes.txt"
    p.write_text("alcohol\ncandy\ncanned_food\n", encoding="utf-8")
    assert W.load_classes(p) == ["alcohol", "candy", "canned_food"]


def test_load_classes_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        W.load_classes(tmp_path / "nope.txt")


def test_open_camera_invalid_index_raises_with_enumeration():
    # index 999 cannot exist; the error must TELL the user what does exist,
    # because silently grabbing the wrong camera is the classic failure here.
    with pytest.raises(RuntimeError, match="[Cc]amera"):
        W.open_camera(999)


@pytest.mark.skipif(not (WEIGHTS.exists() and REAL_IMAGES.is_dir()),
                    reason="trained weights or real_eval images not present")
def test_detect_frame_finds_objects_in_a_real_cluttered_image():
    from ultralytics import YOLO

    img_path = next(REAL_IMAGES.glob("*.jpg"))
    frame = cv2.imread(str(img_path))
    assert frame is not None

    model = YOLO(str(WEIGHTS))
    dets = W.detect_frame(model, frame, conf=0.25)

    assert len(dets) >= 1, "model found nothing in a real cluttered checkout scene"
    h, w = frame.shape[:2]
    for d in dets:
        assert 0 <= d["cls"] <= 16, f"class index {d['cls']} outside coarse-17 range"
        assert 0.0 <= d["conf"] <= 1.0
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, "box outside image bounds"


@pytest.mark.skipif(not (WEIGHTS.exists() and REAL_IMAGES.is_dir()),
                    reason="trained weights or real_eval images not present")
def test_draw_detections_returns_same_shape_frame():
    from ultralytics import YOLO

    img_path = next(REAL_IMAGES.glob("*.jpg"))
    frame = cv2.imread(str(img_path))
    model = YOLO(str(WEIGHTS))
    dets = W.detect_frame(model, frame, conf=0.25)

    out = W.draw_detections(frame.copy(), dets, ["c"] * 17)
    assert out.shape == frame.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_webcam_demo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webcam_demo'`

- [ ] **Step 3: Write minimal implementation**

```python
"""
webcam_demo.py — Live overhead-camera demo for the coarse-17 INT8 detector.

Usage:
    python webcam_demo.py --model rpc_coarse17_int8_320.tflite --camera 0

POINT THE CAMERA DOWN at products on a plain surface. This model was trained on
top-down RPC checkout scenes; at eye level it is out-of-domain and will underperform.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
EXPECTED_CLASSES = 17


def load_classes(path) -> list[str]:
    """Read classes.txt — one class name per line, in class-index order."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"classes file not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def list_available_cameras(max_index: int = 5) -> list[int]:
    """Probe camera indices 0..max_index-1 and return the ones that actually open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


def open_camera(index: int):
    """Open a camera, or fail loudly listing the indices that DO work.

    Silently opening the wrong camera is the classic failure here — virtual cams
    (OBS, Teams) and external webcams shift the indices around.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        available = list_available_cameras()
        raise RuntimeError(
            f"could not open camera index {index}. Working camera indices: "
            f"{available if available else 'NONE FOUND'}")
    return cap


def detect_frame(model, frame, conf: float = 0.25) -> list[dict]:
    """Run the detector on one BGR frame. Pure w.r.t. the camera, so it is testable."""
    results = model.predict(frame, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append({
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "box": (x1, y1, x2, y2),
            })
    return detections


def draw_detections(frame, detections: list[dict], names: list[str]):
    """Draw boxes + '<class> <conf>' labels onto the frame (mutates and returns it)."""
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"{names[d['cls']]} {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_webcam_demo.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add deploy/webcam_demo.py tests/test_webcam_demo.py
git commit -m "feat: webcam demo camera + detection helpers"
```

---

### Task 8: Webcam demo — the camera loop and CLI

**Files:**
- Modify: `deploy/webcam_demo.py`
- Test: `tests/test_webcam_demo.py`

**Interfaces:**
- Consumes: everything from Task 7.
- Produces: `main() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_webcam_demo.py

def test_main_missing_model_returns_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv",
                        ["webcam_demo.py", "--model", str(tmp_path / "nope.tflite")])
    rc = W.main()
    assert rc == 1                          # never silently continue without a model
    assert "ERROR" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_webcam_demo.py::test_main_missing_model_returns_error -v`
Expected: FAIL — `AttributeError: module 'webcam_demo' has no attribute 'main'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to webcam_demo.py

def run_loop(model, cap, names, conf: float) -> None:
    """Read frames, detect, draw, show FPS. 'q' quits."""
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: dropped frame from camera", file=sys.stderr)
            break

        detections = detect_frame(model, frame, conf=conf)
        frame = draw_detections(frame, detections, names)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"{fps:5.1f} FPS  |  {len(detections)} objects",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "POINT CAMERA DOWN at a plain surface",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow("EdgeObjectDetector — coarse-17 (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> int:
    p = argparse.ArgumentParser(description="Live overhead-camera demo (coarse-17 INT8).")
    p.add_argument("--model", default=str(HERE / "rpc_coarse17_int8_320.tflite"))
    p.add_argument("--classes", default=str(HERE / "classes.txt"))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    try:
        names = load_classes(args.classes)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if len(names) != EXPECTED_CLASSES:
        print(f"ERROR: expected {EXPECTED_CLASSES} classes, got {len(names)} — "
              f"wrong classes.txt bundled? Every box would be mislabeled.", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ERROR: ultralytics not installed ({exc}). Run: pip install ultralytics "
              f"opencv-python", file=sys.stderr)
        return 1

    try:
        cap = open_camera(args.camera)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    model = YOLO(str(model_path), task="detect")
    print(f"Running {model_path.name} on camera {args.camera}. Press 'q' to quit.")
    print("POINT THE CAMERA DOWN at products on a plain surface.")

    try:
        run_loop(model, cap, names, args.conf)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (all tests, including the existing `test_build_real_coarse.py` suite)

- [ ] **Step 5: Commit**

```bash
git add deploy/webcam_demo.py tests/test_webcam_demo.py
git commit -m "feat: webcam demo camera loop + CLI"
```

---

### Task 9: Bundle and human-verify on the laptop

**Files:** `deploy/` (generated contents).

The demo cannot be automated — "does it draw a correct box on a real bag of chips" needs eyes.

- [ ] **Step 1: Assemble the bundle for the chosen size**

Run: `python export_int8.py --bundle 320` (substitute the size chosen from Task 6's table)
Expected: `Bundled rpc_coarse17_int8_320.tflite + classes.txt -> deploy/`

- [ ] **Step 2: Copy `deploy/` to the laptop**

USB stick or cloud drive. It is ~3 MB. Do NOT try to `git pull` it — `.tflite` is gitignored.

- [ ] **Step 3: Set up the laptop**

Run: `pip install ultralytics opencv-python`

- [ ] **Step 4: Rig the camera OVERHEAD**

Point the camera **DOWN** at a plain surface (a plain table or a sheet of paper). Place a few
retail products on it, separated. Eye-level results are not a valid measure of this model.

- [ ] **Step 5: Run the demo**

Run: `python webcam_demo.py --model rpc_coarse17_int8_320.tflite --camera 0`

If the camera index is wrong, the error lists the working indices — use one of those.

- [ ] **Step 6: Verify with your eyes**

Confirm: boxes appear on products, labels are plausible coarse categories (a soda → `drink`, chips
→ `puffed_food`), and FPS is displayed. Report what you see — including if it is bad. A poor
overhead result is real information, not a failure to hide.

---

## Verification Summary

| Acceptance criterion (from the spec) | Task |
|---|---|
| `export/report.md` with real numbers for 640/448/320 + baseline | 6 |
| Quantization gate applied (<2 clean, 2–5 acceptable, >5 RED FLAG) | 2 (coded), 6 (read) |
| A size chosen for Coral, justified by the table | 6 |
| `deploy/` bundle runs on the laptop, correct boxes on a live overhead feed | 9 |
| Calibration uses real cluttered images | 3 (`build_export_kwargs` test) |
| Class-count guard (17) | 3, 4, 8 |
| Camera failure enumerates working indices | 7 |
| Failed export size is reported, never silently omitted | 2, 4 |
