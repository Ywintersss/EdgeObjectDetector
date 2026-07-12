# INT8 Export + Laptop Overhead-Camera Demo — Design

**Date:** 2026-07-12
**Status:** Approved (pending spec review)
**Phase:** 1 of 2. Phase 2 (Coral Dev Board Mini deployment) gets its own spec.

## Problem

We have a converged YOLO11n coarse-17 detector (`runs/detect/rpc_real_blend_b0_full/weights/best.pt`)
verified on held-out data:

| Split | mAP@50 | mAP@50-95 |
|---|---|---|
| Cluttered (`real_eval`, 3021 imgs) | 0.995 | 0.879 |
| Single-item (studio eval, 2000 imgs) | 0.995 | 0.945 |

The target hardware is a **Coral Dev Board Mini**, whose Edge TPU runs **only fully-integer (INT8)**
models. Quantization is therefore mandatory, and it is the step where accuracy silently degrades.

This phase answers two questions **before** any Coral work begins:

1. **What does INT8 quantization cost us in accuracy, at each candidate input size?**
2. **Does the model actually work on a live camera feed?**

## Goals

- Export `best.pt` to INT8 TFLite at **640 / 448 / 320**, calibrated on the real deployment distribution.
- Measure mAP for each size against the same held-out splits used for the PyTorch baseline.
- Produce one comparison table that makes the Coral size decision an informed one.
- Run the chosen model live on a laptop overhead camera and see correct boxes.

## Non-Goals (YAGNI)

- **Coral deployment** (`edgetpu_compiler`, WSL2, Mendel, PyCoral) — Phase 2.
- **ONNX** — not on the Coral path. Excluded deliberately.
- Retraining, architecture changes, or accuracy improvements. The model is fixed.
- Any production UI. The demo is a diagnostic, not a product.

## Critical Domain Constraint: Viewpoint

**RPC cluttered scenes are top-down overhead shots** — camera above a counter, products lying on a
flat plain surface. The model's competence is concentrated at that viewpoint.

A laptop front camera used conventionally (eye-level, horizontal, holding a product up) is
**out-of-domain** and will underperform. This is the same domain-gap failure mode that defeated the
first two training attempts, in a new disguise.

**Therefore the demo is designed for an overhead rig:** camera pointed DOWN at products on a plain
surface. The demo displays an on-screen reminder to that effect. Eye-level results are not a valid
measure of this model.

## Architecture

The desktop does all heavy lifting; the laptop receives only a small model file.

```
DESKTOP (RTX 5060, holds the ~10 GB dataset)     LAPTOP (Windows + NVIDIA, has camera)
┌────────────────────────────────────┐           ┌──────────────────────────┐
│ best.pt (15.9 MB)                  │           │  deploy/                 │
│   │                                │   copy    │   ├─ <chosen>.tflite     │
│   ├─ export_int8.py                │  ~3 MB    │   ├─ classes.txt         │
│   │   ├─ calibrate on REAL         │ ────────► │   ├─ webcam_demo.py      │
│   │   │   cluttered images         │           │   └─ README.md           │
│   │   ├─ export @ 640 / 448 / 320  │           │          │               │
│   │   └─ val each + PT baseline    │           │          ▼               │
│   │                                │           │   OVERHEAD camera        │
│   └─ export/report.md              │           │   → live boxes + FPS     │
│      (the size/accuracy table)     │           │                          │
└────────────────────────────────────┘           └──────────────────────────┘
```

**The dataset never leaves the desktop.** INT8 calibration must read real cluttered images, which
live here. This is the entire reason export runs on the desktop rather than the laptop.

## Components

### 1. `export_int8.py` (desktop) — the measurement tool

Responsibility: produce the size-vs-accuracy table.

For each size in `--sizes 640,448,320`:
- Export `best.pt` → INT8 TFLite, **calibrating on `dataset_real_blend.yml`** (the real cluttered
  deployment distribution).
- Val the exported model on both held-out splits: cluttered (`dataset_real_blend.yml`) and
  single-item (`eval_single_item.yml`).

Also vals the **FP32 PyTorch baseline** for the delta reference.

Emits `export/report.md` — a table of *size → mAP@50 / mAP@50-95 / file size / latency*, with the
delta vs the PyTorch baseline.

**Latency caveat:** latency is measured on the **desktop CPU** and is a *relative* indicator between
sizes only. It is **not** a prediction of Coral throughput — the Dev Board Mini has a different
processor (quad-core Cortex-A35) and an Edge TPU accelerator. Use it to compare 640 vs 320 against
each other, never as an absolute FPS forecast.

**Calibration is the load-bearing detail.** If INT8 calibration samples the wrong distribution
(e.g. studio singles), the quantization ranges are tuned for a domain we do not deploy in. The
resulting accuracy loss looks like "quantization is lossy" but is actually "we calibrated on the
wrong data."

### 2. `deploy/` bundle — the transfer unit

Self-contained; the laptop needs no dataset and no repo checkout.

- The chosen `.tflite`
- `classes.txt` — the 17 coarse names, **derived from the yaml `names:` block**, never hand-typed
- `webcam_demo.py`
- `README.md` with the exact run command

Transferred by **USB stick or cloud drive, not git** — `.gitignore` excludes `*.tflite`, and the
repo should not carry binaries.

### 3. `webcam_demo.py` (laptop) — the live test

- Opens the camera, runs the INT8 TFLite model, draws boxes + class + confidence, overlays live FPS.
- On-screen hint: point the camera DOWN at a plain surface.
- `--camera N` (default 0) and `--model PATH`.

**Key design seam:** `detect_frame(model, frame) → detections` is a **pure function, separate from
the camera loop**. Inference is therefore testable without a camera — feed it a real dataset image
and assert boxes come back. The camera loop is a thin shell around it.

## Error Handling

No silent failures.

| Condition | Behavior |
|---|---|
| `best.pt` or calibration yaml missing | Hard error naming the missing path. No fallback to a default model. |
| One size fails to export | Record as **FAILED** in the report, print prominently, continue other sizes. Never silently omit a row. |
| Camera won't open | Error that **enumerates the working camera indices found**. Prevents silently grabbing the wrong camera (virtual cams / OBS / external webcams shift the index). |
| Class-count mismatch | Assert the exported model outputs exactly **17** classes and `classes.txt` has 17 entries. |

The class-count guard matters most: if class names are out of order or the wrong model is bundled,
**every box is confidently mislabeled and nothing crashes.** That is the worst possible failure —
it looks like success.

## Known Risk: the INT8 export toolchain

Ultralytics performs INT8 TFLite export via `onnx` → `onnx2tf` → `tensorflow`. The environment is
**Python 3.13 + Windows**, the least-traveled combination for that stack. Dependency install or
mid-conversion failure is plausible.

Contingency, in order:

1. Attempt in the current environment.
2. If it breaks → a dedicated **Python 3.11 venv** for export only.
3. If it still breaks → run the export inside **WSL2**, which Phase 2 requires anyway (the
   `edgetpu_compiler` is Linux-x86-only). Not wasted work.

**Mitigation:** validate the export path at a **single size (320, the fastest) before running the
full sweep.** Fail fast on the risky step.

## Testing

Existing `tests/` + pytest convention. TDD: tests first.

**Unit tests (real behavior, no mocks):**
- `parse_sizes("640,448,320")` → `[640, 448, 320]`; rejects malformed input.
- `load_class_names(yaml)` → 17 names **in correct index order** (guards the mislabeling bug).
- `build_report_table(...)` → correct rows, correct delta arithmetic vs baseline.
- `detect_frame(model, frame)` fed a **real cluttered dataset image** → returns ≥1 detection, class
  indices within `[0, 16]`, boxes in-bounds.
- Invalid camera index → raises the enumerating error.

**Integration verification (actually executed, never assert-by-reading):**
- Single-size export end-to-end; confirm `report.md` contains real numbers.
- Then the full sweep.
- The webcam demo is **human-verified** — "does it draw a correct box on a real product" cannot be
  automated. Run it and look.

## Acceptance Criteria

1. `export/report.md` exists with real measured numbers for 640 / 448 / 320 plus the FP32 baseline.
2. **Quantization gate:** mAP@50-95 delta vs the FP32 cluttered baseline (0.879).
   - Drop **< 2 points** → clean. Proceed.
   - Drop **2–5 points** → acceptable; record the cost and proceed, noting it in the report.
   - Drop **> 5 points** → red flag. Stop and investigate; the first suspect is **calibration data**,
     not input size. Do not proceed to Coral on a model that failed this gate.
3. A size is chosen for Coral, justified by the table.
4. The `deploy/` bundle runs on the laptop and draws correct boxes on a live **overhead** feed.

## Open Question Deferred to Phase 2

Edge TPU **op-support / CPU fallback**. The `edgetpu_compiler` reports how many ops map to the TPU
versus fall back to the CPU; ops that fall back destroy latency. This cannot be measured until
Phase 2. Phase 1 deliberately de-risks it by producing smaller candidate models up front.
