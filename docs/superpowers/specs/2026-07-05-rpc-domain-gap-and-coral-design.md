# Design: Close the RPC Train/Test Domain Gap, then Deploy to Coral Edge TPU

**Date:** 2026-07-05
**Project:** EdgeObjectDetector (YOLO11 on the RPC 200-class retail dataset)
**Status:** Approved design — pending implementation plan

---

## 1. Problem

The `rpc_yolo11n-2` run localizes products well but misclassifies them. Evidence
from `runs/detect/rpc_yolo11n-2/`:

| Signal | Value | Meaning |
|---|---|---|
| `train/cls_loss` | 3.03 → **0.41** | Learns training-set SKUs almost perfectly |
| `val/cls_loss` | 5.33 → **6.09 (rising)** | Classification on val *diverges* |
| `mAP50` | peaks 0.065 @ epoch 3, decays to ~0.05 | Model memorizes studio appearance |
| `precision / recall (B)` | ~0.15 / ~0.08 | Very low |

**Root cause — the defining RPC challenge (a train/test domain gap), not hyperparameters:**

| | Training images | Val / test images |
|---|---|---|
| Content | ONE isolated product | 5–15 densely packed products |
| Setup | Product on a white turntable, studio lighting, multiple camera angles | Real checkout, top-down, cluttered, occluded |
| Labels | Exactly 1 box/image (verified) | Many boxes/image |

Objectness (there is a product here) transfers across domains; fine-grained 200-way
SKU identity learned from isolated turntable shots does not survive the jump to
cluttered checkout scenes. The model never sees a multi-object scene in training.

## 2. Goals & Non-Goals

**Goals**
- Close the domain gap so `val/cls_loss` decreases and `mAP50` rises well above ~0.05.
- Produce a Coral Edge TPU–deployable int8 model once accuracy is real.

**Non-Goals**
- Chasing a specific mAP number (cut-out quality bounds it; direction of `val/cls_loss`
  is the honest pass/fail signal).
- Two-stage detector+classifier architectures (worse for edge; rejected).
- Optimizing/quantizing the current low-accuracy model.

## 3. Approach (decisions made during brainstorming)

- **Fix:** synthesize checkout-like training scenes by compositing isolated products
  (the RPC-standard fix). Chosen over augmentation-only tuning, which cannot manufacture
  realistic checkout clutter from single-item shots.
- **Sequencing:** accuracy first, Coral second — the two phases are **gated**.
- **Background source:** checkout-tray background patches sampled from real val images
  (cleanest labels, matches the deployment domain).
- **Cut-out method:** OpenCV **GrabCut** seeded from each product's bounding box.
- **Training mix:** synthetic scenes **+ a small fraction of single-item images** (canonical
  class appearance).
- **Capacity probe:** run `yolo11s` once to measure headroom; keep `yolo11n` as the Coral
  deployment target.

## 4. Architecture

```
Phase 1 (accuracy):
  extract_cutouts.py  -> cutouts/<class_id>/*.png   (GrabCut RGBA products)
  harvest_backgrounds.py -> backgrounds/*.png       (empty tray patches from val)
  compose_scenes.py   -> dataset_synth/{images,labels}/train + dataset_synth.yml
  (verify)            -> drawn-box previews + class-balance histogram
  train.py --data dataset_synth.yml   (yolo11n; plus a yolo11s probe run)
  validate on REAL val -> val/cls_loss must decrease, mAP50 must climb

  === GATE: proceed only if val accuracy is materially improved ===

Phase 2 (Coral, gated):
  export_edgetpu.py -> int8 TFLite (calibrated) -> edgetpu_compiler -> *_edgetpu.tflite
  validate int8 mAP vs float mAP; measure on-device latency
```

Each Phase-1 stage is a standalone script writing inspectable output, so we can verify
between stages and rerun any one in isolation.

## 5. Component Design (Phase 1)

### 5.1 `extract_cutouts.py`
- **Input:** `dataset/` full-res single-item images + YOLO labels (1 box each).
- **Process:** denormalize the box → seed `cv2.grabCut` with that rect → binary mask →
  write RGBA PNG (product pixels + alpha) to `cutouts/<class_id>/<stem>.png`.
- **Subsample:** cap ~40 camera-angle views per class to bound library size while keeping
  rotation diversity (there are many near-duplicate `cameraN-XX` frames per SKU).
- **Robustness:** per-image try/except; a failed GrabCut is skipped and tallied, never
  aborts the batch (same discipline as `resize_dataset.py`). Multiprocessing via
  `ProcessPoolExecutor`.
- **Output:** cut-out library + printed per-class cut-out counts.
- **Interface:** `--src dataset --out cutouts --per-class 40 --workers 0`.

### 5.2 `harvest_backgrounds.py`
- **Input:** val images + val labels.
- **Process:** sample fixed-size patches (e.g. 128×128) from regions with **zero** bbox
  overlap → clean gray-tray tiles → `backgrounds/*.png`.
- **Output:** background-tile library (target a few thousand tiles).
- **No leakage:** only product-*free* tray texture is sampled (zero bbox overlap). No val
  product pixels or labels enter training — the pasted products come from the *train* cut-out
  library, so the val set remains an honest held-out measure of the target domain.
- **Interface:** `--src dataset/images/val --labels dataset/labels/val --out backgrounds
  --patch 128 --count 4000`.

### 5.3 `compose_scenes.py` (core)
- **Canvas:** 640×640; fill by tiling random background patches + light Gaussian blur on
  seams to hide tile edges.
- **Paste loop:** sample `K ~ Uniform(3,15)` cut-outs with **class-balanced** sampling
  (track per-class paste counts; prefer under-represented classes). For each cut-out:
  random scale (≈15–40% of canvas, matched to val product sizes), random rotation, random
  position; allow partial overlap/occlusion; alpha-blend; light brightness/color jitter.
- **Labels:** emit a YOLO box per pasted product, clipped to canvas. Occluded products keep
  their box (matches val, where overlapping products are all labeled). Drop a box only if a
  later paste covers > ~85% of it.
- **Output:** `dataset_synth/images/train/*.jpg`, `dataset_synth/labels/train/*.txt`,
  and `dataset_synth.yml` (reusing the 200 class names; UTF-8, ASCII hyphen — same care as
  `resize_dataset.py`).
- **Single-item mix:** copy a small fraction (e.g. ~10%) of the original single-item images
  + labels into the synthetic train split for canonical class appearance.
- **Interface:** `--cutouts cutouts --backgrounds backgrounds --out dataset_synth
  --num 20000 --min-objs 3 --max-objs 15 --single-item-frac 0.1 --workers 0`.
- **`val` split:** points at the **untouched real val set** so metrics reflect the true target.

### 5.4 Verification harness
- Reuse the box-drawing verify from `resize_dataset.py` to render a few synthetic images
  with boxes → confirm labels align with pasted products.
- Print the per-class paste-count histogram to confirm balance (no class starved).

### 5.5 Training (Phase 1)
- `python train.py --data dataset_synth.yml` (defaults otherwise as tuned: imgsz 512,
  batch 32, patience 20, auto-GPU).
- **Capacity probe:** one `--model yolo11s.pt --name rpc_yolo11s_probe` run to measure the
  accuracy ceiling; `yolo11n` remains the deployment target.
- Augmentation: keep mosaic (further densifies scenes) with `close_mosaic` near the end;
  standard color aug. No exotic settings — the data now matches the domain.

## 6. Component Design (Phase 2 — Coral, gated)

### 6.1 `export_edgetpu.py`
- Ultralytics: `YOLO(best.pt).export(format='edgetpu', int8=True, data='dataset_synth.yml',
  imgsz=320)` → SavedModel → full-int8 TFLite (representative-dataset calibration drawn
  from our data) → `edgetpu_compiler` → `*_edgetpu.tflite`.
- **imgsz 320** (edge-friendly latency); evaluate the accuracy/speed trade later.

### 6.2 Constraints (called out honestly)
- `edgetpu_compiler` is **Linux-only**. On this Windows host the int8 TFLite exports fine,
  but the final compile step requires **WSL2/Docker or Colab**. The plan will treat the
  compile as a Linux step, not pretend it runs natively on Windows.
- YOLO11 uses SiLU activations and a detect head; the Edge TPU maps the backbone but the
  detect head/postprocess typically fall back to CPU. Report the compiler's op-mapping
  summary rather than assuming full-TPU execution.

### 6.3 Success criteria (Phase 2)
- Compiled `*_edgetpu.tflite` runs.
- int8 mAP within an acceptable delta of the float model (validate with `yolo val` on the
  TFLite before and after compilation).
- On-device latency measured and reported.

## 7. Success Criteria (Phase 1 — the gate)

- `val/cls_loss` **decreases** across epochs (no longer diverging) — the primary signal.
- `mAP50` climbs materially above the current ~0.05.
- `val_batchN_pred.jpg` shows correct class names on real checkout scenes.

## 8. New / Reused Files

**New:** `extract_cutouts.py`, `harvest_backgrounds.py`, `compose_scenes.py`,
`dataset_synth.yml` (generated), `export_edgetpu.py` (Phase 2).
**Reused:** `train.py` (point `--data` at `dataset_synth.yml`), `resize_dataset.py`
verify helper, existing `dataset/` originals + labels.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| GrabCut masks noisy on light/white products | Tally failures; spot-check previews; fall back to bbox crop for low-alpha masks |
| Synthetic scenes still don't match val lighting/scale | Match K, scale, and tray color to measured val stats; verify visually before a full train |
| Nano lacks capacity for 200 fine classes | `yolo11s` probe quantifies the ceiling before committing |
| int8 quantization drops mAP too far | Representative calibration from in-domain data; compare int8 vs float mAP; consider larger imgsz if needed |
| `edgetpu_compiler` unavailable on Windows | Do the compile in WSL2/Docker/Colab — planned as a Linux step |

## 10. Notes

- Repository is not under git (`git init` not yet run), so this design is saved to disk but
  not committed. Recommend `git init` before implementation so work is tracked.
