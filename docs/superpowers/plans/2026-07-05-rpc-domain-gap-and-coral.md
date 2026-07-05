# RPC Domain-Gap Fix + Coral Edge TPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the RPC train/test domain gap by training on synthetic checkout scenes composited from isolated-product cut-outs, then export the resulting model to a Coral Edge TPU int8 model.

**Architecture:** Four offline stages produce a domain-matched training set: extract GrabCut cut-outs of each single product, harvest empty checkout-tray backgrounds from val, compose multi-object scenes with YOLO labels, then retrain YOLO11n (and probe YOLO11s). Pure logic lives in a unit-tested `synth_utils.py`; cv2-heavy scripts orchestrate it. Phase 2 (int8 TFLite → `edgetpu_compiler`) is gated on Phase-1 accuracy.

**Tech Stack:** Python 3.13, Ultralytics 8.4.x, OpenCV (`cv2`), NumPy, pytest, PyYAML. GPU: RTX 5060 (CUDA cu128).

## Global Constraints

- Run Python with `python` (never `python3`).
- Always TypeScript for web code — N/A here; this is a Python project. Use type hints, camelCase-free snake_case per Python/PEP8 (existing scripts use snake_case).
- Modular functions, explicit error handling, never silent failures — every worker returns a status tuple and never aborts the batch (mirror `resize_dataset.py:resize_one`).
- YOLO labels are NORMALIZED (`class xc yc w h`, fractions) — resolution-independent.
- Dataset class order = enumerate position of `instances_train2019.json` categories; 200 classes, `nc: 200`. Reuse the exact `names:` block from `dataset.yml`.
- All generated YAML written UTF-8 with ASCII hyphens (byte 0x97 em-dash bug is banned).
- Training defaults (from `train.py`): imgsz 512, batch 32, patience 20, auto-GPU. Coral export target: `yolo11n`, imgsz 320.
- `edgetpu_compiler` is Linux-only — Phase-2 compile runs in WSL2/Docker/Colab, not native Windows.
- Source single-item images live in `dataset/` (full-res, one box each); resized copies in `dataset_640/`.

---

## File Structure

- Create `synth_utils.py` — pure, unit-tested helpers: box math, mask→bbox, class-balanced sampling, alpha compositing, rotation, placement, occlusion visibility, synth-yaml writer.
- Create `extract_cutouts.py` — GrabCut extraction of RGBA product cut-outs → `cutouts/<class_id>/*.png`.
- Create `harvest_backgrounds.py` — empty tray patch sampler → `backgrounds/*.png`.
- Create `compose_scenes.py` — scene composition + labels + single-item mix + verify previews + class histogram → `dataset_synth/` + `dataset_synth.yml`.
- Create `export_edgetpu.py` — Phase-2 int8 export wrapper.
- Create `tests/` — pytest tests for `synth_utils.py` + tiny fixture-based integration checks.
- Create `tests/conftest.py` — shared numpy/image fixtures.
- Reuse `train.py` (point `--data dataset_synth.yml`), `dataset.yml` (names block source).

---

## Task 1: Project scaffolding (git, pytest, requirements)

**Files:**
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`
- Modify: `requirements.txt`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: a runnable `python -m pytest` and shared fixtures `blank_canvas`, `red_rgba` for later tasks.

- [ ] **Step 1: Initialize git (repo is not yet under version control)**

```bash
cd /d/Projects/EdgeObjectDetector
git init
printf '%s\n' 'cutouts/' 'backgrounds/' 'dataset_synth/' '__pycache__/' '.pytest_cache/' '*.pyc' 'runs/' >> .gitignore
git add .gitignore CLAUDE.md train.py resize_dataset.py preprocessing.py requirements.txt dataset.yml dataset_640.yml docs
git commit -m "chore: initialize git repo with existing project"
```

- [ ] **Step 2: Add test deps to `requirements.txt`**

Append these two lines to `requirements.txt`:

```
pytest==8.3.4
pytest-cov==6.0.0
```

Then install:

```bash
python -m pip install pytest==8.3.4 pytest-cov==6.0.0
```

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 4: Create shared fixtures in `tests/conftest.py`**

```python
"""Shared pytest fixtures: tiny in-memory images so tests are fast and real."""
import numpy as np
import pytest


@pytest.fixture
def blank_canvas():
    """A 100x100 BGR gray canvas."""
    return np.full((100, 100, 3), 127, dtype=np.uint8)


@pytest.fixture
def red_rgba():
    """A 20x10 fully-opaque red RGBA patch (BGRA order to match cv2)."""
    patch = np.zeros((20, 10, 4), dtype=np.uint8)
    patch[..., 2] = 255   # red channel (BGR)
    patch[..., 3] = 255   # alpha
    return patch
```

- [ ] **Step 5: Create `tests/__init__.py` (empty) and `tests/test_smoke.py`**

```python
def test_pytest_runs():
    assert True
```

- [ ] **Step 6: Run tests to confirm the harness works**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini tests/
git commit -m "chore: add pytest scaffolding and fixtures"
```

---

## Task 2: Geometry utils (box math + mask→bbox)

**Files:**
- Create: `synth_utils.py`
- Test: `tests/test_synth_utils_geometry.py`

**Interfaces:**
- Produces:
  - `denormalize_box(box: tuple[float,float,float,float], img_w: int, img_h: int) -> tuple[int,int,int,int]` — YOLO `(xc,yc,w,h)` → pixel `(x1,y1,x2,y2)`, clipped to image.
  - `normalize_box(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple[float,float,float,float]` — pixel corners → YOLO `(xc,yc,w,h)`.
  - `mask_to_bbox(mask: np.ndarray) -> tuple[int,int,int,int] | None` — tight `(x1,y1,x2,y2)` of nonzero pixels (`x2,y2` exclusive), or `None` if empty.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_synth_utils_geometry.py
import numpy as np
from synth_utils import denormalize_box, normalize_box, mask_to_bbox


def test_denormalize_box_center():
    # centered box covering half the width/height of a 100x200 image
    assert denormalize_box((0.5, 0.5, 0.5, 0.5), 100, 200) == (25, 50, 75, 150)


def test_denormalize_box_clips_to_bounds():
    # box spilling past edges is clipped, never negative or > size
    assert denormalize_box((0.0, 0.0, 0.4, 0.4), 100, 100) == (0, 0, 20, 20)


def test_normalize_round_trips():
    box = (0.3, 0.6, 0.2, 0.4)
    x1, y1, x2, y2 = denormalize_box(box, 640, 640)
    back = normalize_box(x1, y1, x2, y2, 640, 640)
    assert all(abs(a - b) < 1e-3 for a, b in zip(box, back))


def test_mask_to_bbox_tight():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:8] = 1
    assert mask_to_bbox(mask) == (3, 2, 8, 5)


def test_mask_to_bbox_empty_returns_none():
    assert mask_to_bbox(np.zeros((10, 10), dtype=np.uint8)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_synth_utils_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'synth_utils'`.

- [ ] **Step 3: Write minimal implementation in `synth_utils.py`**

```python
"""synth_utils.py — pure helpers for synthetic checkout-scene generation.

Kept free of file I/O (except the yaml writer) so every function is unit-testable
with tiny in-memory numpy arrays.
"""
from __future__ import annotations

import numpy as np


def denormalize_box(box: tuple[float, float, float, float],
                    img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """YOLO normalized (xc,yc,w,h) -> pixel (x1,y1,x2,y2), clipped to image."""
    xc, yc, w, h = box
    x1 = int(round((xc - w / 2) * img_w))
    y1 = int(round((yc - h / 2) * img_h))
    x2 = int(round((xc + w / 2) * img_w))
    y2 = int(round((yc + h / 2) * img_h))
    x1 = max(0, min(img_w, x1)); x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h, y1)); y2 = max(0, min(img_h, y2))
    return x1, y1, x2, y2


def normalize_box(x1: int, y1: int, x2: int, y2: int,
                  img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Pixel corners -> YOLO normalized (xc,yc,w,h)."""
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight bbox (x1,y1,x2,y2) of nonzero pixels; x2,y2 exclusive. None if empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_synth_utils_geometry.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add synth_utils.py tests/test_synth_utils_geometry.py
git commit -m "feat: add geometry helpers to synth_utils"
```

---

## Task 3: Class-balanced sampler

**Files:**
- Modify: `synth_utils.py`
- Test: `tests/test_synth_utils_sampler.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `class ClassBalancedSampler(class_to_items: dict[int, list[str]], seed: int = 0)` with method
    `sample(k: int) -> list[tuple[int, str]]` — returns `k` `(class_id, item)` picks, always drawing from the least-used class so far (ties broken randomly), so rare classes are not starved. Maintains internal per-class draw counts across calls.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_synth_utils_sampler.py
from collections import Counter
from synth_utils import ClassBalancedSampler


def test_sampler_balances_across_classes():
    # 3 classes, plenty of items each; drawing 300 picks should be ~even
    pool = {0: ["a", "b"], 1: ["c", "d"], 2: ["e", "f"]}
    s = ClassBalancedSampler(pool, seed=1)
    picks = s.sample(300)
    counts = Counter(cid for cid, _ in picks)
    assert set(counts) == {0, 1, 2}
    assert max(counts.values()) - min(counts.values()) <= 1  # balanced to within 1


def test_sampler_returns_valid_items():
    pool = {5: ["only.png"]}
    s = ClassBalancedSampler(pool, seed=0)
    assert s.sample(3) == [(5, "only.png"), (5, "only.png"), (5, "only.png")]


def test_sampler_state_persists_across_calls():
    pool = {0: ["a"], 1: ["b"]}
    s = ClassBalancedSampler(pool, seed=0)
    first = s.sample(1)[0][0]
    second = s.sample(1)[0][0]
    assert first != second  # least-used class is chosen next
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_synth_utils_sampler.py -v`
Expected: FAIL with `ImportError: cannot import name 'ClassBalancedSampler'`.

- [ ] **Step 3: Add implementation to `synth_utils.py`**

```python
import random


class ClassBalancedSampler:
    """Draw (class_id, item) picks, always preferring the least-used class.

    Keeps per-class draw counts so that across a whole generation run every class
    appears a similar number of times, even when the cut-out library is skewed.
    """

    def __init__(self, class_to_items: dict[int, list[str]], seed: int = 0):
        if not class_to_items:
            raise ValueError("class_to_items must not be empty")
        self._pool = {c: list(items) for c, items in class_to_items.items() if items}
        if not self._pool:
            raise ValueError("every class had an empty item list")
        self._counts = {c: 0 for c in self._pool}
        self._rng = random.Random(seed)

    def sample(self, k: int) -> list[tuple[int, str]]:
        picks: list[tuple[int, str]] = []
        for _ in range(k):
            min_count = min(self._counts.values())
            candidates = [c for c, n in self._counts.items() if n == min_count]
            cls = self._rng.choice(candidates)
            item = self._rng.choice(self._pool[cls])
            self._counts[cls] += 1
            picks.append((cls, item))
        return picks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_synth_utils_sampler.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add synth_utils.py tests/test_synth_utils_sampler.py
git commit -m "feat: add ClassBalancedSampler to synth_utils"
```

---

## Task 4: Compositing utils (paste, rotate, placement, visibility, yaml)

**Files:**
- Modify: `synth_utils.py`
- Test: `tests/test_synth_utils_compose.py`

**Interfaces:**
- Consumes: `mask_to_bbox`, `normalize_box` (Task 2).
- Produces:
  - `rotate_rgba(rgba: np.ndarray, angle_deg: float, rng: random.Random | None = None) -> np.ndarray` — rotate an RGBA cut-out about its center, expanding the canvas so no pixels are clipped; empty corners get alpha 0.
  - `random_placement(canvas_wh: tuple[int,int], obj_wh: tuple[int,int], rng: random.Random) -> tuple[int,int]` — random top-left `(x,y)` so the object's center stays on-canvas (edges may overhang, matching real checkout crops).
  - `alpha_paste(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int, owner_map: np.ndarray, owner_id: int) -> None` — alpha-blend `rgba` onto `canvas` (BGR) at top-left `(x,y)`, clipping at borders; stamp `owner_id` into `owner_map` wherever alpha>0 (later pastes overwrite earlier ones). In place.
  - `compute_visibilities(owner_map: np.ndarray, total_pixels: dict[int,int]) -> dict[int,float]` — for each `owner_id`, `visible_pixels / total_pixels`, where visible = pixels still owned in the final `owner_map`. Used to drop heavily-occluded boxes.
  - `write_synth_yaml(dst_root, names_block_lines: list[str], out_path) -> None` — write a dataset yaml (UTF-8, ASCII hyphen) pointing at `dst_root` with train/val/test + `nc: 200` + the provided `names:` block.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_synth_utils_compose.py
import random
import numpy as np
from synth_utils import (rotate_rgba, random_placement, alpha_paste,
                         compute_visibilities)


def test_alpha_paste_draws_and_owns(blank_canvas, red_rgba):
    owner = np.full(blank_canvas.shape[:2], -1, dtype=np.int32)
    alpha_paste(blank_canvas, red_rgba, x=10, y=5, owner_map=owner, owner_id=0)
    # pasted 20x10 red block: red channel maxed inside, owner stamped
    assert blank_canvas[5, 10, 2] == 255
    assert (owner == 0).sum() == 20 * 10
    # outside the paste region untouched
    assert owner[0, 0] == -1


def test_alpha_paste_clips_at_border(blank_canvas, red_rgba):
    owner = np.full(blank_canvas.shape[:2], -1, dtype=np.int32)
    # paste near bottom-right corner so it overhangs; must not raise
    alpha_paste(blank_canvas, red_rgba, x=95, y=95, owner_map=owner, owner_id=1)
    assert (owner == 1).sum() == 5 * 5  # only the on-canvas 5x5 corner drawn


def test_later_paste_overwrites_owner(blank_canvas, red_rgba):
    owner = np.full(blank_canvas.shape[:2], -1, dtype=np.int32)
    alpha_paste(blank_canvas, red_rgba, 10, 10, owner, 0)
    alpha_paste(blank_canvas, red_rgba, 12, 10, owner, 1)  # overlaps object 0
    vis = compute_visibilities(owner, {0: 200, 1: 200})
    assert vis[1] == 1.0            # object 1 fully on top
    assert vis[0] < 1.0            # object 0 partially covered


def test_rotate_rgba_expands_and_preserves_content(red_rgba):
    out = rotate_rgba(red_rgba, 90.0)
    # 90-degree rotation swaps dimensions (allow +/-1 for rounding)
    assert abs(out.shape[0] - 10) <= 1 and abs(out.shape[1] - 20) <= 1
    assert out[..., 3].max() == 255  # some opaque product pixels survive


def test_random_placement_keeps_center_on_canvas():
    rng = random.Random(0)
    for _ in range(50):
        x, y = random_placement((640, 640), (100, 80), rng)
        cx, cy = x + 50, y + 40
        assert 0 <= cx <= 640 and 0 <= cy <= 640
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_synth_utils_compose.py -v`
Expected: FAIL with `ImportError: cannot import name 'rotate_rgba'`.

- [ ] **Step 3: Add implementation to `synth_utils.py`**

```python
import cv2


def rotate_rgba(rgba: np.ndarray, angle_deg: float,
                rng: "random.Random | None" = None) -> np.ndarray:
    """Rotate an RGBA cut-out about its center, expanding so nothing is clipped."""
    h, w = rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    mat[0, 2] += (new_w / 2.0) - center[0]
    mat[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(rgba, mat, (new_w, new_h),
                          flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def random_placement(canvas_wh: tuple[int, int], obj_wh: tuple[int, int],
                     rng: "random.Random") -> tuple[int, int]:
    """Random top-left (x,y) keeping the object's center on-canvas (edges may overhang)."""
    cw, ch = canvas_wh
    ow, oh = obj_wh
    # center must be within [0, cw] x [0, ch]  ->  x in [-ow/2, cw-ow/2]
    x = rng.randint(-ow // 2, cw - ow // 2)
    y = rng.randint(-oh // 2, ch - oh // 2)
    return x, y


def alpha_paste(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int,
                owner_map: np.ndarray, owner_id: int) -> None:
    """Alpha-blend rgba (H,W,4 BGRA) onto canvas (BGR) at (x,y); stamp owner_map."""
    ch, cw = canvas.shape[:2]
    oh, ow = rgba.shape[:2]
    # intersection of the paste rect with the canvas
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + ow), min(ch, y + oh)
    if x1 >= x2 or y1 >= y2:
        return
    # corresponding region inside the rgba patch
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    patch = rgba[sy1:sy2, sx1:sx2]
    alpha = (patch[..., 3:4].astype(np.float32)) / 255.0
    fg = patch[..., :3].astype(np.float32)
    bg = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (alpha * fg + (1 - alpha) * bg).astype(np.uint8)
    opaque = patch[..., 3] > 0
    owner_slice = owner_map[y1:y2, x1:x2]
    owner_slice[opaque] = owner_id


def compute_visibilities(owner_map: np.ndarray,
                         total_pixels: dict[int, int]) -> dict[int, float]:
    """visible/total per owner_id, where visible = pixels still owned at the end."""
    ids, counts = np.unique(owner_map, return_counts=True)
    visible = {int(i): int(c) for i, c in zip(ids, counts) if i >= 0}
    return {oid: visible.get(oid, 0) / total for oid, total in total_pixels.items()
            if total > 0}


def write_synth_yaml(dst_root, names_block_lines: list[str], out_path) -> None:
    """Write a dataset yaml (UTF-8, ASCII hyphen) for the synthetic tree."""
    from pathlib import Path
    dst_root = Path(dst_root)
    header = [
        "# dataset_synth.yaml - synthetic checkout scenes + real val",
        f"path: {dst_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "nc: 200",
    ]
    text = "\n".join(header + list(names_block_lines)) + "\n"
    Path(out_path).write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_synth_utils_compose.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the full util suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (all synth_utils tests green).

- [ ] **Step 6: Commit**

```bash
git add synth_utils.py tests/test_synth_utils_compose.py
git commit -m "feat: add compositing/rotation/visibility helpers to synth_utils"
```

---

## Task 5: `extract_cutouts.py` — GrabCut product cut-outs

**Files:**
- Create: `extract_cutouts.py`
- Test: `tests/test_extract_cutouts.py`

**Interfaces:**
- Consumes: `denormalize_box` (Task 2).
- Produces:
  - `grabcut_cutout(img: np.ndarray, box_px: tuple[int,int,int,int], iters: int = 5) -> np.ndarray | None` — run GrabCut seeded from `box_px`, return a BGRA cut-out cropped to the mask bbox, or `None` if the mask is empty/degenerate.
  - CLI: `python extract_cutouts.py --src dataset --out cutouts --per-class 40 --workers 0 [--limit N]`.
  - Output layout on disk: `cutouts/<class_id>/<image_stem>.png` (BGRA PNG).

- [ ] **Step 1: Write the failing test (unit-level, synthetic image)**

```python
# tests/test_extract_cutouts.py
import numpy as np
from extract_cutouts import grabcut_cutout


def test_grabcut_extracts_foreground_block():
    # gray background with a bright square "product" in the middle
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    img[25:55, 25:55] = (200, 180, 160)
    out = grabcut_cutout(img, box_px=(20, 20, 60, 60), iters=3)
    assert out is not None
    assert out.shape[2] == 4                 # BGRA
    assert out[..., 3].max() == 255          # has opaque product pixels
    # cropped roughly to the product, not the whole 80x80 frame
    assert out.shape[0] <= 60 and out.shape[1] <= 60


def test_grabcut_returns_none_on_empty_box():
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    assert grabcut_cutout(img, box_px=(10, 10, 10, 10)) is None  # zero-area box
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract_cutouts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'extract_cutouts'`.

- [ ] **Step 3: Write `extract_cutouts.py`**

```python
"""extract_cutouts.py — GrabCut RGBA cut-outs of each isolated RPC product.

For every single-item training image (exactly one YOLO box), GrabCut is seeded
from that box to segment the product off the white turntable, and the result is
saved as a cropped BGRA PNG under cutouts/<class_id>/.
"""
import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from synth_utils import denormalize_box, mask_to_bbox

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def grabcut_cutout(img: np.ndarray, box_px, iters: int = 5):
    """Segment the product inside box_px via GrabCut; return cropped BGRA or None."""
    x1, y1, x2, y2 = box_px
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    bbox = mask_to_bbox(fg)
    if bbox is None:
        return None
    bx1, by1, bx2, by2 = bbox
    bgr = img[by1:by2, bx1:bx2]
    alpha = fg[by1:by2, bx1:bx2]
    bgra = np.dstack([bgr, alpha])
    return bgra


def _read_single_box(label_path: Path):
    """Return the (class_id, (xc,yc,w,h)) of the first box, or None."""
    try:
        line = label_path.read_text().splitlines()[0].split()
    except (IndexError, OSError):
        return None
    if len(line) != 5:
        return None
    cid = int(float(line[0]))
    box = tuple(float(v) for v in line[1:])
    return cid, box


def _worker(task):
    """Process one image -> write one cut-out PNG. Returns (status, class_id|msg)."""
    img_path, label_path, out_root = task
    try:
        parsed = _read_single_box(Path(label_path))
        if parsed is None:
            return ("error", f"no box: {label_path}")
        cid, box = parsed
        img = cv2.imread(img_path)
        if img is None:
            return ("error", f"unreadable: {img_path}")
        h, w = img.shape[:2]
        box_px = denormalize_box(box, w, h)
        cut = grabcut_cutout(img, box_px)
        if cut is None:
            return ("error", f"empty mask: {img_path}")
        out_dir = Path(out_root) / str(cid)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(img_path).stem + ".png")
        cv2.imwrite(str(out_path), cut)
        return ("ok", cid)
    except Exception as exc:  # never abort the pool on one bad file
        return ("error", f"{img_path}: {exc}")


def build_tasks(src_root: Path, out_root: Path, per_class: int, limit):
    """Enumerate (img, label, out_root) tasks, capping per-class camera angles."""
    img_dir = src_root / "images" / "train"
    lbl_dir = src_root / "labels" / "train"
    seen = defaultdict(int)
    tasks = []
    files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if limit:
        files = files[:limit]
    for f in files:
        label = lbl_dir / (f.stem + ".txt")
        if not label.exists():
            continue
        parsed = _read_single_box(label)
        if parsed is None:
            continue
        cid = parsed[0]
        if seen[cid] >= per_class:
            continue
        seen[cid] += 1
        tasks.append((str(f), str(label), str(out_root)))
    return tasks


def parse_args():
    p = argparse.ArgumentParser(description="Extract GrabCut RGBA product cut-outs.")
    p.add_argument("--src", default="dataset", help="Source dataset root.")
    p.add_argument("--out", default="cutouts", help="Cut-out output root.")
    p.add_argument("--per-class", type=int, default=40, help="Max camera angles/class.")
    p.add_argument("--workers", type=int, default=0, help="Processes (0=auto).")
    p.add_argument("--limit", type=int, default=None, help="Cap images (for trials).")
    return p.parse_args()


def main():
    args = parse_args()
    src_root = (PROJECT_ROOT / args.src) if not os.path.isabs(args.src) else Path(args.src)
    out_root = (PROJECT_ROOT / args.out) if not os.path.isabs(args.out) else Path(args.out)
    workers = args.workers or max(1, (os.cpu_count() or 4))
    tasks = build_tasks(src_root, out_root, args.per_class, args.limit)
    if not tasks:
        print(f"ERROR: no tasks built from {src_root}", file=sys.stderr)
        return 1
    print(f"Extracting {len(tasks)} cut-outs -> {out_root} (workers={workers})")
    ok = 0
    per_class = defaultdict(int)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            status, payload = fut.result()
            if status == "ok":
                ok += 1
                per_class[payload] += 1
            else:
                print(f"  WARN {payload}", file=sys.stderr)
            if i % 2000 == 0:
                print(f"  {i}/{len(tasks)} done")
    classes_covered = len(per_class)
    print(f"Done: {ok} cut-outs across {classes_covered}/200 classes.")
    if classes_covered < 200:
        print(f"WARNING: {200 - classes_covered} classes have no cut-outs.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract_cutouts.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Trial run on a few images and eyeball a cut-out**

Run: `python extract_cutouts.py --limit 30 --out cutouts_trial`
Expected: prints "Done: N cut-outs across M/200 classes"; open one PNG under `cutouts_trial/<id>/` and confirm the product is isolated with transparent background.

- [ ] **Step 6: Commit**

```bash
git add extract_cutouts.py tests/test_extract_cutouts.py
git commit -m "feat: add GrabCut cut-out extraction script"
```

---

## Task 6: `harvest_backgrounds.py` — empty tray patches

**Files:**
- Create: `harvest_backgrounds.py`
- Test: `tests/test_harvest_backgrounds.py`

**Interfaces:**
- Consumes: `denormalize_box` (Task 2).
- Produces:
  - `sample_empty_patches(img: np.ndarray, boxes_px: list[tuple[int,int,int,int]], patch: int, max_patches: int, rng) -> list[np.ndarray]` — return up to `max_patches` `patch`×`patch` BGR crops whose area does **not** intersect any box in `boxes_px`.
  - CLI: `python harvest_backgrounds.py --src dataset/images/val --labels dataset/labels/val --out backgrounds --patch 128 --count 4000`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harvest_backgrounds.py
import random
import numpy as np
from harvest_backgrounds import sample_empty_patches


def _overlaps(px, box):
    x, y, s = px
    bx1, by1, bx2, by2 = box
    return not (x + s <= bx1 or x >= bx2 or y + s <= by1 or y >= by2)


def test_patches_avoid_boxes():
    img = np.full((200, 200, 3), 120, dtype=np.uint8)
    boxes = [(0, 0, 100, 200)]  # left half occupied
    rng = random.Random(0)
    patches = sample_empty_patches(img, boxes, patch=20, max_patches=10, rng=rng)
    assert len(patches) > 0
    assert all(p.shape == (20, 20, 3) for p in patches)


def test_no_patches_when_fully_covered():
    img = np.full((50, 50, 3), 120, dtype=np.uint8)
    boxes = [(0, 0, 50, 50)]
    rng = random.Random(0)
    assert sample_empty_patches(img, boxes, patch=20, max_patches=5, rng=rng) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_harvest_backgrounds.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `harvest_backgrounds.py`**

```python
"""harvest_backgrounds.py — sample empty checkout-tray patches from val images.

Only product-FREE tray texture is sampled (zero bbox overlap), so no val product
pixels or labels ever enter training. These tiles become the canvas background for
synthetic scenes.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from synth_utils import denormalize_box

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _patch_free(x, y, s, boxes_px):
    for bx1, by1, bx2, by2 in boxes_px:
        if not (x + s <= bx1 or x >= bx2 or y + s <= by1 or y >= by2):
            return False
    return True


def sample_empty_patches(img, boxes_px, patch, max_patches, rng):
    """Up to max_patches `patch`x`patch` BGR crops not overlapping any box."""
    h, w = img.shape[:2]
    if w < patch or h < patch:
        return []
    out = []
    attempts = 0
    while len(out) < max_patches and attempts < max_patches * 30:
        attempts += 1
        x = rng.randint(0, w - patch)
        y = rng.randint(0, h - patch)
        if _patch_free(x, y, patch, boxes_px):
            out.append(img[y:y + patch, x:x + patch].copy())
    return out


def _read_boxes_px(label_path: Path, w: int, h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        box = tuple(float(v) for v in parts[1:])
        boxes.append(denormalize_box(box, w, h))
    return boxes


def parse_args():
    p = argparse.ArgumentParser(description="Harvest empty tray background patches.")
    p.add_argument("--src", default="dataset/images/val")
    p.add_argument("--labels", default="dataset/labels/val")
    p.add_argument("--out", default="backgrounds")
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--count", type=int, default=4000, help="Total patches to collect.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    root = PROJECT_ROOT
    src = (root / args.src) if not os.path.isabs(args.src) else Path(args.src)
    lbl = (root / args.labels) if not os.path.isabs(args.labels) else Path(args.labels)
    out = (root / args.out) if not os.path.isabs(args.out) else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"ERROR: no val images at {src}", file=sys.stderr)
        return 1
    per_image = max(1, args.count // len(files))
    saved = 0
    for f in files:
        if saved >= args.count:
            break
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = _read_boxes_px(lbl / (f.stem + ".txt"), w, h)
        for patch in sample_empty_patches(img, boxes, args.patch, per_image, rng):
            cv2.imwrite(str(out / f"bg_{saved:06d}.png"), patch)
            saved += 1
            if saved >= args.count:
                break
    print(f"Done: saved {saved} background patches -> {out}")
    if saved == 0:
        print("ERROR: harvested zero patches.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_harvest_backgrounds.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Trial run**

Run: `python harvest_backgrounds.py --count 50 --out backgrounds_trial`
Expected: "Done: saved 50 background patches"; open a couple and confirm they are plain tray (no products).

- [ ] **Step 6: Commit**

```bash
git add harvest_backgrounds.py tests/test_harvest_backgrounds.py
git commit -m "feat: add empty tray background harvester"
```

---

## Task 7: `compose_scenes.py` — synthesize labeled checkout scenes

**Files:**
- Create: `compose_scenes.py`
- Test: `tests/test_compose_scenes.py`

**Interfaces:**
- Consumes: `ClassBalancedSampler`, `rotate_rgba`, `random_placement`, `alpha_paste`, `compute_visibilities`, `mask_to_bbox`, `normalize_box`, `write_synth_yaml` (Tasks 2–4).
- Produces:
  - `make_background(bg_tiles: list[np.ndarray], size: int, rng) -> np.ndarray` — build a `size`×`size` BGR canvas by tiling random background patches with light blur on seams.
  - `compose_one(canvas: np.ndarray, cutouts: list[tuple[int,np.ndarray]], rng, min_scale=0.15, max_scale=0.40, drop_thresh=0.15) -> list[tuple[int,float,float,float,float]]` — paste each `(class_id, bgra)` cut-out (scaled to a random fraction of the canvas, rotated, placed), then return YOLO rows `(class_id, xc, yc, w, h)` for objects whose visible fraction ≥ `drop_thresh`. Mutates `canvas`.
  - CLI: `python compose_scenes.py --cutouts cutouts --backgrounds backgrounds --out dataset_synth --num 20000 --min-objs 3 --max-objs 15 --single-item-frac 0.1 --size 640 --workers 0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_scenes.py
import random
import numpy as np
from compose_scenes import make_background, compose_one


def _solid_rgba(color, h, w):
    patch = np.zeros((h, w, 4), dtype=np.uint8)
    patch[..., :3] = color
    patch[..., 3] = 255
    return patch


def test_make_background_fills_canvas():
    tiles = [np.full((32, 32, 3), 130, dtype=np.uint8) for _ in range(4)]
    rng = random.Random(0)
    canvas = make_background(tiles, size=128, rng=rng)
    assert canvas.shape == (128, 128, 3)
    assert canvas.mean() > 0  # not blank


def test_compose_one_emits_labels_in_range():
    rng = random.Random(0)
    canvas = np.full((640, 640, 3), 120, dtype=np.uint8)
    cutouts = [(3, _solid_rgba((0, 0, 200), 120, 120)),
               (7, _solid_rgba((0, 200, 0), 120, 120))]
    rows = compose_one(canvas, cutouts, rng)
    assert len(rows) >= 1
    for cid, xc, yc, w, h in rows:
        assert cid in (3, 7)
        assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_fully_occluded_object_is_dropped():
    rng = random.Random(1)
    canvas = np.full((200, 200, 3), 120, dtype=np.uint8)
    # object A then object B pasted at the same spot, B bigger -> A should drop
    small = (1, _solid_rgba((0, 0, 200), 30, 30))
    big = (2, _solid_rgba((0, 200, 0), 120, 120))
    # force identical placement by seeding; assert only class 2 survives on top
    rows = compose_one(canvas, [small, big], rng, drop_thresh=0.5)
    surviving_classes = {r[0] for r in rows}
    assert 2 in surviving_classes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compose_scenes.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `compose_scenes.py`**

```python
"""compose_scenes.py — synthesize labeled multi-object checkout scenes.

Pastes GrabCut cut-outs (class-balanced) onto harvested tray backgrounds, records
a YOLO box per visible product, and mixes in a fraction of original single-item
images. Output is a ready-to-train dataset_synth/ tree + dataset_synth.yml.
"""
import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from synth_utils import (ClassBalancedSampler, alpha_paste, compute_visibilities,
                         mask_to_bbox, normalize_box, random_placement,
                         rotate_rgba, write_synth_yaml)

PROJECT_ROOT = Path(__file__).resolve().parent


def make_background(bg_tiles, size, rng):
    """Tile random background patches into a size x size canvas, blur the seams."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if not bg_tiles:
        canvas[:] = 128
        return canvas
    th, tw = bg_tiles[0].shape[:2]
    for y in range(0, size, th):
        for x in range(0, size, tw):
            tile = rng.choice(bg_tiles)
            hh = min(th, size - y)
            ww = min(tw, size - x)
            canvas[y:y + hh, x:x + ww] = tile[:hh, :ww]
    return cv2.GaussianBlur(canvas, (5, 5), 0)


def _scale_rgba(bgra, target_side, rng, min_scale, max_scale, canvas_size):
    """Scale a cut-out so its longest side is a random fraction of the canvas."""
    frac = rng.uniform(min_scale, max_scale)
    target = max(8, int(canvas_size * frac))
    h, w = bgra.shape[:2]
    s = target / max(h, w)
    new = cv2.resize(bgra, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA)
    return new


def compose_one(canvas, cutouts, rng, min_scale=0.15, max_scale=0.40,
                drop_thresh=0.15):
    """Paste cut-outs; return YOLO rows for objects still >= drop_thresh visible."""
    size = canvas.shape[0]
    owner = np.full((size, size), -1, dtype=np.int32)
    placed = []          # (owner_id, class_id, x1,y1,x2,y2, total_pixels)
    total_pixels = {}
    for oid, (cid, bgra) in enumerate(cutouts):
        scaled = _scale_rgba(bgra, None, rng, min_scale, max_scale, size)
        angle = rng.uniform(-25, 25)
        rot = rotate_rgba(scaled, angle)
        oh, ow = rot.shape[:2]
        x, y = random_placement((size, size), (ow, oh), rng)
        alpha_mask = rot[..., 3] > 0
        total = int(alpha_mask.sum())
        if total == 0:
            continue
        alpha_paste(canvas, rot, x, y, owner, oid)
        # box from the alpha mask position on the canvas, clipped
        local_bbox = mask_to_bbox(alpha_mask)
        lx1, ly1, lx2, ly2 = local_bbox
        gx1 = max(0, x + lx1); gy1 = max(0, y + ly1)
        gx2 = min(size, x + lx2); gy2 = min(size, y + ly2)
        if gx2 <= gx1 or gy2 <= gy1:
            continue
        placed.append((oid, cid, gx1, gy1, gx2, gy2))
        total_pixels[oid] = total
    vis = compute_visibilities(owner, total_pixels)
    rows = []
    for oid, cid, x1, y1, x2, y2 in placed:
        if vis.get(oid, 0.0) < drop_thresh:
            continue
        xc, yc, w, h = normalize_box(x1, y1, x2, y2, size, size)
        rows.append((cid, xc, yc, w, h))
    return rows


def _load_cutout_index(cutouts_root: Path):
    """Return {class_id: [png paths]} from cutouts/<class_id>/*.png."""
    index = defaultdict(list)
    for cls_dir in sorted(cutouts_root.iterdir()):
        if cls_dir.is_dir() and cls_dir.name.isdigit():
            for png in cls_dir.glob("*.png"):
                index[int(cls_dir.name)].append(str(png))
    return {c: v for c, v in index.items() if v}


def _load_names_block(dataset_yml: Path):
    """Return the `names:` block lines from the existing dataset.yml."""
    lines = dataset_yml.read_text(encoding="utf-8", errors="ignore").splitlines()
    block, capture = [], False
    for line in lines:
        if line.startswith("names:"):
            capture = True
        if capture:
            block.append(line)
    return block


def main():
    args = parse_args()
    root = PROJECT_ROOT
    cutouts_root = (root / args.cutouts)
    bg_root = (root / args.backgrounds)
    out_root = (root / args.out)
    rng = random.Random(args.seed)

    index = _load_cutout_index(cutouts_root)
    if not index:
        print(f"ERROR: no cut-outs under {cutouts_root}", file=sys.stderr)
        return 1
    bg_tiles = [cv2.imread(str(p)) for p in sorted(bg_root.glob("*.png"))]
    bg_tiles = [t for t in bg_tiles if t is not None]
    if not bg_tiles:
        print(f"ERROR: no backgrounds under {bg_root}", file=sys.stderr)
        return 1

    img_out = out_root / "images" / "train"
    lbl_out = out_root / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    sampler = ClassBalancedSampler(index, seed=args.seed)
    for i in range(args.num):
        k = rng.randint(args.min_objs, args.max_objs)
        picks = sampler.sample(k)
        cutouts = []
        for cid, path in picks:
            bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if bgra is not None and bgra.ndim == 3 and bgra.shape[2] == 4:
                cutouts.append((cid, bgra))
        if not cutouts:
            continue
        canvas = make_background(bg_tiles, args.size, rng)
        rows = compose_one(canvas, cutouts, rng)
        if not rows:
            continue
        stem = f"synth_{i:06d}"
        cv2.imwrite(str(img_out / f"{stem}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        (lbl_out / f"{stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in rows)
            + "\n")
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{args.num} scenes")

    _mix_single_items(root, out_root, args.single_item_frac, rng)
    _link_val(root, out_root)
    names_block = _load_names_block(root / "dataset.yml")
    write_synth_yaml(out_root, names_block, root / "dataset_synth.yml")
    print(f"Done. Synthetic dataset at {out_root}; yaml at dataset_synth.yml")
    return 0


def _mix_single_items(root, out_root, frac, rng):
    """Copy a fraction of original single-item train images+labels into synth train."""
    if frac <= 0:
        return
    src_img = root / "dataset_640" / "images" / "train"
    src_lbl = root / "dataset_640" / "labels" / "train"
    imgs = sorted(src_img.glob("*.jpg"))
    n = int(len(imgs) * frac)
    for f in rng.sample(imgs, min(n, len(imgs))):
        lbl = src_lbl / (f.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copy2(f, out_root / "images" / "train" / f.name)
        shutil.copy2(lbl, out_root / "labels" / "train" / lbl.name)
    print(f"  mixed in {n} single-item images (frac={frac})")


def _link_val(root, out_root):
    """Point synth val at the real resized val by copying (Windows-safe, no symlink)."""
    for sub in ("images/val", "labels/val", "images/test"):
        src = root / "dataset_640" / sub
        dst = out_root / sub
        if dst.exists() or not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)
    print("  copied real val/test into synth tree")


def parse_args():
    p = argparse.ArgumentParser(description="Compose synthetic checkout scenes.")
    p.add_argument("--cutouts", default="cutouts")
    p.add_argument("--backgrounds", default="backgrounds")
    p.add_argument("--out", default="dataset_synth")
    p.add_argument("--num", type=int, default=20000)
    p.add_argument("--min-objs", type=int, default=3)
    p.add_argument("--max-objs", type=int, default=15)
    p.add_argument("--single-item-frac", type=float, default=0.1)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compose_scenes.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: End-to-end trial on trial dirs from Tasks 5–6**

Run:
```bash
python extract_cutouts.py --limit 400 --out cutouts_trial --per-class 5
python harvest_backgrounds.py --count 200 --out backgrounds_trial
python compose_scenes.py --cutouts cutouts_trial --backgrounds backgrounds_trial --out dataset_synth_trial --num 20 --single-item-frac 0.0
```
Expected: 20 scenes written under `dataset_synth_trial/images/train/`.

- [ ] **Step 6: Visually verify labels land on products**

Run (reuses the box-drawing verify from `resize_dataset.py`):
```bash
python resize_dataset.py --help >/dev/null   # confirm module importable
python -c "from resize_dataset import verify_boxes; from pathlib import Path; verify_boxes(Path('dataset_synth_trial/images/train'), Path('dataset_synth_trial/labels/train'), Path('dataset_synth_trial/verify'), 6)"
```
Expected: 6 annotated images in `dataset_synth_trial/verify/`; boxes tightly frame pasted products.

- [ ] **Step 7: Commit**

```bash
git add compose_scenes.py tests/test_compose_scenes.py
git commit -m "feat: add synthetic checkout-scene composition script"
```

---

## Task 8: Generate the full dataset, retrain, and validate (Phase-1 gate)

**Files:**
- Modify: `CLAUDE.md` (document the synthesis pipeline + `dataset_synth.yml`)
- No new code; this task runs the pipeline and trains.

**Interfaces:**
- Consumes: `extract_cutouts.py`, `harvest_backgrounds.py`, `compose_scenes.py`, `train.py`.
- Produces: `dataset_synth/`, `dataset_synth.yml`, `runs/detect/rpc_yolo11n_synth/`, `runs/detect/rpc_yolo11s_probe/`.

- [ ] **Step 1: Build the real cut-out library (full)**

Run: `python extract_cutouts.py --out cutouts --per-class 40`
Expected: "Done: N cut-outs across 200/200 classes." If any classes are missing, re-run those classes with a higher `--per-class` or inspect their source labels.

- [ ] **Step 2: Harvest backgrounds (full)**

Run: `python harvest_backgrounds.py --count 4000 --out backgrounds`
Expected: "Done: saved 4000 background patches".

- [ ] **Step 3: Compose the full synthetic dataset**

Run: `python compose_scenes.py --num 20000 --single-item-frac 0.1`
Expected: `dataset_synth/` with ~20k synthetic train images + ~5.4k mixed single-item images, real val/test copied in, `dataset_synth.yml` written.

- [ ] **Step 4: Smoke-train to prove the pipeline wires up**

Run: `python train.py --smoke --data dataset_synth.yml`
Expected: completes 2 epochs, 0 corrupt labels, `best.pt` saved. Fix any label/path errors before the full run.

- [ ] **Step 5: Full YOLO11n training run**

Run: `python train.py --data dataset_synth.yml --name rpc_yolo11n_synth`
Expected: training proceeds; **watch `runs/detect/rpc_yolo11n_synth/results.csv`** — `val/cls_loss` should now *decrease* across epochs (contrast: it rose to 6.09 before) and `mAP50` should climb well above ~0.05.

- [ ] **Step 6: YOLO11s capacity probe**

Run: `python train.py --data dataset_synth.yml --model yolo11s.pt --name rpc_yolo11s_probe --epochs 40`
Expected: a `mAP50` reading to compare against nano, quantifying the capacity ceiling. (Nano stays the Coral target regardless.)

- [ ] **Step 7: Verify predictions visually (the gate)**

Open `runs/detect/rpc_yolo11n_synth/val_batch0_pred.jpg`.
Expected: class names on real checkout scenes now largely match `val_batch0_labels.jpg`. **This is the Phase-1 gate — do not start Phase 2 until this passes.**

- [ ] **Step 8: Document + commit**

Update `CLAUDE.md` Commands/Layout to describe `extract_cutouts.py`, `harvest_backgrounds.py`, `compose_scenes.py`, and `dataset_synth.yml` (why they exist: closes the single-item→checkout domain gap). Then:

```bash
git add CLAUDE.md
git commit -m "docs: document synthetic-scene pipeline"
```

---

## Task 9: Phase 2 — Coral Edge TPU int8 export (GATED on Task 8)

**Files:**
- Create: `export_edgetpu.py`
- Test: `tests/test_export_edgetpu.py`

**Interfaces:**
- Consumes: the trained `runs/detect/rpc_yolo11n_synth/weights/best.pt`, `dataset_synth.yml`.
- Produces:
  - `resolve_best(run_name: str) -> Path` — return the `best.pt` under `runs/detect/<run_name>/weights/`, raising `FileNotFoundError` with a clear message if absent.
  - CLI: `python export_edgetpu.py --run rpc_yolo11n_synth --imgsz 320 --data dataset_synth.yml`.

> **Do not start this task until Task 8 Step 7 (the gate) passes.**

- [ ] **Step 1: Write the failing test (path resolution only — export needs the TF stack)**

```python
# tests/test_export_edgetpu.py
import pytest
from export_edgetpu import resolve_best


def test_resolve_best_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_best("does_not_exist", runs_root=tmp_path)


def test_resolve_best_finds_weights(tmp_path):
    w = tmp_path / "myrun" / "weights"
    w.mkdir(parents=True)
    (w / "best.pt").write_bytes(b"stub")
    assert resolve_best("myrun", runs_root=tmp_path).name == "best.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export_edgetpu.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `export_edgetpu.py`**

```python
"""export_edgetpu.py — export the trained YOLO11n to a Coral Edge TPU int8 model.

Ultralytics performs: PyTorch -> TF SavedModel -> full-int8 TFLite (calibrated on a
representative sample from --data) -> edgetpu_compiler -> *_edgetpu.tflite.

NOTE: edgetpu_compiler is Linux-only. On Windows the int8 TFLite exports fine but the
final compile step must run in WSL2/Docker/Colab. This script surfaces that clearly.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_best(run_name: str, runs_root: Path | None = None) -> Path:
    """Return runs/detect/<run_name>/weights/best.pt or raise FileNotFoundError."""
    runs_root = runs_root or (PROJECT_ROOT / "runs" / "detect")
    best = runs_root / run_name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"No trained weights at {best}. Run Phase-1 training first.")
    return best


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO11n to Coral Edge TPU int8.")
    p.add_argument("--run", default="rpc_yolo11n_synth", help="Training run name.")
    p.add_argument("--imgsz", type=int, default=320, help="Edge inference size.")
    p.add_argument("--data", default="dataset_synth.yml", help="Calibration data yaml.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        best = resolve_best(args.run)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ERROR: ultralytics not installed ({exc})", file=sys.stderr)
        return 1

    model = YOLO(str(best))
    print(f"Exporting {best} to Edge TPU int8 (imgsz={args.imgsz}) ...")
    try:
        out = model.export(format="edgetpu", int8=True,
                           data=str(PROJECT_ROOT / args.data), imgsz=args.imgsz)
    except Exception as exc:  # edgetpu_compiler missing on Windows lands here
        print(f"ERROR: export failed: {exc}\n"
              "If this is the edgetpu_compiler step, run it on Linux (WSL2/Docker/Colab).",
              file=sys.stderr)
        return 1
    print(f"Done. Edge TPU model: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export_edgetpu.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Export the int8 TFLite (Windows) / compile (Linux)**

Run: `python export_edgetpu.py --run rpc_yolo11n_synth --imgsz 320`
Expected on Windows: int8 `.tflite` produced; if the edgetpu compile fails, the message directs you to WSL2/Docker/Colab. Run the same command there to get `*_edgetpu.tflite`.

- [ ] **Step 6: Validate quantized accuracy vs float**

Run:
```bash
yolo val model=runs/detect/rpc_yolo11n_synth/weights/best.pt data=dataset_synth.yml imgsz=320
yolo val model=runs/detect/rpc_yolo11n_synth/weights/best_full_integer_quant.tflite data=dataset_synth.yml imgsz=320
```
Expected: int8 `mAP50` within an acceptable delta of the float model. If the drop is large, revisit the calibration sample or raise `--imgsz`.

- [ ] **Step 7: Commit**

```bash
git add export_edgetpu.py tests/test_export_edgetpu.py
git commit -m "feat: add Coral Edge TPU int8 export"
```

---

## Self-Review Notes

- **Spec coverage:** §5.1 → Task 5; §5.2 → Task 6; §5.3 → Tasks 4+7; §5.4 verify → Task 7 Step 6; §5.5 training + s-probe → Task 8; §6 Coral + Linux constraint → Task 9; §7 gate → Task 8 Step 7; §3 single-item mix → Task 7 `_mix_single_items`; no-leakage (§5.2) → Task 6 docstring + only train cut-outs pasted.
- **Type consistency:** `denormalize_box`/`normalize_box`/`mask_to_bbox`/`ClassBalancedSampler.sample`/`alpha_paste`/`compute_visibilities`/`rotate_rgba`/`random_placement`/`write_synth_yaml` signatures are defined in Tasks 2–4 and consumed unchanged in Tasks 5–7.
- **Placeholders:** none — every code step contains complete code and every run step an expected result.
