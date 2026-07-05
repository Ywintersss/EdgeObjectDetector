# EdgeObjectDetector

YOLO11 object detector for the **RPC (Retail Product Checkout)** dataset — **200 retail-product classes** (`1_puffed_food` … `200_stationery`). Targets edge deployment, so the default model is **YOLO11n** (nano).

## Layout

```
dataset_640.yml             # DEFAULT training config -> resized 640px images (fast)
dataset.yml                 # full-resolution config (200 classes) — slow; kept for reference
preprocessing.py            # COCO instances_*.json -> YOLO .txt labels (already run)
resize_dataset.py           # one-time offline downscale to 640px (already run)
train.py                    # Training entry point (--smoke for a fast pipeline test)
dataset_640/                # 640px-resized copy used for training (~10 GB)
  images/{train,val,test}/  # 53,739 / 6,000 / 24,000 images (max side 640)
  labels/{train,val}/       # YOLO labels copied unchanged (normalized -> resolution-agnostic)
dataset/                    # original full-res data (1751px); source for resizing
  images/{train,val,test}/  # 53,739 / 6,000 / 24,000 images
  labels/{train,val}/       # YOLO-format labels, 100% aligned to image basenames
instances_{train,val,test}2019.json   # original COCO annotations (source of truth for classes)
runs/detect/<name>/         # training outputs: weights/best.pt, curves, metrics
```

Note: `coco_converted/`, `coco_converted-2/`, `yolo_format/`, and `dataset_640_trial/` are
leftover/trial folders and can be deleted.

## Environment

- Python **3.13**, `ultralytics` **8.4.x**, `torch` **2.11.0+cu128** (**CUDA enabled**).
- GPU: **NVIDIA RTX 5060 (8 GB, Blackwell / sm_120)** — active and used automatically.
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
- To reinstall the CUDA build if the environment is ever reset:
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```

## Commands

```bash
python train.py --smoke        # fast end-to-end pipeline test (~1% data, 2 epochs)
python train.py                # full run on dataset_640: 100 epochs max, imgsz 512,
                               # batch 32, patience=20 early-stop, auto GPU
python train.py --data dataset.yml   # train on full-res originals instead (much slower)
```

Speed defaults are tuned for the RTX 5060 (8 GB): trains on the **640px-resized** dataset
(`dataset_640.yml`, the `--data` default), `imgsz=512` (~36% less compute than 640), `batch=32`,
and `patience=20` (early-stop when val mAP plateaus).

**Why the resized dataset exists:** source images are ~1751×1751, so a decoded cache would need
**~873 GB** (won't fit) and per-epoch JPEG decode dominated training. `resize_dataset.py` did a
one-time downscale to 640px; YOLO labels are normalized so they needed no changes. To rebuild it:
`python resize_dataset.py --verify` (or `--size N` for a different resolution).

Trained weights land in `runs/detect/<name>/weights/best.pt`.

## Conventions

- Run Python with `python` (not `python3`).
- Ultralytics locates labels by swapping `/images/` → `/labels/` in each image path — keep the
  `dataset/images/*` ↔ `dataset/labels/*` parallel structure intact.
- Class order in `dataset.yml` is generated from `instances_train2019.json` category order
  (YOLO index = enumerate position). If you regenerate labels, regenerate `dataset.yml` the same way.
