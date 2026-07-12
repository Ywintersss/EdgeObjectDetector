# EdgeObjectDetector — Laptop Demo Bundle

Self-contained. No dataset and no repo checkout needed.

## Setup

    pip install ultralytics opencv-python ai-edge-litert

`ai-edge-litert` is the runtime that actually executes the `.tflite` model. Ultralytics
would otherwise try to pip-install it on the first frame, which fails offline.

No CUDA required — the INT8 model runs on CPU, which is the point: it approximates
what the Coral Edge TPU will execute.

## Run

    python webcam_demo.py --camera 0

The model is found automatically: exactly one `.tflite` is bundled into this folder by
`export_int8.py --bundle SIZE`. Pass `--model path/to.tflite` only to override it.

## POINT THE CAMERA DOWN

This model was trained on RPC checkout scenes, which are **top-down overhead shots of
products lying on a flat, plain surface**.

Held at eye level, pointed at a product in your hand, it is out-of-domain and will
underperform. That is not a bug — it is the training distribution.

Prop the camera **above a table, looking down**, and place products on a plain surface.
