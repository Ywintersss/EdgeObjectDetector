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
