# EdgeObjectDetector — Coral Dev Board Mini

Runs the coarse-17 detector on the Edge TPU from a live overhead USB webcam.

**No Ultralytics, no torch.** Post-processing is hand-written numpy (`decode.py`), pinned
against Ultralytics on the desktop by `tests/test_coral_oracle.py`.

## ⚠️ Do this BEFORE you plug the camera in

The webcam takes the **USB-C OTG port** — which is the port `mdt` talks over. The moment
it is plugged in, `mdt shell` is gone.

**So: bring up Wi-Fi and confirm you can SSH in over it while `mdt` still works.**

```bash
mdt shell                       # while USB is still yours
nmtui                           # join Wi-Fi
ip addr show wlan0              # note the IP
```

Then, from your laptop, **prove SSH works over Wi-Fi** (`ssh mendel@<board-ip>`) before
unplugging USB. Skip this and you lock yourself out of the board.

## Setup (on the board)

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-pycoral
git clone --depth 1 <your-repo-url> EdgeObjectDetector
cd EdgeObjectDetector/coral
python3 probe_board.py
```

`probe_board.py` checks numpy, OpenCV, the model, `classes.txt`, the Edge TPU delegate,
and that the webcam yields a real frame. Fix whatever it reports before going further.

## Run

```bash
python3 detect.py --display stream          # then open http://<board-ip>:8080 on your laptop
python3 detect.py --display both            # ...plus an HDMI window, if Wayland allows one
```

**POINT THE CAMERA DOWN** at products on a plain surface. The model was trained on
top-down RPC checkout scenes; at eye level it is out-of-domain.

Ctrl-C prints the per-stage timing report.

## Reading the timings

Six stages are timed separately because a single FPS number cannot tell you which of
these is the bottleneck:

- **capture** — the USB webcam on a bandwidth-limited OTG port
- **preprocess** — letterbox + quantize, on the Cortex-A35
- **invoke** — the TPU… *plus* the 18 ops (of 475) that fall back onto the A35
- **decode** — 2100×21 of numpy, on the A35
- **draw** — annotating boxes onto the frame, on the A35
- **sink** — JPEG-encoding each frame for the stream, on the A35

Five of those six are CPU. If `invoke` is small and the rest dominate, the TPU is doing
its job and the A35 is the wall — which is the expected shape of the result.
