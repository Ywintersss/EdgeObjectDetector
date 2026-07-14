# Coral Dev Board Mini — Deployment Runbook

Everything you need to get `rpc_coarse17_int8_320_edgetpu.tflite` running on the Edge TPU
from a live overhead USB webcam, and to come away with real numbers.

Work through it in order. **Part 2 is the one that will bite you if you skip ahead.**

---

## Part 0 — What you are pushing, and why the whole repo

Push the **entire repository**, not just `coral/`.

| | |
|---|---|
| Tracked files | 50 |
| Total size | **4.2 MB** (3.8 MB of that is the compiled model) |
| Not pushed | `dataset/`, `dataset_640/`, `dataset_real/`, `runs/`, all `*.pt`, all scratch `*.tflite` — every multi-GB tree is gitignored |
| Secrets | none tracked (checked) |

A separate deploy-only repo would mean two sources of truth, and they drift the first time you
re-export the model. The board just does a shallow clone and runs out of the `coral/`
subdirectory.

The `coral/` bundle is **complete in git** — 8 files, including the model and `classes.txt`:

```
coral/README.md          coral/decode.py        coral/probe_board.py
coral/classes.txt        coral/detect.py        coral/rpc_coarse17_int8_320_edgetpu.tflite
coral/interpreter.py     coral/sinks.py
```

(This needed narrow `.gitignore` negations — `*.tflite` was ignored repo-wide, so without them
the board would have cloned the code and **no model and no class names**.)

---

## Part 1 — Create the GitHub repo and push (on Windows)

You have no git remote and `gh` is not installed, so do this in the browser.

**1.1** Go to <https://github.com/new>. Create a repo — **private is fine** and is what I'd pick
unless you want this public. **Do not** let GitHub add a README, .gitignore, or licence; the repo
must start empty or the push will conflict.

**1.2** Back in `D:\Projects\EdgeObjectDetector`, wire it up and push:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

**1.3** Confirm the model actually made it. On the GitHub page, open `coral/` — you must see
`rpc_coarse17_int8_320_edgetpu.tflite` listed at ~3.8 MB. **If it is missing, stop.** The board
will clone an empty bundle and `probe_board.py` will fail at the "model present" check.

> **If the repo is private**, the board needs credentials to clone. Simplest: make it public for
> the duration of the exercise. Otherwise generate a GitHub personal access token (repo: read) and
> clone with `https://<token>@github.com/...`. Do not paste a token into any file in the repo.

---

## Part 2 — ⚠️ Wi-Fi FIRST. Do not plug the camera in yet.

**Read this before touching hardware.**

The Dev Board Mini has **one** USB-C data port (OTG). That is the same port `mdt` talks over.
The moment your webcam occupies it, **`mdt shell` is gone**. If you have not already proven you
can reach the board another way, you have locked yourself out and will need a serial cable to get
back in.

So: bring up Wi-Fi and **verify SSH over it** while USB still works.

**2.1** Get a shell the way you already know works:

```bash
mdt shell
```

**2.2** Join your Wi-Fi. Mendel runs NetworkManager, so either:

```bash
nmtui                                              # menu-driven; pick "Activate a connection"
# ...or non-interactively:
nmcli device wifi connect "<YOUR_SSID>" password "<YOUR_PASSWORD>"
```

**2.3** Find the board's IP and write it down:

```bash
ip addr show wlan0 | grep "inet "
```

You want the `192.168.x.x` (or similar) address. Call it `<board-ip>` from here on.

**2.4** **Prove SSH works — from your laptop, in a separate terminal, while USB is still connected:**

```bash
ssh mendel@<board-ip>
```

Default password is `mendel` unless you changed it.

**Do not continue until this succeeds.** This is the whole point of Part 2. If SSH over Wi-Fi is
not working *now*, it will not start working after you unplug the cable.

**2.5** While still on USB, install what the board needs (see Part 3), because a working `mdt`
shell is a nice safety net for the install step.

---

## Part 3 — Install dependencies on the board

In the board shell (`mdt shell` or SSH):

```bash
sudo apt-get update
sudo apt-get install -y git python3-opencv
```

**PyCoral / libedgetpu:** you have already run a Coral example successfully, so the Edge TPU
runtime is almost certainly present. Confirm it rather than assume:

```bash
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```

Expected: a non-empty list, e.g. `[{'type': 'usb', 'path': '/sys/bus/...'}]`. If it errors or
returns `[]`:

```bash
sudo apt-get install -y python3-pycoral python3-tflite-runtime
```

Check the Python version too — our code targets **3.7+** and the board is Debian 10:

```bash
python3 --version
```

---

## Part 4 — Clone the repo onto the board

Still in the board shell:

```bash
cd ~
git clone --depth 1 https://github.com/<your-username>/<your-repo>.git EdgeObjectDetector
cd EdgeObjectDetector/coral
ls -la
```

You must see all 8 files, **including `rpc_coarse17_int8_320_edgetpu.tflite` at ~3.8 MB**. If the
model is absent, the push in Part 1 didn't include it — go back to step 1.3.

---

## Part 5 — Now plug the camera in

Only now. Wi-Fi is confirmed, SSH is confirmed, the code is on the board.

**5.1** Disconnect the USB-C cable from your laptop. (`mdt` is now unavailable — that is expected.
You reach the board via `ssh mendel@<board-ip>`.)

**5.2** Connect the webcam through a **USB-C OTG adapter** into the board's data port.

**5.3** Keep the board powered via its **separate USB-C power port**. Do not try to power the
board through the OTG port.

**5.4** SSH back in and confirm the camera enumerated:

```bash
ssh mendel@<board-ip>
ls /dev/video*
```

Expect `/dev/video0`. If nothing appears, or it appears but reads fail, the OTG port is very
likely not sourcing enough current for the camera — **use a powered OTG hub.** This is the single
most common hardware snag on this board.

---

## Part 6 — Probe before you run

```bash
cd ~/EdgeObjectDetector/coral
python3 probe_board.py
```

This is step 0 on purpose. Its only job is to tell you which foundation is missing, in ten
seconds, from output you cannot misread — instead of fifteen minutes later from a confusing
failure deep inside the detect loop. It checks, in order:

1. **Python version** — must be ≥ 3.7
2. **numpy**
3. **OpenCV**
4. **model present** — the `*_edgetpu.tflite` actually arrived in the clone
5. **classes.txt** — 17 names
6. **Edge TPU delegate** — binds `libedgetpu.so.1` and loads the model onto the TPU
7. **USB webcam** — opens `/dev/video0` and captures one real frame

You want six `ok` lines. Fix anything it reports before going further — see Troubleshooting below.

---

## Part 7 — Run it

**Mount the camera pointing DOWN** at products on a plain surface, roughly 40–60 cm above them.
This model was trained on top-down RPC checkout scenes; at eye level it is out-of-domain and will
underperform. That is the training distribution, not a bug.

```bash
python3 detect.py --display stream
```

Then on your laptop, open:

```
http://<board-ip>:8080
```

You should see the live feed with green boxes and labels (`drink 0.94`, `chocolate 0.92`, …).

Options:

```bash
python3 detect.py --display stream            # MJPEG only (default) — always works
python3 detect.py --display both              # ...plus an HDMI window, IF Mendel's Wayland allows one
python3 detect.py --display stream --conf 0.4 # raise the confidence threshold
```

`--display hdmi`/`both` is **best-effort**: OpenCV's `imshow` is built against GTK/X11 and Mendel
runs Wayland, so the window may refuse to open. If it does, the program says so once and carries
on streaming. Losing the window never loses the run.

Press **Ctrl-C** to stop. It prints the timing report on exit.

---

## Part 8 — Record the numbers (this is the deliverable)

On Ctrl-C you get a six-stage breakdown:

```
stage           median       p90      n
capture         xx.x ms   xx.x ms   xxx
preprocess      xx.x ms   xx.x ms   xxx
invoke          xx.x ms   xx.x ms   xxx
decode          xx.x ms   xx.x ms   xxx
draw            xx.x ms   xx.x ms   xxx
sink            xx.x ms   xx.x ms   xxx

end-to-end: NN.N FPS (xx.x ms per frame, summed medians)
```

**Read it like this.** A single FPS number would mislead you — four different things can cap it
independently, and they call for opposite responses:

- **`capture`** — the USB webcam on a bandwidth-limited OTG port. May simply not deliver frames
  faster than 15–30 fps no matter how fast the model is.
- **`invoke`** — the Edge TPU… **plus the 18 ops (of 475) that fall back onto the Cortex-A35.**
  We measured 96.2% TPU coverage; the detect-head tail runs on CPU.
- **`preprocess` / `decode` / `draw` / `sink`** — all Cortex-A35. The numpy decode alone chews
  2100×21 values per frame, and `sink` is JPEG-encoding every frame for the stream.

**Five of the six stages are CPU.** So the expected shape of the result is: `invoke` is small, the
CPU stages dominate — which means *the TPU is doing its job and the A35 is the wall*. If instead
`invoke` dominates, something is wrong (most likely the model is not actually running on the TPU).

For the cleanest TPU-only figure, run the still-image benchmark, which times `invoke()` alone with
the first call discarded (delegate warm-up):

```bash
# grab a frame first, or scp one over
python3 detect.py --image /path/to/a/frame.jpg
```

**Paste both outputs into `coral/results.md`, commit, and push.** The finding is **which stage
dominates** — not the FPS number.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `probe_board.py`: **model present → FAIL** | The `.tflite` never made it to GitHub | Re-check Part 1.3. `.gitignore` negations must be present. |
| `probe_board.py`: **Edge TPU delegate → FAIL** | `libedgetpu` missing, or the device is busy | `python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"` — if empty, reinstall `python3-pycoral`. If another process holds the TPU, kill it. |
| `probe_board.py`: **USB webcam → FAIL** | OTG port not sourcing enough current | **Powered OTG hub.** This is the most common failure. |
| `TypeError: type 'list' is not subscriptable` | You are on an old commit | Should be impossible on `main` — all board modules defer annotations for Python 3.7. `git pull`. |
| `RuntimeError: ... was never compiled for the Edge TPU` | You pointed `--model` at the plain CPU `.tflite` | Use the `*_edgetpu.tflite`. The runner refuses to run an uncompiled model on the TPU path rather than silently executing it all on CPU and reporting a fake TPU number. |
| Can't reach `http://<board-ip>:8080` | Wrong IP, or a firewall | `ip addr show wlan0` on the board. Make sure laptop and board are on the same network. |
| `OSError: Address already in use` | A previous `detect.py` still holds port 8080 | `pkill -f detect.py`, or `--port 8081`. |
| Locked out — `mdt` gone, SSH doesn't work | You plugged the camera in before confirming Wi-Fi | Unplug the camera, restore `mdt`, redo **Part 2**. This is exactly what Part 2 exists to prevent. |
| Boxes appear but labels look wrong | Stale `classes.txt` | It ships in the bundle; don't substitute one. The code raises a named error on a class-count mismatch. |
| Detections are poor / nothing found | Camera not pointing **down** | Overhead, plain surface, ~40–60 cm. Eye-level is out-of-domain. |

---

## What each file in `coral/` does

| File | Job |
|---|---|
| `probe_board.py` | Step 0. Proves the ground before anything is built on it. |
| `decode.py` | Pure numpy: letterbox, input quantization, box decode, class-wise NMS. No TPU, no camera — which is why it could be validated on the desktop against Ultralytics before the board existed. |
| `interpreter.py` | Loads the `.tflite` and binds the Edge TPU delegate. **The only module that differs between desktop and board** — that is what makes "one variable changed" true at the hardware crossing. Refuses to fall back to CPU silently. |
| `sinks.py` | MJPEG HTTP stream (must work) + best-effort HDMI window. |
| `detect.py` | The runner. Camera → preprocess → invoke → decode → draw → sinks, with all six stages timed. |
| `classes.txt` | The 17 coarse category names, in class-index order. |
| `rpc_coarse17_int8_320_edgetpu.tflite` | The compiled model. 96.2% of ops on the TPU. |

---

## Reference — the numbers we already have

Measured on the desktop, for context when you read the board's results:

| | mAP@50 | mAP@50-95 |
|---|---|---|
| FP32 baseline @ 640 | 0.995 | 0.879 |
| INT8 @ 320 (cluttered) | **0.978** | 0.758 |

Edge TPU op coverage: **457 on TPU / 18 on CPU = 96.2%**.

Desktop CPU latency at 320 was ~4 ms/frame — that is **not** a Coral forecast. The board has a
different processor entirely. That is what you are about to find out.
