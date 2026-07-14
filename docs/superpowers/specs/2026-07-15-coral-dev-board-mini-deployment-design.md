# Coral Dev Board Mini Deployment — Design

**Date:** 2026-07-15
**Phase:** 2 of 2. Phase 1 (INT8 export + laptop demo) is complete and merged (`d284d4c`).

## Problem

Phase 1 produced `export/rpc_coarse17_int8_320_edgetpu.tflite` — a compiled, fully-integer
coarse-17 detector with **96.2% of ops mapped to the Edge TPU** (457 TPU / 18 CPU). It has never
run on the target hardware. Every number we have is from a desktop CPU.

This phase runs it on the **Coral Dev Board Mini** with a live overhead USB webcam, and measures
what the hardware actually does.

### The real work is not the TPU

On the laptop, `ultralytics.YOLO(...).predict()` silently did the letterboxing, input quantization,
box decoding, and NMS. **Ultralytics cannot come to the board** — it pulls in torch, and the target
is a quad-core Cortex-A35 running Mendel Linux.

So the raw `[1, 21, 2100]` int8 output tensor becomes ours to interpret. That hand-written decode —
not the TPU — is the part most likely to fail, and it fails *silently*: a transposed axis or a
missing dequantize produces plausible-looking boxes, not an exception. Phase 1 was nearly lost to
exactly this class of bug (INT8 score collapse looked perfectly healthy by every surface check).

## Goals

- Run the compiled model on the Edge TPU, from a live overhead USB webcam.
- Reimplement YOLO post-processing without Ultralytics, and **prove it correct on the desktop
  before the board is ever involved**.
- Measure per-stage timings that identify the true bottleneck.
- Distribute via `git clone` onto the board.

## Non-Goals (YAGNI)

- **Hitting a specific FPS target.** We have no requirement. We measure; we do not promise.
- Re-exporting, retraining, or trying other input sizes. The 320 model is fixed.
- Any production UI, service, autostart, or enclosure.
- MIPI-CSI camera support. USB webcam only.

## Decisions

### D1 — Distribution: one repo, new `coral/` directory

Push `EdgeObjectDetector` to GitHub as-is (all multi-GB data is already gitignored). The board does
`git clone --depth 1` and runs out of `coral/`.

**A separate deploy repo is rejected:** two sources of truth drift the first time the model is
re-exported.

**`coral/` is separate from `deploy/`, not an extension of it.** They are different programs:
`deploy/` is Ultralytics + the CPU-runnable model and cannot run on the board at all. Concretely,
dropping the Edge TPU model into `deploy/` would *hard-fail* — `webcam_demo.py:35`
(`resolve_model_path`) raises when it finds more than one `.tflite`, by design. `deploy/` is left
untouched.

### D2 — Commit the deployment artifacts

`.gitignore` currently ignores `*.tflite` (line 24) and `deploy/classes.txt` (line 52). A board
cloning the repo today would arrive **with no model and no class names**.

Those rules are correct for training weights and rebuildable exports. A *deployment artifact* is
different in kind: it is the release, and the whole point is that the board gets it in one clone. At
3.9 MB it is far inside GitHub's limits — no LFS.

Narrow negations, not a loosened rule:

```
!coral/*_edgetpu.tflite
!coral/classes.txt
```

### D3 — Port the decode, and validate it against a desktop oracle

Rejected alternatives:

- **Write the decode on the board and debug it there.** You would be debugging numpy indexing over
  SSH, on a slow board, with no reference to compare against, while simultaneously fighting
  first-time OTG/webcam/Wi-Fi setup. Every bug would present as a TPU bug.
- **Use PyCoral's `detect.get_objects()` adapter.** It expects the SSD/MobileNet output signature
  (four separate tensors: boxes, classes, scores, count). Our YOLO head emits one fused
  `[1, 21, 2100]` tensor. Bending the adapter to fit is more work than ~40 lines of numpy. We still
  use PyCoral to load the model and bind the TPU delegate — just not its detection adapter.

**Chosen:** write `decode.py` as pure numpy, then assert on the desktop that it reproduces
Ultralytics' output on the CPU twin (`deploy/rpc_coarse17_int8_320.tflite`) — the *same model*,
already verified end-to-end. Only then does anything go to the board, where the sole new variable is
the TPU delegate.

The channel layout (`[21, 2100]` vs its transpose; box channels leading or trailing) is **pinned
down by that test, not asserted from memory.**

### D4 — Display is a sink, not a mode

The loop produces one annotated frame and hands it to a list of sinks. `--display stream|hdmi|both`.

- **MJPEG (must work):** threaded HTTP server, `multipart/x-mixed-replace`, viewed at
  `http://<board-ip>:8080`. It is a socket; it is boring; it works headless and while the camera is
  mounted overhead.
- **HDMI (best-effort):** Mendel runs a Wayland compositor and OpenCV's `imshow` is built against
  GTK/X11, so the window may not open. If it doesn't, warn once and continue — never kill the run.

## Architecture

```
coral/
  rpc_coarse17_int8_320_edgetpu.tflite   committed, 3.9 MB
  classes.txt                            committed, 17 names
  decode.py         pure numpy: int8 tensor -> boxes. No TPU, no camera. Runs anywhere.
  interpreter.py    load a .tflite; bind the Edge TPU delegate, or don't. One flag.
  detect.py         board runner: camera -> preprocess -> invoke -> decode -> sinks
  sinks.py          MJPEG HTTP sink; best-effort HDMI sink
  probe_board.py    step 0 on the board, before anything else
  README.md
tests/
  test_coral_decode.py
```

`interpreter.py` is the **only** module that differs between desktop and board — desktop binds no
delegate and loads the CPU twin; the board binds the Edge TPU delegate and loads the compiled model.
Isolating it in one function with one flag is what makes "only one variable changed" literally true
at the hardware crossing, rather than merely aspirational.

### Data flow

```
webcam frame (BGR)
  -> letterbox to 320x320, BGR->RGB
  -> quantize with the INPUT tensor's own scale/zero_point (read at runtime, never hardcoded)
  -> interpreter.invoke()
  -> [1, 21, 2100] int8
  -> dequantize with the OUTPUT tensor's own scale/zero_point
  -> split 4 box channels / 17 score channels
  -> conf = max score per anchor; cls = argmax; threshold
  -> xywh -> xyxy; undo letterbox padding back to frame coords
  -> class-wise NMS
  -> draw -> sinks
```

Boxes arrive normalized 0..1 because the export went through `format="edgetpu"`, which is the only
format for which Ultralytics normalizes them (this is the Phase 1 fix; see `export_int8.py`).

## Verification

### Desktop (before the board)

`tests/test_coral_decode.py`:

- **The oracle test.** Run a real `real_eval` image through both paths — Ultralytics on the CPU twin,
  and our letterbox + interpreter + decode on the same file. Assert the same detection count, the
  same classes, and boxes agreeing within tolerance.
  Tolerance, not equality: Ultralytics' letterbox and NMS tie-breaks are not bit-identical to ours.
  The test exists to catch **structural** errors — transposed axis, missing dequantize, class offset
  by four — which are the ones that actually occur.
- Unit tests for NMS (overlapping boxes suppressed, distinct boxes kept), the xywh->xyxy conversion,
  and letterbox coordinate round-tripping.

### Board

`probe_board.py`, run first over SSH, proves the ground before anything is built on it: the Edge TPU
delegate loads, OpenCV imports, `/dev/video0` exists and yields one real frame. Ten seconds, and it
tells you which of those is broken — instead of finding out later from a confusing failure inside the
detect loop.

## Measurement

**A single FPS number would mislead.** At least four things can independently cap it, and they call
for opposite responses:

1. **The TPU is not doing all the work.** 18 ops (the detect-head tail) fall back to the Cortex-A35.
2. **The numpy decode** chews 2100x21 values on that same modest CPU.
3. **JPEG-encoding** each frame for the MJPEG stream costs more A35 time.
4. **The USB webcam** on a bandwidth-limited OTG port may not deliver frames faster than 15-30 fps
   regardless of model speed.

"22 FPS" is consistent with all four and distinguishes none of them.

So the loop times **five stages separately** — capture, preprocess, invoke, decode, sink — reporting
median and p90 for each alongside end-to-end FPS. Separately, a warm micro-benchmark runs `invoke()`
alone on one fixed frame, **discarding the first call** (delegate warm-up), to isolate the figure
comparable to Coral's published latencies.

Results land in `coral/results.md`.

## Success Criteria

1. `probe_board.py` passes on the board.
2. The desktop oracle test is green — our decode matches Ultralytics on the CPU twin.
3. The board runs the live overhead camera and draws correct boxes on real products.
4. A per-stage timing breakdown is recorded, with TPU invoke latency isolated.

Explicitly **not** a criterion: any particular FPS figure.

## Risks

| Risk | Mitigation |
|---|---|
| **Locking yourself out of the board.** The webcam takes the OTG port, which is how `mdt` connects. | **Set up Wi-Fi and verify SSH over it while `mdt` still works.** Note the IP. *Then* plug in the camera. This ordering is not optional. |
| OTG port may not source enough current for the webcam | Powered OTG hub. `probe_board.py` surfaces this immediately (no `/dev/video0`, or failing reads). |
| OpenCV absent on Mendel | `sudo apt-get install python3-opencv`. The probe reports it in ten seconds. |
| `cv2.imshow` fails under Wayland | Exactly why MJPEG is the guaranteed sink and HDMI is best-effort. |
| PyCoral / `tflite_runtime` version friction on Mendel's Python | The probe pins down what is actually installed before we write against it. |
| Decode is subtly wrong | The desktop oracle test. This is the central risk and the central mitigation. |
