"""Load a .tflite model, with or without the Edge TPU delegate.

This is the ONLY module that differs between the desktop and the Coral. Everything else
-- preprocessing, decode, NMS, drawing -- is byte-identical on both. That is deliberate:
it means the desktop oracle test exercises the same code the board runs, and the hardware
crossing changes exactly one variable. If the board misbehaves, it is here or it is the
hardware; it is not the decode.

Runtime preference order: tflite_runtime (what PyCoral installs on Mendel) -> ai_edge_litert
(the desktop runtime) -> tensorflow.lite. Whichever is present wins.
"""

from __future__ import annotations

import platform
from pathlib import Path

# libedgetpu ships under a different name on each OS. Only the Linux one matters for the
# board; the others exist so a wrong-platform attempt fails with a clear message.
EDGETPU_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}

# edgetpu_compiler always embeds a custom op with this name into the compiled flatbuffer.
# Its presence/absence is a reliable, hardware-free way to tell a compiled *_edgetpu.tflite
# apart from an ordinary .tflite -- which is what make_interpreter uses it for below.
EDGETPU_CUSTOM_OP = b"edgetpu-custom-op"


def is_edgetpu_compiled(model_path):
    """-> bool: whether model_path was run through edgetpu_compiler.

    Pure byte check, no runtime and no hardware involved -- that is what makes it
    testable on the desktop.
    """
    return EDGETPU_CUSTOM_OP in Path(model_path).read_bytes()


def _load_runtime():
    """-> (Interpreter, load_delegate) from whichever TFLite runtime is installed."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        pass
    try:
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError as exc:
        raise RuntimeError(
            "no TFLite runtime found. On the Coral board: "
            "sudo apt-get install python3-tflite-runtime python3-pycoral. "
            "On a desktop: pip install ai-edge-litert") from exc


def make_interpreter(model_path, use_tpu):
    """Return an ALLOCATED interpreter for model_path.

    use_tpu=True binds the Edge TPU delegate and requires the *_edgetpu.tflite model.
    use_tpu=False runs the plain CPU model -- which is how the desktop oracle test drives
    the very same code path the board uses.
    """
    path = Path(model_path)
    if not path.is_file():
        raise RuntimeError(f"model not found: {path}")

    Interpreter, load_delegate = _load_runtime()

    if not use_tpu:
        it = Interpreter(model_path=str(path))
        it.allocate_tensors()
        return it

    # A missing delegate library or absent device is NOT the only way this goes wrong.
    # load_delegate() below happily binds to a plain .tflite that was never run through
    # edgetpu_compiler -- allocate_tensors() and invoke() then both succeed, with every
    # op silently executing on the CPU. The caller would walk away believing they
    # measured the Edge TPU. Catch that before the delegate ever binds.
    if not is_edgetpu_compiled(path):
        raise RuntimeError(
            f"{path.name} was never compiled for the Edge TPU -- it has no "
            f"'{EDGETPU_CUSTOM_OP.decode()}' op, so the Edge TPU delegate would still bind "
            f"and every op would silently run on the CPU while this call claims to report "
            f"a TPU run. Run `edgetpu_compiler` on the plain .tflite and pass the resulting "
            f"*_edgetpu.tflite here instead.")

    lib = EDGETPU_LIB.get(platform.system())
    if lib is None:
        raise RuntimeError(f"no Edge TPU delegate is available for {platform.system()}")
    try:
        delegate = load_delegate(lib)
    except (ValueError, OSError) as exc:
        # The single most common board failure. Do NOT fall back to CPU silently -- that
        # would report a plausible FPS number for a run that never touched the TPU, which
        # is the whole thing we are here to measure.
        raise RuntimeError(
            f"could not load the Edge TPU delegate ({lib}): {exc}. Is the board a Coral, "
            f"is libedgetpu installed, and is the device free? Refusing to fall back to "
            f"CPU -- that would silently invalidate the benchmark.") from exc

    it = Interpreter(model_path=str(path), experimental_delegates=[delegate])
    it.allocate_tensors()
    return it
