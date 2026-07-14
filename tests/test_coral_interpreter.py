import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coral"))
import interpreter as I  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CPU_MODEL = PROJECT_ROOT / "deploy" / "rpc_coarse17_int8_320.tflite"
EDGETPU_MODEL = PROJECT_ROOT / "coral" / "rpc_coarse17_int8_320_edgetpu.tflite"

pytestmark = pytest.mark.skipif(
    not CPU_MODEL.exists(),
    reason=f"{CPU_MODEL} missing -- run: python export_int8.py --bundle 320")


def test_make_interpreter_loads_the_cpu_model_with_the_expected_signature():
    it = I.make_interpreter(CPU_MODEL, use_tpu=False)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    assert list(inp["shape"]) == [1, 320, 320, 3]
    assert list(out["shape"]) == [1, 21, 2100]
    # Fully-integer both ends -- the property the Edge TPU actually requires.
    assert inp["dtype"].__name__ == "int8"
    assert out["dtype"].__name__ == "int8"


def test_make_interpreter_allocates_so_invoke_works_immediately():
    import numpy as np
    it = I.make_interpreter(CPU_MODEL, use_tpu=False)
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], np.zeros((1, 320, 320, 3), dtype=np.int8))
    it.invoke()   # would raise if allocate_tensors() had not been called
    assert it.get_tensor(it.get_output_details()[0]["index"]).shape == (1, 21, 2100)


def test_missing_model_is_named_not_swallowed(tmp_path):
    with pytest.raises(RuntimeError, match="model not found"):
        I.make_interpreter(tmp_path / "nope.tflite", use_tpu=False)


def test_requesting_the_tpu_without_one_fails_loudly_and_says_why():
    # There is no Edge TPU on the desktop. The failure must NAME the delegate, so that
    # on the board a real delegate problem is distinguishable from a missing model.
    with pytest.raises(RuntimeError, match="Edge TPU delegate"):
        I.make_interpreter(CPU_MODEL, use_tpu=True)


@pytest.mark.skipif(
    not EDGETPU_MODEL.exists(),
    reason=f"{EDGETPU_MODEL} missing -- run: edgetpu_compiler on the plain .tflite")
def test_is_edgetpu_compiled_true_for_the_real_compiled_artifact():
    assert I.is_edgetpu_compiled(EDGETPU_MODEL) is True


def test_is_edgetpu_compiled_false_for_the_real_plain_artifact():
    assert I.is_edgetpu_compiled(CPU_MODEL) is False


def test_make_interpreter_refuses_an_uncompiled_model_on_the_tpu_path():
    # CPU_MODEL is a plain .tflite -- never run through edgetpu_compiler. Requesting the
    # TPU with it must be caught BEFORE the delegate binds, or every op silently runs on
    # the CPU while the caller believes they measured the Edge TPU. The failure must name
    # the real diagnosis (compilation), not merely "no delegate found" -- match on a
    # substring that only the compilation-check message contains, so this test cannot
    # pass vacuously via the pre-existing missing-delegate error.
    with pytest.raises(RuntimeError, match="compiled"):
        I.make_interpreter(CPU_MODEL, use_tpu=True)
