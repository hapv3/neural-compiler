"""TFLite interpreter with Neural-AI INT8 MAC/requant kernels; other ops use TFLite."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MAC_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}


def make_interpreter(model, kind="neural_ai_reference", preserve=False, threads=1):
    if kind in ("neural_ai_afu_mean_reference", "tflite_afu_mean_reference"):
        from afu_mean import AFUMeanInterpreter
        base = "neural_ai_reference" if kind.startswith("neural_ai_") else "tflite_reference"
        return AFUMeanInterpreter(model, base, preserve, threads, make_interpreter)
    if kind in ("neural_ai_afu_dfl_reference", "tflite_afu_dfl_reference"):
        from afu_dfl import AFUDFLInterpreter
        base = "neural_ai_reference" if kind.startswith("neural_ai_") else "tflite_reference"
        return AFUDFLInterpreter(model, base, preserve, threads, make_interpreter)
    options = {"num_threads": threads, "experimental_preserve_all_tensors": preserve}
    options["model_content" if isinstance(model, bytes) else "model_path"] = model if isinstance(model, bytes) else str(model)
    if kind in ("neural_ai_reference", "tflite_reference"):
        options["experimental_op_resolver_type"] = tf.lite.experimental.OpResolverType.BUILTIN_REF
    elif kind != "tflite_optimized":
        raise ValueError(f"Unknown runtime: {kind}")
    if kind == "neural_ai_reference":
        manifest = json.loads((ROOT / "build/manifest.json").read_text())
        library = ROOT / "build/libneural_ai_reference.so"
        if manifest["tensorflow_version"] != tf.__version__:
            raise RuntimeError("TensorFlow version changed; rebuild the delegate")
        if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["library_sha256"]:
            raise RuntimeError("Delegate library does not match build manifest")
        if hashlib.sha256((ROOT / "delegate.cpp").read_bytes()).hexdigest() != manifest["source_sha256"]:
            raise RuntimeError("Delegate source changed; rebuild before running")
        options["experimental_delegates"] = [tf.lite.experimental.load_delegate(str(library))]
    interpreter = tf.lite.Interpreter(**options)
    interpreter.allocate_tensors()
    return interpreter


def coverage(interpreter):
    """Use actual delegated outputs, not assumptions about supported op names."""
    if hasattr(interpreter, "reference_coverage"):
        return interpreter.reference_coverage()
    ops = interpreter._get_ops_details()
    delegated = {int(t) for op in ops if op["op_name"] == "DELEGATE" for t in op["outputs"]}
    nodes = [{"index": int(op["index"]), "operator": op["op_name"],
              "outputs": [int(t) for t in op["outputs"]],
              "neural_ai_kernel": all(int(t) in delegated for t in op["outputs"])}
             for op in ops if op["op_name"] != "DELEGATE"]
    return {"nodes": nodes,
            "neural_ai": dict(Counter(op["operator"] for op in nodes if op["neural_ai_kernel"])),
            "tflite_fallback": dict(Counter(op["operator"] for op in nodes if not op["neural_ai_kernel"])),
            "mac_fallback_nodes": [op["index"] for op in nodes
                                   if op["operator"] in MAC_OPS and not op["neural_ai_kernel"]]}
