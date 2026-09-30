"""Make an explicit image-shape profile for a host-provided ONNX model.

This tool runs during model preparation. The Rust service does not use Python.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def specialize(source: Path, destination: Path, width: int, height: int):
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    if not all(64 <= d <= 1920 and d % 8 == 0 for d in (width, height)):
        raise ValueError("image dimensions must be 64 through 1920 and divisible by 8")
    model = onnx.load(source, load_external_data=True)
    if len(model.graph.input) != 1 or model.graph.input[0].name != "image":
        raise ValueError("expected one grayscale input named image")
    shape = model.graph.input[0].type.tensor_type.shape.dim
    if len(shape) != 4:
        raise ValueError("expected NCHW input")
    for dimension, size in zip(shape, (1, 1, height, width)):
        if dimension.dim_value not in (0, size):
            raise ValueError("requested shape conflicts with a fixed model dimension")
        dimension.ClearField("dim_param")
        dimension.dim_value = size
    # Dynamic value annotations can prevent otherwise static accelerator partitions.
    model.graph.ClearField("value_info")
    model = onnx.shape_inference.infer_shapes(model)
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.checker.check_model(model)
    encoded = model.SerializeToString()
    errors = validate(source, encoded, width, height)
    destination.write_bytes(encoded)
    receipt = {
        "input_shape": [1, 1, height, width],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(encoded).hexdigest(),
        "parity_max_absolute_error": errors,
        "scope": "CPU graph parity; accelerator placement and image quality need separate tests",
    }
    destination.with_suffix(".shape.json").write_text(json.dumps(receipt, indent=2))


def validate(source, encoded, width, height):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    original = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    fixed = ort.InferenceSession(encoded, options, providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(7)
    inputs = [np.zeros((1, 1, height, width), dtype=np.float32), generator.random((1, 1, height, width), dtype=np.float32)]
    errors = []
    for image in inputs:
        before = original.run(None, {"image": image})
        after = fixed.run(None, {"image": image})
        if len(before) != len(after):
            raise ValueError("output count changed")
        for expected, actual in zip(before, after):
            if expected.shape != actual.shape or not np.all(np.isfinite(actual)):
                raise ValueError("invalid output shape or values")
            if not np.allclose(expected, actual, rtol=1e-5, atol=1e-6):
                raise ValueError("fixed-shape model failed numerical parity")
            errors.append(float(np.max(np.abs(expected - actual))))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    args = parser.parse_args()
    specialize(args.source, args.destination, args.width, args.height)


if __name__ == "__main__":
    main()
