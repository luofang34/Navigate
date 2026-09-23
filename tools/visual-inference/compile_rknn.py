"""Compile dense XFeat for RK3588 and check one input in the vendor simulator."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN

OUTPUT_SHAPES = {
    "descriptors": (1, 64, 72, 100),
    "keypoint_logits": (1, 65, 72, 100),
    "reliability": (1, 1, 72, 100),
}


def compare_outputs(actual, expected, max_error):
    """Reject incompatible or non-finite tensors before numeric comparison."""
    if len(actual) != len(OUTPUT_SHAPES) or len(expected) != len(OUTPUT_SHAPES):
        raise ValueError("Expected all three dense XFeat outputs")
    result = {}
    for (name, shape), value, reference in zip(OUTPUT_SHAPES.items(), actual, expected):
        if value.shape != shape or reference.shape != shape:
            raise ValueError(f"{name}: expected shape {shape}")
        if not np.isfinite(value).all() or not np.isfinite(reference).all():
            raise ValueError(f"{name}: non-finite output")
        error = np.abs(value - reference)
        result[name] = {
            "shape": list(shape),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
        }
        if result[name]["max_abs_error"] > max_error:
            raise ValueError(f"{name}: maximum absolute error exceeds {max_error}")
    return result


def simulate(rknn, source, input_path, output, max_error):
    data = np.fromfile(input_path, dtype="<f4")
    if data.size != 576 * 800 or not np.isfinite(data).all():
        raise ValueError("Input must contain 1x1x576x800 finite float32 values")
    data = data.reshape(1, 1, 576, 800)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    cpu = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    expected = cpu.run(list(OUTPUT_SHAPES), {cpu.get_inputs()[0].name: data})
    check_stage("initialize simulator", rknn.init_runtime())
    actual = rknn.inference(inputs=[data], data_format=["nchw"])
    if actual is None:
        raise ValueError("RKNN simulator returned no outputs")
    result = compare_outputs(actual, expected, max_error)
    for name, value in zip(OUTPUT_SHAPES, actual):
        np.save(output / f"{name}.npy", value)
    return {
        "scope": "compiler simulator; not board execution or pose acceptance",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "max_absolute_error_limit": max_error,
        "outputs": result,
    }


def check_stage(stage, code):
    if code not in (0, None):
        raise RuntimeError(f"RKNN {stage} failed: {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validate-input", type=Path)
    parser.add_argument("--max-absolute-error", type=float, default=0.2)
    args = parser.parse_args()
    if not np.isfinite(args.max_absolute_error) or args.max_absolute_error <= 0:
        parser.error("Maximum absolute error must be finite and positive")
    source_bytes = args.source.read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    rknn = RKNN(verbose=True, verbose_file=str(args.output / "compiler.log"))
    report = {
        "compiler": importlib.metadata.version("rknn-toolkit2"),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "target": "rk3588",
        "quantization": False,
        "device_validation": "not performed",
    }
    artifact = args.output / "xfeat-rk3588-f16.rknn"
    started = time.monotonic()
    try:
        check_stage(
            "config",
            rknn.config(target_platform="rk3588", mean_values=[[0]], std_values=[[1]]),
        )
        check_stage("load", rknn.load_onnx(model=str(args.source)))
        check_stage("build", rknn.build(do_quantization=False))
        check_stage("export", rknn.export_rknn(str(artifact)))
        report.update(
            compile_elapsed_s=time.monotonic() - started,
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            size=artifact.stat().st_size,
        )
        if args.validate_input:
            report["simulation"] = simulate(
                rknn,
                args.source,
                args.validate_input,
                args.output,
                args.max_absolute_error,
            )
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        rknn.release()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
