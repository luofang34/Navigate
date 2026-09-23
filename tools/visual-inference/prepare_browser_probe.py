"""Prepare native CPU references for the local LighterGlue WebGPU test."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    values = {
        item["name"]: np.fromfile(
            args.inputs.parent / item["path"], np.float32
        ).reshape(item["shape"])
        for item in json.loads(args.inputs.read_text())
    }
    if set(values) != {"keypoints0", "keypoints1", "descriptors0", "descriptors1"}:
        raise ValueError("Expected the four LighterGlue inputs")
    if any(value.shape[0] != 1 or value.shape[1] < 1024 for value in values.values()):
        raise ValueError("Supply at least 1024 features in each image")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(args.model), options, providers=["CPUExecutionProvider"]
    )
    cases = []
    for n in [512, 1024]:
        inputs = {key: value[:, :n, :].copy() for key, value in values.items()}
        case = {"id": f"real-features-{n}", "inputs": {}}
        for key, value in inputs.items():
            name = f"{n}-{key}.f32"
            value.tofile(args.output / name)
            case["inputs"][key] = {"shape": list(value.shape), "path": name}
        expected = session.run(None, inputs)[0]
        if not np.isfinite(expected).all():
            raise ValueError("Native reference has nonfinite assignments")
        name = f"{n}-expected.f32"
        expected.tofile(args.output / name)
        case["expected"] = name
        cases.append(case)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2))
    report = {
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "onnxruntime": ort.__version__,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.iterdir())
            if p.is_file()
        },
    }
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
