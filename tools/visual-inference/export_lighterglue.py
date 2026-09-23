"""Export XFeat LighterGlue with finite log scores for weak visual evidence.

Model sources: verlab/accelerated_features and cvg/LightGlue (Apache-2.0).
This is an asset-build tool. Navigate does not call Python during inference.
"""

import argparse
import hashlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from kornia.feature.lightglue import LightGlue


def log_sigmoid(x):
    return -F.relu(-x) - torch.log1p(torch.exp(-torch.abs(x)))


class ExportMatcher(torch.nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        self.net = LightGlue(
            None,
            input_dim=64,
            descriptor_dim=96,
            n_layers=6,
            num_heads=1,
            flash=False,
            depth_confidence=-1,
            width_confidence=-1,
        )
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        for i in range(6):
            weights = {
                k.replace(f"self_attn.{i}", f"transformers.{i}.self_attn")
                .replace(f"cross_attn.{i}", f"transformers.{i}.cross_attn")
                .replace("matcher.", ""): v
                for k, v in weights.items()
            }
        missing, unexpected = self.net.load_state_dict(weights, strict=False)
        if not all(k.startswith("extractor.") for k in unexpected) or set(missing) - {
            "confidence_thresholds"
        }:
            raise ValueError(
                f"Incompatible LighterGlue weights: missing={missing}, unexpected={unexpected}"
            )

    def forward(self, keypoints0, keypoints1, descriptors0, descriptors1):
        n = self.net
        a, b = n.input_proj(descriptors0), n.input_proj(descriptors1)

        def encoding(k):
            projected = n.posenc.Wr(k)
            return projected.cos().repeat_interleave(
                2, -1
            ), projected.sin().repeat_interleave(2, -1)

        def rotary(t, e):
            pairs = t.reshape(1, -1, 48, 2)
            rotated = torch.stack(
                (-pairs[:, :, :, 1], pairs[:, :, :, 0]), dim=3
            ).reshape(1, -1, 96)
            return t * e[0] + rotated * e[1]

        def self_block(block, x, e):
            qkv = block.Wqkv(x).reshape(1, -1, 96, 3)
            q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
            q, k = rotary(q, e), rotary(k, e)
            message = block.out_proj(
                torch.softmax(q @ k.transpose(-1, -2) / 96**0.5, dim=-1) @ v
            )
            return x + block.ffn(torch.cat((x, message), dim=-1))

        e0, e1 = encoding(keypoints0), encoding(keypoints1)
        for layer in n.transformers:
            a, b = (
                self_block(layer.self_attn, a, e0),
                self_block(layer.self_attn, b, e1),
            )
            c = layer.cross_attn
            q0, q1 = c.to_qk(a) / 96**0.25, c.to_qk(b) / 96**0.25
            sim = q0 @ q1.transpose(-1, -2)
            m0 = c.to_out(torch.softmax(sim, dim=-1) @ c.to_v(b))
            m1 = c.to_out(torch.softmax(sim.transpose(-1, -2), dim=-1) @ c.to_v(a))
            a, b = (
                a + c.ffn(torch.cat((a, m0), dim=-1)),
                b + c.ffn(torch.cat((b, m1), dim=-1)),
            )
        final = n.log_assignment[-1]
        d0, d1 = final.final_proj(a) / 96**0.25, final.final_proj(b) / 96**0.25
        sim = d0 @ d1.transpose(-1, -2)
        return (
            F.log_softmax(sim, dim=-1)
            + F.log_softmax(sim, dim=-2)
            + log_sigmoid(final.matchability(a))
            + log_sigmoid(final.matchability(b)).transpose(-1, -2)
        )


def validate(model, session, inputs):
    with torch.inference_mode():
        expected = model(*inputs).numpy()
        net = model.net
        a, b = net.input_proj(inputs[2]), net.input_proj(inputs[3])
        e0, e1 = net.posenc(inputs[0]), net.posenc(inputs[1])
        for layer in net.transformers:
            a, b = layer(a, b, e0, e1)
        native = net.log_assignment[-1](a, b)[0][:, :-1, :-1].numpy()
        check_assignments(native, expected)
        names = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]
        actual = session.run(None, dict(zip(names, [t.numpy() for t in inputs])))[0]
        if not np.isfinite(actual).all():
            raise ValueError("The exported model produced nonfinite assignments")
        check_assignments(actual, expected)
        return {
            "shape": list(actual.shape),
            "max_abs_error": float(np.max(np.abs(actual - expected))),
        }


def check_assignments(actual, expected):
    # Rejected pairs can have very negative log scores. Check both score drift and selected pairs.
    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.001)
    np.testing.assert_allclose(np.exp(actual), np.exp(expected), atol=0.001, rtol=0.001)

    def pairs(scores):
        scores = scores[0]
        rows = scores.argmax(axis=1)
        columns = scores.argmax(axis=0)
        ids = np.arange(len(rows))
        valid = (columns[rows] == ids) & (scores[ids, rows] > np.log(0.1))
        return np.column_stack((ids[valid], rows[valid]))

    np.testing.assert_array_equal(pairs(actual), pairs(expected))


def check_log_sigmoid_export():
    class LogSigmoid(torch.nn.Module):
        def forward(self, x):
            return log_sigmoid(x)

    logits = torch.tensor([-1000.0, -100.0, -80.0, 0.0, 80.0, 1000.0])
    output = io.BytesIO()
    torch.onnx.export(
        LogSigmoid(),
        (logits,),
        output,
        input_names=["logits"],
        output_names=["scores"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(
        output.getvalue(), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"logits": logits.numpy()})[0]
    if not np.isfinite(actual).all():
        raise ValueError("Log-sigmoid export produced nonfinite values")
    np.testing.assert_allclose(
        actual, F.logsigmoid(logits).numpy(), atol=1e-6, rtol=1e-6
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validation-inputs", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    torch.manual_seed(7)
    model = ExportMatcher(args.weights).eval()
    names = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]
    sample = (
        torch.rand(1, 128, 2),
        torch.rand(1, 96, 2),
        torch.rand(1, 128, 64),
        torch.rand(1, 96, 64),
    )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample,
            args.output,
            input_names=names,
            output_names=["log_assignment"],
            dynamic_axes={
                **{n: {1: "n" if n.endswith("0") else "m"} for n in names},
                "log_assignment": {1: "n", 2: "m"},
            },
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(args.output), options, providers=["CPUExecutionProvider"]
    )
    checks = []
    for n, m in [(128, 96), (31, 47)]:
        inputs = (
            torch.rand(1, n, 2),
            torch.rand(1, m, 2),
            F.normalize(torch.rand(1, n, 64), dim=-1),
            F.normalize(torch.rand(1, m, 64), dim=-1),
        )
        checks.append(validate(model, session, inputs))
    for manifest in args.validation_inputs:
        data = {
            item["name"]: torch.from_numpy(
                np.fromfile(manifest.parent / item["path"], np.float32).reshape(
                    item["shape"]
                )
            )
            for item in json.loads(manifest.read_text())
        }
        check = validate(model, session, tuple(data[n] for n in names))
        check["input_manifest_sha256"] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
        checks.append(check)
    check_log_sigmoid_export()
    report = {
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "checks": checks,
        "contract": "Normalized pixel coordinates; unit-length row-major 64-D descriptors; log_assignment output",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    logging.getLogger(__name__).info("Wrote %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
