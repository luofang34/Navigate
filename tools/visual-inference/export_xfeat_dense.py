"""Export a fixed-shape XFeat backbone for native and browser runtimes.

Load the upstream XFeat model definition and weights from explicit paths.
Model source: verlab/accelerated_features (Apache-2.0).
Python is required only to build assets and check numerical parity.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxconverter_common import float16
from PIL import Image


class Dense(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.patch = torch.nn.Conv2d(1, 64, 8, stride=8, bias=False)
        self.patch.weight.data.copy_(
            net.keypoint_head[0].layer[0].weight.data.reshape(64, 1, 8, 8)
        )

    def forward(self, x):
        n = self.net
        x1 = n.block1(x)
        x2 = n.block2(x1 + n.skip1(x))
        x3 = n.block3(x2)
        x4 = n.block4(x3)
        x5 = n.block5(x4)
        feats = n.block_fusion(
            x3
            + F.interpolate(x4, size=x3.shape[-2:], mode="bilinear")
            + F.interpolate(x5, size=x3.shape[-2:], mode="bilinear")
        )
        k = n.keypoint_head[0].layer[2](n.keypoint_head[0].layer[1](self.patch(x)))
        for layer in n.keypoint_head[1:]:
            k = layer(k)
        return feats, k, n.heatmap_head(feats)


def export_half(model):
    # Public outputs that are consumed by the graph need separate internal names.
    for output in model.graph.output:
        public, internal = output.name, output.name + "_internal"
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == public:
                    node.input[i] = internal
            for i, name in enumerate(node.output):
                if name == public:
                    node.output[i] = internal
        for value in model.graph.value_info:
            if value.name == public:
                value.name = internal
        model.graph.node.append(
            onnx.helper.make_node(
                "Identity", [internal], [public], name=public + "_output"
            )
        )
    return float16.convert_float_to_float16(
        model, keep_io_types=True, min_positive_val=1e-8
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Upstream modules/model.py")
    parser.add_argument("weights", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir()
    spec = importlib.util.spec_from_file_location("xfeat_upstream", args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = module.XFeatModel().eval()
    base.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)
    )
    model = Dense(base).eval()
    image = np.array(Image.open(args.image).convert("L")).copy()
    x = torch.from_numpy(image).float()[None, None] / 255
    x = F.interpolate(x, size=(576, 800), mode="bilinear", align_corners=False)
    names = ["descriptors", "keypoint_logits", "reliability"]
    path = args.output / "xfeat-dense-f32.onnx"
    with torch.inference_mode():
        normalized = base.norm(x)
        expected = base(x)
        for a, b in zip(expected, model(normalized)):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=3e-5)
        torch.onnx.export(
            model,
            (normalized,),
            path,
            input_names=["image"],
            output_names=names,
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"image": normalized.numpy()})
    errors = {}
    for name, a, b in zip(names, expected, actual):
        np.testing.assert_allclose(a.numpy(), b, atol=0.003, rtol=0.0002)
        errors[name] = float(np.max(np.abs(a.numpy() - b)))
        np.save(args.output / (name + "-cpu.npy"), b)
    half = export_half(onnx.load(path))
    onnx.checker.check_model(half)
    half_path = args.output / "xfeat-dense-f16.onnx"
    onnx.save(half, half_path)
    half_session = ort.InferenceSession(
        str(half_path), options, providers=["CPUExecutionProvider"]
    )
    for name, b in zip(names, half_session.run(None, {"image": normalized.numpy()})):
        if not np.isfinite(b).all():
            raise ValueError(f"Nonfinite half-precision output: {name}")
    normalized.numpy().tofile(args.output / "input.f32")
    (args.output / "inputs.json").write_text(
        json.dumps(
            [{"name": "image", "shape": list(normalized.shape), "path": "input.f32"}]
        )
    )
    report = {
        "source_sha256": digest(args.source),
        "weights_sha256": digest(args.weights),
        "image_sha256": digest(args.image),
        "fp32_sha256": digest(path),
        "fp16_sha256": digest(half_path),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "output_max_abs_error": errors,
        "contract": "Normalized grayscale 1x1x576x800; descriptors 64x72x100, logits 65x72x100, reliability 1x72x100",
        "precision_validation": "Float16 requires separate device and matching checks",
    }
    (args.output / "export.json").write_text(json.dumps(report, indent=2))
    logging.getLogger(__name__).info("Wrote %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
