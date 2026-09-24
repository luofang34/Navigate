"""Build the fixed-shape LoFTR-DS asset. Deployment does not require Python.

LoFTR: Copyright SenseTime. Kornia: Copyright 2018 Kornia Team.
Both sources use Apache-2.0. This export uses dual softmax.
Adapters must discard the zero-confidence row that protects empty GPU batches.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import types

import numpy as np
import onnx
import onnxruntime as ort
import torch
from kornia.feature import LoFTR
from PIL import Image
from export_loftr_ops import attention, coarse, fine, fine_preprocess, encoder


class Export(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.net = LoFTR(pretrained=None).eval()
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["state_dict"])

    def forward(self, image0, image1):
        result = self.net({"image0": image0, "image1": image1})
        return result["keypoints0"], result["keypoints1"], result["confidence"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("query", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path")
    original = Export(args.weights).eval()
    model = Export(args.weights).eval()
    for name, operation in [("fine_preprocess", fine_preprocess), ("coarse_matching", coarse), ("fine_matching", fine)]:
        module = getattr(model.net, name)
        module.forward = types.MethodType(operation, module)
    for module in model.modules():
        if type(module).__name__ == "LinearAttention":
            module.forward = types.MethodType(attention, module)
        if type(module).__name__ == "LoFTREncoderLayer":
            module.forward = types.MethodType(encoder, module)
    images = [torch.from_numpy(np.asarray(Image.open(path).convert("L").resize((640, 480))).copy())[None, None].float() / 255 for path in [args.reference, args.query]]
    with torch.inference_mode():
        torch.onnx.export(model, tuple(images), str(args.output), input_names=["image0", "image1"], output_names=["keypoints0", "keypoints1", "confidence"], opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(args.output))
        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        for label, pair in [("image pair", images), ("blank", [torch.zeros_like(i) for i in images]), ("empty matches", [torch.zeros_like(images[0]), torch.ones_like(images[1])])]:
            expected = original(*pair)
            if label == "empty matches" and expected[0].shape[0] != 0:
                raise ValueError("The empty-match export regression must exercise zero correspondences")
            actual = session.run(None, {"image0": pair[0].numpy(), "image1": pair[1].numpy()})
            valid = actual[2] > 0
            actual = [value[valid] for value in actual]
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b.numpy(), atol=.005, rtol=.001)
            logging.info("Original model parity passed: %s", label)
    logging.info("ONNX SHA-256: %s", hashlib.sha256(args.output.read_bytes()).hexdigest())
    logging.info("Weights SHA-256: %s", hashlib.sha256(args.weights.read_bytes()).hexdigest())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
