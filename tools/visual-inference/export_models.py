"""Export the supplied SuperPoint and SuperGlue weights for runtime tests."""

import argparse, json, sys, hashlib
from pathlib import Path

sys.dont_write_bytecode = True
import cv2, numpy as np, torch


class DensePoint(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image):
        n = self.net
        x = image
        for a, b, pool in [
            ("conv1a", "conv1b", True),
            ("conv2a", "conv2b", True),
            ("conv3a", "conv3b", True),
            ("conv4a", "conv4b", False),
        ]:
            x = n.relu(getattr(n, b)(n.relu(getattr(n, a)(x))))
            if pool:
                x = n.pool(x)
        scores = n.convPb(n.relu(n.convPa(x)))
        desc = n.convDb(n.relu(n.convDa(x)))
        return scores, torch.nn.functional.normalize(desc, p=2, dim=1)


class Glue(torch.nn.Module):
    def __init__(self, net, shape):
        super().__init__()
        self.net = net
        self.shape = shape

    def forward(
        self, keypoints0, scores0, descriptors0, keypoints1, scores1, descriptors1
    ):
        image = torch.zeros(self.shape)
        result = self.net(
            dict(
                keypoints0=keypoints0,
                scores0=scores0,
                descriptors0=descriptors0,
                keypoints1=keypoints1,
                scores1=scores1,
                descriptors1=descriptors1,
                image0=image,
                image1=image,
            )
        )
        return result["matches0"], result["matching_scores0"]


def save_inputs(output, name, tensors, names):
    rows = []
    for key, tensor in zip(names, tensors):
        path = output / f"{name}-{key}.f32"
        value = tensor.detach().numpy().astype("<f4")
        path.write_bytes(value.tobytes())
        rows.append(dict(name=key, shape=list(value.shape), path=path.name))
    (output / f"{name}-inputs.json").write_text(json.dumps(rows, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-directory", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--keypoints", type=int, default=512)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.model_directory))
    from models.matching import Matching

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    net = Matching(
        dict(
            superpoint=dict(
                max_keypoints=a.keypoints, nms_radius=4, keypoint_threshold=0.005
            ),
            superglue=dict(
                weights="outdoor", sinkhorn_iterations=20, match_threshold=0.2
            ),
        )
    ).eval()
    images = [
        torch.from_numpy(cv2.imread(str(path), 0).astype(np.float32) / 255)[None, None]
        for path in [a.reference, a.query]
    ]
    dense = DensePoint(net.superpoint).eval()
    torch.onnx.export(
        dense,
        (images[0],),
        str(a.output / "superpoint-dense.onnx"),
        input_names=["image"],
        output_names=["scores", "descriptors"],
        opset_version=18,
        dynamo=False,
    )
    save_inputs(a.output, "superpoint", [images[0]], ["image"])
    f = [net.superpoint({"image": image}) for image in images]
    names = []
    tensors = []
    for i, features in enumerate(f):
        for key in ["keypoints", "scores", "descriptors"]:
            names.append(key + str(i))
            tensors.append(features[key][0][None])
    glue = Glue(net.superglue, images[0].shape).eval()
    axes = {name: {2 if "descriptors" in name else 1: "n" + name[-1]} for name in names}
    axes.update(matches0={1: "n0"}, matching_scores0={1: "n0"})
    torch.onnx.export(
        glue,
        tuple(tensors),
        str(a.output / "superglue.onnx"),
        input_names=names,
        output_names=["matches0", "matching_scores0"],
        dynamic_axes=axes,
        opset_version=18,
        dynamo=False,
    )
    save_inputs(a.output, "superglue", tensors, names)
    predicted = glue(*tensors)
    (a.output / "pytorch-reference.json").write_text(
        json.dumps(
            {
                "matches0": predicted[0].tolist(),
                "matching_scores0": predicted[1].tolist(),
            },
            indent=2,
        )
    )
    (a.output / "provenance.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "keypoints": a.keypoints,
                "input_shapes": {n: list(t.shape) for n, t in zip(names, tensors)},
                "models": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in a.output.glob("*.onnx")
                },
                "scope": "Dense SuperPoint backbone and full SuperGlue matcher. Keypoint selection and final geometry are outside these timings.",
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "output": str(a.output),
                "shapes": {n: list(t.shape) for n, t in zip(names, tensors)},
            }
        )
    )


if __name__ == "__main__":
    main()
