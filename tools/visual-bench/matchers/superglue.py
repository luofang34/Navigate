"""Write image-bound correspondences for visual-bench refine."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.contracts import PixelPairs

import cv2
import numpy as np
import torch


class SuperGlueMatcher:
    def __init__(
        self,
        model_directory,
        device="mps",
        keypoints=2048,
        nms_radius=4,
        keypoint_threshold=0.005,
    ):
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(Path(model_directory).resolve()))
        from models.matching import Matching

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        self._device = device
        self._cache = OrderedDict()
        cv2.setNumThreads(4)
        cv2.setRNGSeed(42)
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        self._net = (
            Matching(
                {
                    "superpoint": {
                        "nms_radius": nms_radius,
                        "keypoint_threshold": keypoint_threshold,
                        "max_keypoints": keypoints,
                    },
                    "superglue": {
                        "weights": "outdoor",
                        "sinkhorn_iterations": 20,
                        "match_threshold": 0.2,
                    },
                }
            )
            .eval()
            .to(device)
        )
        weights = Path(model_directory) / "models/weights"
        digest = hashlib.sha256()
        for filename in ["superpoint_v1.pth", "superglue_outdoor.pth"]:
            digest.update((weights / filename).read_bytes())
        self.identity = f"superpoint-superglue-outdoor-{digest.hexdigest()}-{device}-kp{keypoints}-nms{nms_radius}-threshold{keypoint_threshold}"

    def _features(self, image):
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("Expected a grayscale uint8 image")
        key = (image.shape, hashlib.sha256(image.tobytes()).digest())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        tensor = torch.from_numpy(image.astype(np.float32) / 255)[None, None].to(
            self._device
        )
        result = self._net.superpoint({"image": tensor})
        features = {
            "image": tensor,
            **{
                name: torch.stack(value) if isinstance(value, (list, tuple)) else value
                for name, value in result.items()
            },
        }
        self._cache[key] = features
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return features

    def match_images(self, first, second):
        a, b = self._features(first), self._features(second)
        result = self._net.superglue(
            {
                **{name + "0": value for name, value in a.items()},
                **{name + "1": value for name, value in b.items()},
            }
        )
        indices = result["matches0"][0].cpu().numpy()
        valid = indices >= 0
        return PixelPairs(
            a["keypoints"][0].cpu().numpy()[valid],
            b["keypoints"][0].cpu().numpy()[indices[valid]],
            self.identity,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-directory",
        type=Path,
        required=True,
        help="Directory containing the upstream models package",
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--query",
        type=Path,
        required=True,
        help="Grayscale PNG with the calibrated dimensions",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        required=True,
        help="Little-endian float32 optical depth from visual-bench render",
    )
    parser.add_argument(
        "--camera", type=Path, required=True, help="JSON camera or prior file"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--keypoint-threshold", type=float, default=0.005)
    args = parser.parse_args()
    matcher = SuperGlueMatcher(
        args.model_directory,
        args.device,
        args.max_keypoints,
        args.nms_radius,
        args.keypoint_threshold,
    )
    from demo.geometry import match_reference

    camera = json.loads(args.camera.read_text())
    payload, metrics = match_reference(
        matcher, args.reference, args.query, args.depth, camera.get("camera", camera)
    )
    with args.output.open("x") as stream:
        json.dump(payload, stream)
    with args.output.with_suffix(".metrics.json").open("x") as stream:
        json.dump(metrics, stream, indent=2)


if __name__ == "__main__":
    main()
