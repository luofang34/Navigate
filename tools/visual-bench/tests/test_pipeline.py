"""Behavioral tests use a backend with no model runtime."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2
from demo.contracts import PixelPairs
from demo.geometry import depth_inliers, match_reference
from demo.pipeline import evaluate_candidates


class PipelineTests(unittest.TestCase):
    def test_nonplanar_depth_geometry_and_missing_surface(self):
        camera = dict(width=640, height=480, fx=500.0, fy=500.0, cx=319.5, cy=239.5)
        a = np.array(
            [(x, y) for y in range(60, 420, 40) for x in range(60, 600, 40)], float
        )
        rng = np.random.default_rng(6)
        d = rng.uniform(35, 120, len(a))
        depth = np.zeros((480, 640), np.float32)
        depth[a[:, 1].astype(int), a[:, 0].astype(int)] = d
        obj = np.column_stack(
            ((a[:, 0] - 319.5) / 500 * d, (a[:, 1] - 239.5) / 500 * d, d)
        )
        b = cv2.projectPoints(
            obj,
            np.array([0.09, -0.11, 0.06]),
            np.array([3.0, -2.0, 5.0]),
            np.array([[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1.0]]),
            None,
        )[0][:, 0]
        b[:12] = rng.uniform([50, 50], [590, 430], (12, 2))
        pairs = PixelPairs(a, b, "custom-coordinate-backend")
        first, _, count = depth_inliers(pairs, depth, camera)
        self.assertEqual(count, len(a))
        self.assertGreater(len(first), 100)
        self.assertLess(len(first), len(a) - 9)
        self.assertEqual(len(depth_inliers(pairs, depth * 0, camera)[0]), 0)
        depth[:] = np.nan
        self.assertEqual(len(depth_inliers(pairs, depth, camera)[0]), 0)

    def test_custom_backend_uses_the_shared_reference_boundary(self):
        camera = dict(width=96, height=72, fx=90.0, fy=90.0, cx=47.5, cy=35.5)
        points = np.array(
            [(x, y) for y in range(10, 65, 10) for x in range(10, 90, 10)], float
        )

        class Coordinates:
            def match_images(self, first, second):
                self.shapes = (first.shape, second.shape)
                return PixelPairs(points, points, "custom-coordinates")

        backend = Coordinates()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((72, 96), np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "reference.png"), image))
            self.assertTrue(cv2.imwrite(str(root / "query.png"), image))
            (root / "depth.bin").write_bytes(
                np.full((72, 96), 80.0, dtype="<f4").tobytes()
            )
            payload, metrics = match_reference(
                backend,
                root / "reference.png",
                root / "query.png",
                root / "depth.bin",
                camera,
            )
        self.assertEqual(backend.shapes, ((72, 96), (72, 96)))
        self.assertEqual(payload["backend_identity"], "custom-coordinates")
        self.assertEqual(len(payload["matches"]), len(points))
        self.assertEqual(metrics["geometric_filter"], "rendered_depth_pnp")
        self.assertEqual(len(payload["reference_depth_sha256"]), 64)

    def test_all_candidates_and_refinements_survive(self):
        calls, latest = [], {}

        class Worker:
            def request(self, value):
                self_value = list(latest.values())
                return {
                    "observation": {
                        "accepted": False,
                        "decision": "unresolved",
                        "candidate_hypotheses": self_value,
                    }
                }

        def refine(candidate, candidate_id, iteration):
            calls.append((candidate_id, iteration))
            result = dict(accepted=True, candidate_id=candidate_id, **candidate)
            latest[candidate_id] = result
            return result

        candidates = [
            dict(
                candidate=dict(position_enu_m=[x, 0, 110], eye_to_enu_xyzw=[0, 0, 0, 1])
            )
            for x in [0, 500]
        ]
        result = evaluate_candidates(Worker(), candidates, refine)
        self.assertEqual(calls, [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(len(result["candidate_hypotheses"]), 2)
        self.assertEqual(len(result["geometric_attempts"]), 4)
        self.assertNotIn("position_enu_m", result)
        self.assertFalse(result["accepted"])


if __name__ == "__main__":
    unittest.main()
