"""Photo-to-photo orientation seeds preserve their limited geometric meaning."""
import unittest

import cv2
import numpy as np

from photo_direction import orientation_proposal
from photo_surface import attitude


class DirectionTests(unittest.TestCase):
    def test_recovers_rotation_with_spatially_held_out_correspondences(self):
        rng = np.random.default_rng(11)
        K = np.array([[480., 0, 319.5], [0, 480., 239.5], [0, 0, 1]])
        first = attitude(270, 20, 3)
        second = attitude(262, 22, 1)
        a = np.diag([1, -1, -1]) @ first.T
        b = np.diag([1, -1, -1]) @ second.T
        transform = K @ b @ a.T @ np.linalg.inv(K)
        points = rng.uniform([30, 30], [570, 420], (500, 2))
        target = cv2.perspectiveTransform(points[:, None], transform).reshape(-1, 2)
        result = orientation_proposal(np.stack([points, target], axis=1), K, K,
                                      first, (640, 480), (640, 480))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['rotation'], second, atol=1e-6)
        self.assertFalse(result['geometry_accepted'])
        self.assertTrue(result['requires_map_verification'])
        self.assertLess(result['score']['held_median_px'], .01)

    def test_tiny_local_patch_cannot_determine_a_heading(self):
        rng = np.random.default_rng(12)
        points = rng.uniform([300, 200], [330, 230], (500, 2))
        pairs = np.stack([points, points + [2, 3]], axis=1)
        self.assertIsNone(orientation_proposal(pairs, np.eye(3), np.eye(3),
                                               np.eye(3), (640, 480), (640, 480)))


if __name__ == '__main__':
    unittest.main()
