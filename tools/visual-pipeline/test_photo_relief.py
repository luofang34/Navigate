"""Joint relief fitting separates elevated structures from the camera position."""
import unittest

import cv2
import numpy as np

from photo_relief import fit_relief
from photo_surface import attitude


class ReliefTests(unittest.TestCase):
    def scene(self):
        rng = np.random.default_rng(7)
        eye = np.array([0., 0., 100.])
        rotation = attitude(0, 40, -5)
        K = np.array([[480., 0, 319.5], [0, 480., 239.5], [0, 0, 1]])
        vector = cv2.Rodrigues(np.diag([1, -1, -1]) @ rotation.T)[0]
        regions = []
        for height, count in [(15, 400), (0, 200)]:
            points = np.column_stack([rng.uniform(-70, 70, count),
                                      rng.uniform(80, 200, count), np.zeros(count)])
            actual = points + [0, 0, height]
            pixels = cv2.projectPoints(actual - eye, vector, np.zeros(3), K, None)[0].reshape(-1, 2)
            pixels += rng.normal(0, .15, pixels.shape)
            regions.append((points, pixels))
        return regions, eye, rotation, K

    def test_recovers_height_and_camera_from_separate_planes(self):
        regions, eye, rotation, K = self.scene()
        saved = [points.copy() for points, _ in regions]
        result = fit_relief(regions, eye + [2, -3, 4], rotation, K, (640, 480), [0, 0, 1])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['height_offset_m'], 15, delta=.3)
        self.assertLess(np.linalg.norm(result['eye'] - eye), .5)
        for score in result['scores']:
            self.assertLess(score['held_median_px'], .5)
            self.assertEqual(score['positive_depth_fraction'], 1)
        self.assertTrue(result['requires_render_verification'])
        for (points, _), original in zip(regions, saved):
            np.testing.assert_array_equal(points, original)

    def test_missing_ground_support_does_not_produce_a_pose(self):
        regions, eye, rotation, K = self.scene()
        regions[1] = tuple(array[:10] for array in regions[1])
        self.assertIsNone(fit_relief(regions, eye, rotation, K, (640, 480), [0, 0, 1]))

    def test_invalid_gravity_direction_is_rejected(self):
        regions, eye, rotation, K = self.scene()
        with self.assertRaises(ValueError):
            fit_relief(regions, eye, rotation, K, (640, 480), [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
