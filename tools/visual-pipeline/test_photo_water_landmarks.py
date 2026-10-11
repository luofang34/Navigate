"""Landmark association and geometry checks retain independent map evidence."""
import unittest

import numpy as np

from photo_water_landmarks import associate_footprints, pair_regions, polygon_distance, landmark_residual


class WaterLandmarkTests(unittest.TestCase):
    def test_reflection_cannot_match_a_nearer_subpixel_water_footprint(self):
        tiny = np.array([[9.9, 9.9], [10.1, 9.9], [10.1, 10.1], [9.9, 10.1]])
        lake = np.array([[5., 6], [25, 6], [25, 14], [5, 14]])
        second = np.array([[75., 55], [85, 55], [85, 65], [75, 65]])
        targets = np.array([[10., 10], [80, 60]])
        columns, _, areas = associate_footprints([tiny, lake, second], targets, [32, 25])
        np.testing.assert_array_equal(columns, [1, 2])
        np.testing.assert_allclose(areas, [160, 100])
        with self.assertRaisesRegex(ValueError, 'no plausible'):
            associate_footprints([tiny, second], targets, [32, 25])
        with self.assertRaisesRegex(ValueError, 'no plausible'):
            associate_footprints([lake, second], targets + [1000, 0], [32, 25])

    def test_concave_bays_do_not_count_as_water(self):
        shoreline = np.array([[0., 0], [6, 0], [6, 6], [4, 6], [4, 2], [2, 2], [2, 6], [0, 6]])
        points = np.array([[1., 5], [5, 5], [3, 4], [3, 1], [7, 3], [2, 3]])
        expected = [0, 0, 1, 0, 1, 0]
        np.testing.assert_allclose(polygon_distance(points, shoreline), expected)
        np.testing.assert_allclose(polygon_distance(points, shoreline[::-1]), expected)

    def test_image_cycle_pairing_rejects_glare_without_a_counterpart(self):
        def region(x, y, n=25):
            return {'center': np.array([x, y], float), 'pixels': np.tile([x, y], (n, 1)).astype(float)}
        a = [region(20, 30), region(65, 50), region(200, 10)]
        b = [region(70, 57), region(25, 37), region(400, 300)]
        H = np.array([[1., 0, 5], [0, 1, 7], [0, 0, 1]])
        observations, pairs = pair_regions([a, b], H)
        self.assertEqual(len(pairs), 2)
        np.testing.assert_array_equal(observations[0][0], a[0]['pixels'])
        np.testing.assert_array_equal(observations[1][0], b[1]['pixels'])
        with self.assertRaisesRegex(ValueError, 'two independent photo pairs'):
            pair_regions([a, [region(25, 37), region(400, 300)]], H)

    def test_held_landmark_is_not_in_training_residual(self):
        camera = (np.zeros(3), np.zeros(3), np.eye(3), np.eye(3))
        polygon = np.array([[0., 0, 1], [3, 0, 1], [3, 3, 1], [0, 3, 1]])
        observed = [[np.array([[1., 1]]), np.array([[2., 2]])]] * 2
        polygons = [polygon, polygon + [10, 0, 0]]
        np.testing.assert_array_equal(landmark_residual([camera] * 2, observed, polygons), [0, 0])
        np.testing.assert_allclose(landmark_residual([camera] * 2, observed, polygons, (1,)), [8, 8])


if __name__ == '__main__':
    unittest.main()
