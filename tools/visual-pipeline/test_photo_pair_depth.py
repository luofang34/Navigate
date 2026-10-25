"""Low epipolar error alone cannot validate terrain-relative depth."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from photo_pair_depth import clearance, forest_check, forest_pairs, forest_residual, forest_training_pairs
from photo_reference import tile
from photo_surface import Surface
from photo_testset import digest
from photo_water_pair import epipolar
from photo_water_context import canopy_residual


class PairDepthTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        z = 17
        x, y = tile(41, -75, z)
        lon = (x + .5) / 2**z * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + .5) / 2**z))))
        tiles = []
        for kind, color in [('imagery', (20, 80, 120)), ('dem', (0, 100, 128))]:
            path = root / (kind + '.png')
            self.assertTrue(cv2.imwrite(str(path), np.full((256, 256, 3), color, np.uint8)))
            tiles.append({'kind': kind, 'xyz': [z, x, y], 'path': path.name, 'sha256': digest(path)})
        catalog = root / 'catalog.json'
        catalog.write_text(json.dumps({'complete': True, 'configuration': {
            'bbox': [lon - .01, lat - .01, lon + .01, lat + .01]}, 'tiles': tiles}))
        self.surface = Surface(catalog)
        center = self.surface.local(lat, lon, 120)
        offsets = np.array([(a, b, 0) for a in np.linspace(-40, 40, 8) for b in np.linspace(-40, 40, 8)])
        points = center + offsets
        K = np.array([[480., 0, 319.5], [0, 480., 239.5], [0, 0, 1]])
        rotation = np.diag([1., -1., -1.])
        vector = cv2.Rodrigues(rotation)[0].ravel()
        self.cameras = [(center + [x, 0, 880], vector, rotation, K) for x in [-50, 50]]
        self.pairs = np.stack([cv2.projectPoints(points - eye, vector, np.zeros(3), K, None)[0].reshape(-1, 2)
                               for eye, _, _, _ in self.cameras], axis=1)

    def test_scaled_baseline_has_perfect_pixels_but_impossible_forest_depth(self):
        check = forest_check(self.cameras, self.pairs, self.surface)
        self.assertEqual(check['status'], 'consistent_with_clearance_prior')
        self.assertAlmostEqual(check['height_quantiles_m'][1], 20, delta=.01)
        wrong = list(self.cameras)
        eye, vector, rotation, K = wrong[1]
        wrong[1] = (eye + [20, 0, 0], vector, rotation, K)
        self.assertLess(abs(epipolar(wrong, self.pairs)).max(), 1e-9)
        check = forest_check(wrong, self.pairs, self.surface)
        self.assertEqual(check['status'], 'terrain_conflict')
        self.assertEqual(check['pairs'], len(self.pairs))
        self.assertLess(check['height_quantiles_m'][1], -100)
        self.assertGreater(np.linalg.norm(forest_residual(wrong, self.pairs, self.surface)), 100)

    def test_degenerate_or_uncovered_depth_is_not_dropped_from_denominator(self):
        zero_baseline = [self.cameras[0], self.cameras[0]]
        check = forest_check(zero_baseline, self.pairs, self.surface)
        self.assertEqual(check['valid_depth_pairs'], 0)
        self.assertEqual(check['status'], 'terrain_conflict')
        self.assertTrue((forest_residual(zero_baseline, self.pairs, self.surface) > 100).all())
        displaced = [(eye + [10000, 0, 0], vector, rotation, K)
                     for eye, vector, rotation, K in self.cameras]
        _, valid = clearance(displaced, self.pairs, self.surface)
        self.assertFalse(valid.any())
        self.assertEqual(forest_check(displaced, self.pairs, self.surface)['within_bounds_fraction'], 0)

    def test_semantic_selection_requires_both_images_and_never_wraps_pixels(self):
        a = {'labels': np.full((8, 8), 4, np.uint8)}
        b = {'labels': np.full((8, 8), 4, np.uint8)}
        b['labels'][3, 3] = 6
        pairs = np.array([[[2, 2], [2, 2]], [[2, 2], [3, 3]], [[-1, 2], [2, 2]], [[2, 2], [9, 2]]])
        np.testing.assert_array_equal(forest_pairs([a, b], pairs), [True, False, False, False])
        self.assertEqual(forest_check(self.cameras, self.pairs[:10], self.surface)['status'], 'insufficient_support')

    def test_canopy_fit_does_not_target_the_negative_validation_tolerance(self):
        shifted = [(eye - [0, 0, 23], vector, rotation, K)
                   for eye, vector, rotation, K in self.cameras]
        self.assertEqual(forest_check(shifted, self.pairs, self.surface)['status'],
                         'consistent_with_clearance_prior')
        self.assertGreater(np.linalg.norm(canopy_residual(shifted, self.pairs, self.surface)), 1)
        np.testing.assert_array_equal(canopy_residual(self.cameras, self.pairs, self.surface), 0)

    def test_depth_budget_cannot_erase_sparse_forest_support(self):
        labels = np.zeros((8, 200), np.uint8)
        labels[2, 1::4] = 4
        data = [{'labels': labels}, {'labels': labels}]
        points = np.column_stack([np.arange(200), np.full(200, 2)])
        pairs = np.stack([points, points], axis=1)
        selected = forest_training_pairs(data, pairs)
        np.testing.assert_array_equal(selected, pairs[1::4])
        self.assertEqual(len(selected), 50)
        labels[:] = 4
        bounded = forest_training_pairs(data, pairs)
        self.assertEqual(len(bounded), 120)
        np.testing.assert_array_equal(bounded[[0, -1]], pairs[[0, -1]])
        self.assertEqual(len(forest_training_pairs(data, pairs[:0])), 0)


if __name__ == '__main__':
    unittest.main()
