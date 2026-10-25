"""Relief proposals require independent evidence beyond a dominant image patch."""
import unittest
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from photo_relief_search import dominant_region, exclude_region, fixed_verification, load_seed, ground_patch_pairs


class ReliefSearchTests(unittest.TestCase):
    def test_patch_coordinates_return_to_full_image_and_sparse_areas_are_skipped(self):
        class MarkerMatcher:
            color = True

            def __init__(self):
                self.calls = 0

            def match(self, reference, query):
                self.calls += 1
                a, b = cv2.imread(str(reference), 0), cv2.imread(str(query), 0)
                np.testing.assert_array_equal(a, b)
                y, x = np.nonzero(a)
                pairs = [[[float(px), float(py)], [float(px), float(py)]] for px, py in zip(x, y)]
                # A crop-local outlier must not land in another valid quadrant after translation.
                return {'pairs': pairs + [[[-1., 2.], [2., 2.]]]}

        with tempfile.TemporaryDirectory() as temporary:
            image = np.zeros((81, 103, 3), np.uint8)
            points = np.array([[7, 9], [56, 11], [8, 47], [102, 80]])
            image[points[:, 1], points[:, 0]] = 255
            mask = np.full(image.shape[:2], 255, np.uint8)
            mask[:40, 51:] = 0
            matcher = MarkerMatcher()
            pairs = ground_patch_pairs(image, image, mask, mask, matcher, Path(temporary), 0)
            self.assertEqual(matcher.calls, 3)
            np.testing.assert_array_equal(np.asarray(pairs)[:, 0], points[[0, 2, 3]])
            np.testing.assert_array_equal(np.asarray(pairs)[:, 0], np.asarray(pairs)[:, 1])
            empty = np.zeros_like(mask)
            self.assertEqual(ground_patch_pairs(image, image, mask, empty, matcher, Path(temporary), 0), [])
            self.assertEqual(matcher.calls, 3)

    def test_failed_seed_requires_opt_in_and_retains_its_rejection_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'photo').mkdir()
            attempt = {'prior_gate': True, 'inliers': 800, 'positive_depth_fraction': 1,
                'image_support_fraction': .1, 'holdout_median_px': 2.,
                'holdout_inlier_fraction': .5, 'geometry_accepted': False,
                'seed': 12, 'step': 0, 'resolution': [480, 640]}
            (root / 'photo.json').write_text(json.dumps({'attempts': [
                {**attempt, 'seed': 1, 'prior_gate': False, 'inliers': 2000}, attempt]}))
            np.savez_compressed(root / 'photo' / 'evidence-12-00.npz',
                input_K=np.diag([400., 400., 1.]), focal_px=420.)
            with self.assertRaisesRegex(ValueError, '--unverified-seed'):
                load_seed(root, 'photo')
            state, info = load_seed(root, 'photo', True)
            self.assertTrue(info['seed_is_unverified'])
            self.assertFalse(info['seed_score']['geometry_accepted'])
            self.assertEqual(info['seed_file'], 'evidence-12-00.npz')
            np.testing.assert_array_equal(state['K'], np.diag([420., 420., 1.]))
            np.testing.assert_array_equal(state['size'], [480, 640])
            self.assertFalse((root / 'photo' / 'candidate.npz').exists())

    def test_accepted_candidate_takes_precedence_over_unverified_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'photo').mkdir()
            np.savez_compressed(root / 'photo' / 'candidate.npz', K=np.eye(3))
            state, info = load_seed(root, 'photo', True)
            self.assertFalse(info['seed_is_unverified'])
            np.testing.assert_array_equal(state['K'], np.eye(3))

    def test_dominant_region_is_selected_from_semantics_without_pixel_threshold(self):
        x, y = np.meshgrid(np.linspace(30, 180, 12), np.linspace(30, 150, 10))
        pixels = np.column_stack([x.ravel(), y.ravel()])
        labels = np.full((480, 640), 4, np.uint8)
        labels[:200, :220] = 54
        state = {'pixels': pixels, 'inliers': np.ones(len(pixels), bool)}
        selected, hull, label = dominant_region(state, labels)
        self.assertEqual(label, 54)
        self.assertEqual(len(selected), len(pixels))
        self.assertEqual(hull[90, 90], 255)
        self.assertEqual(hull[300, 300], 0)
        mask = exclude_region(np.full(labels.shape, 255, np.uint8), pixels)
        self.assertEqual(mask[20, 20], 0)
        self.assertEqual(mask[300, 300], 255)

    def test_good_roof_cannot_hide_wrong_ground(self):
        x, y = np.meshgrid(np.linspace(30, 600, 20), np.linspace(30, 440, 20))
        query = np.column_stack([x.ravel(), y.ravel()])
        region = np.zeros((480, 640), np.uint8)
        region[:, :500] = 255
        reference = query.copy()
        reference[query[:, 0] >= 500, 1] += 15
        valid = np.full(region.shape, 255, np.uint8)
        result, _, _ = fixed_verification(np.stack([reference, query], axis=1), valid, region)
        self.assertGreater(result['inliers'] / result['pairs'], .75)
        self.assertFalse(result['passed'])
        result, _, _ = fixed_verification(np.stack([query + .5, query], axis=1), valid, region)
        self.assertTrue(result['passed'])
        self.assertFalse(result['pose_refitted'])

    def test_masked_or_outside_pairs_cannot_supply_verification(self):
        region = np.zeros((480, 640), np.uint8)
        valid = np.zeros_like(region)
        pairs = np.array([[[10, 10], [10, 10]], [[700, 0], [1, 1]]])
        result, kept, _ = fixed_verification(pairs, valid, region)
        self.assertFalse(result['passed'])
        self.assertEqual(len(kept), 0)


if __name__ == '__main__':
    unittest.main()
