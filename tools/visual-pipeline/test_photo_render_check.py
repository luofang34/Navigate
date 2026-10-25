"""Fresh rendered pixels must agree without allowing a new camera fit to hide drift."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from photo_reference import tile
from photo_pose import candidate_priority
from photo_render_check import alignment, merit, verify_pose
from photo_surface import Surface
from photo_testset import digest


class ImageMatcher:
    color = True
    patch_refinement = False

    def match(self, reference, query, refinement=False):
        detector = cv2.SIFT_create(contrastThreshold=.01, edgeThreshold=20)
        a, da = detector.detectAndCompute(cv2.imread(str(reference), 0), None)
        b, db = detector.detectAndCompute(cv2.imread(str(query), 0), None)
        neighbors = cv2.BFMatcher().knnMatch(da, db, k=2)
        pairs = [[a[m.queryIdx].pt, b[m.trainIdx].pt] for m, n in neighbors if m.distance < .7 * n.distance]
        return {'pairs': pairs, 'backend': 'opencv-sift-ratio'}


class RenderCheckTests(unittest.TestCase):
    def test_actual_terrain_render_rejects_camera_drift_without_refitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zoom = 17
            x, y = tile(41, -75, zoom)
            tiles = []
            rng = np.random.default_rng(7)
            texture = cv2.GaussianBlur(rng.integers(0, 256, (256, 256, 3), np.uint8), (3, 3), .5)
            for kind, image in [('imagery', texture), ('dem', np.full((256, 256, 3), (0, 100, 128), np.uint8))]:
                path = root / (kind + '.png')
                self.assertTrue(cv2.imwrite(str(path), image))
                tiles.append({'kind': kind, 'xyz': [zoom, x, y], 'path': path.name, 'sha256': digest(path)})
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps({'complete': True,
                'configuration': {'bbox': [-75.1, 40.9, -74.9, 41.1]}, 'tiles': tiles}))
            surface = Surface(catalog)
            lon = (x + .5) / 2**zoom * 360 - 180
            lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + .5) / 2**zoom))))
            eye = surface.local(lat, lon, 200)
            rotation = surface.camera_rotation(lat, lon, 0, 90)
            K = np.array([[180., 0, 95.5], [0, 180., 95.5], [0, 0, 1]])
            size = (192, 192)
            image, depth, _ = surface.render(eye, rotation, K, size)
            query = root / 'query.png'
            self.assertTrue(cv2.imwrite(str(query), image))
            correct = verify_pose(surface, eye, rotation, K, size, ImageMatcher(), query,
                                  depth > 0, root / 'correct')
            wrong = verify_pose(surface, eye + [8, 0, 0], rotation, K, size, ImageMatcher(),
                                query, depth > 0, root / 'wrong')
            self.assertTrue(correct['passed'], correct)
            self.assertFalse(wrong['passed'], wrong)
            self.assertTrue(correct['consensus']['aligned'], correct)
            self.assertFalse(wrong['consensus']['aligned'], wrong)
            self.assertGreater(wrong['consensus']['held_median_px'], 8)
            self.assertGreater(wrong['pairs'], 30)
            self.assertGreater(wrong['median_px'], 8)
            self.assertFalse(wrong['pose_refitted'])
            self.assertIsNone(merit(wrong))
            self.assertGreater(merit(correct), 0)
            saved = np.load(root / 'wrong/matches.npz')
            np.testing.assert_array_equal(saved['eye'], eye + [8, 0, 0])

    def test_preview_priority_preserves_the_fixed_render_winner(self):
        fitted = {'score': {'image_support_fraction': .4, 'holdout_inlier_fraction': .99,
                           'holdout_median_px': .5, 'inliers': 4000}}
        rendered = {**fitted, 'fixed_render': {'passed': True, 'image_support_fraction': .3,
                                             'inlier_fraction': .85, 'median_px': 1.8}}
        weaker = {**rendered, 'fixed_render': {**rendered['fixed_render'],
                                             'inlier_fraction': .67, 'median_px': 2.2}}
        self.assertGreater(candidate_priority(rendered), candidate_priority(fitted))
        self.assertGreater(candidate_priority(rendered), candidate_priority(weaker))
        self.assertGreater(candidate_priority(fitted),
                           candidate_priority({**rendered, 'anchor_depth': 1}))

    def test_invalid_or_unobserved_reference_pixels_do_not_supply_evidence(self):
        mask = np.ones((80, 100), np.uint8)
        reference = mask.copy()
        reference[10, 10] = 0
        pairs = [[[10, 10], [10, 10]], [[float('nan'), 3], [3, 3]],
                 [[99.7, 4], [4, 4]], [[20, 20], [20, 20]]]
        result, kept, error = alignment(pairs, mask, reference)
        np.testing.assert_array_equal(kept, [[[20, 20], [20, 20]]])
        np.testing.assert_array_equal(error, [0])
        self.assertFalse(result['passed'])
        empty, _, _ = alignment([], mask, reference)
        self.assertIsNone(empty['median_px'])
        self.assertIsNone(merit(empty))
        with self.assertRaisesRegex(ValueError, 'same dimensions'):
            alignment([], mask, reference[:40])


if __name__ == '__main__':
    unittest.main()
