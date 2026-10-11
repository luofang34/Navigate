"""Check image correspondence refinement across lens scale and limited overlap."""
import unittest

import cv2
import numpy as np

from photo_bundle_pairs import refine_flow


class PhotoFlowTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        self.image = cv2.GaussianBlur(rng.integers(0, 256, (640, 640), np.uint8), (5, 5), 1)
        self.transform = np.vstack([cv2.getRotationMatrix2D((319.5, 319.5), 17, 2.2), [0, 0, 1.]])
        self.query = cv2.warpPerspective(self.image, self.transform, (640, 640))
        native = np.array([(x, y) for y in range(240, 401, 16) for x in range(240, 401, 16)], float)
        mapped = cv2.perspectiveTransform(native[None], self.transform)[0]
        self.pairs = np.stack([(native + .5) / 2.5 - .5,
                               (mapped + .5) / 2.5 - .5 + [.3, -.2]], axis=1)
        self.sizes = [np.array([256, 256])] * 2

    def test_zoom_rotation_and_limited_overlap_recover_actual_pixels(self):
        refined = refine_flow([self.image, self.query], self.sizes, self.pairs)
        self.assertGreater(len(refined), len(self.pairs) * .8)
        expected = cv2.perspectiveTransform(((refined[:, 0] + .5) * 2.5 - .5)[None],
                                            self.transform)[0]
        expected = (expected + .5) / 2.5 - .5
        self.assertLess(np.median(np.linalg.norm(refined[:, 1] - expected, axis=1)), .15)

    def test_plausible_coordinates_do_not_validate_unrelated_image(self):
        unrelated = np.random.default_rng(17).integers(0, 256, (640, 640), np.uint8)
        refined = refine_flow([self.image, unrelated], self.sizes, self.pairs)
        self.assertEqual(len(refined), 0)

    def test_different_image_sizes_preserve_query_pixel_centers(self):
        query = self.query[:600, :480].copy()
        pairs = self.pairs.copy()
        pairs[:, 1] = (pairs[:, 1] + .5) * 2.5 / 2 - .5
        inside = ((pairs[:, 1] > 25) & (pairs[:, 1] < [215, 275])).all(axis=1)
        refined = refine_flow([self.image, query], [self.sizes[0], np.array([240, 300])], pairs[inside])
        self.assertGreater(len(refined), inside.sum() * .8)
        expected = cv2.perspectiveTransform(((refined[:, 0] + .5) * 2.5 - .5)[None],
                                            self.transform)[0]
        self.assertLess(np.median(np.linalg.norm(refined[:, 1] - ((expected + .5) / 2 - .5), axis=1)), .15)

    def test_degenerate_and_nonfinite_associations_are_rejected(self):
        for pairs in [self.pairs[:5], np.zeros_like(self.pairs), np.full_like(self.pairs, np.nan)]:
            with self.subTest(pairs=pairs.shape):
                refined = refine_flow([self.image, self.query], self.sizes, pairs)
                self.assertEqual(refined.shape, (0, 2, 2))


if __name__ == '__main__':
    unittest.main()
