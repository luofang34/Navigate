"""Exercise cropped photo matching through padding and native-pixel refinement."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from photo_bundle_pairs import matching_window, match_edges
from photo_testset import digest


class FeatureMatcher:
    color = False

    def __init__(self, variable_sizes):
        self.variable_sizes = variable_sizes

    def match(self, reference, query):
        detector = cv2.ORB_create(nfeatures=2000)
        a, da = detector.detectAndCompute(cv2.imread(str(reference), 0), None)
        b, db = detector.detectAndCompute(cv2.imread(str(query), 0), None)
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        pairs = [[a[m.queryIdx].pt, b[m.trainIdx].pt] for m, n in matches
                 if m.distance < .7 * n.distance]
        return {'pairs': pairs}


class PhotoBundleCropTests(unittest.TestCase):
    def test_reflection_islands_do_not_retain_empty_sky(self):
        mask = np.zeros((480, 640), np.uint8)
        mask[210:470, 10:630] = 255
        mask[:10, :10] = 255
        self.assertEqual(matching_window(mask), (10, 210, 630, 470))

    def test_separate_large_windows_remain_available(self):
        mask = np.zeros((480, 640), np.uint8)
        mask[50:450, 10:250] = 255
        mask[70:400, 400:630] = 255
        self.assertEqual(matching_window(mask), (10, 50, 630, 450))

    def test_empty_mask_fails_before_matching(self):
        with self.assertRaisesRegex(ValueError, 'no valid matching pixels'):
            matching_window(np.zeros((480, 640), np.uint8))

    def test_crop_and_padding_restore_original_photo_coordinates(self):
        image = cv2.GaussianBlur(np.random.default_rng(21).integers(
            0, 256, (480, 640), np.uint8), (5, 5), .8)
        query = cv2.warpAffine(image, np.array([[1., 0, 17], [0, 1., 9]]), (640, 480))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cases = []
            for i, (pixels, top, left) in enumerate([(image, 200, 10), (query, 170, 30)]):
                mask = np.zeros_like(pixels); mask[top:470, left:630] = 255
                mask[:5, :5] = 255
                photo = root / f'image-{i}.png'; valid = root / f'mask-{i}.png'
                cv2.imwrite(str(photo), pixels); cv2.imwrite(str(valid), mask)
                cases.append({'image': photo.name, 'valid_mask': valid.name,
                              'image_sha256': digest(photo), 'mask_sha256': digest(valid)})
            for variable_sizes in [False, True]:
                with self.subTest(variable_sizes=variable_sizes):
                    output = root / str(variable_sizes); output.mkdir()
                    edges, reports = match_edges(cases, root, output,
                                                 FeatureMatcher(variable_sizes), False)
                    a, b, pairs, clear = edges[0]
                    self.assertEqual((a, b), (1, 0))
                    self.assertGreater(len(pairs), 100)
                    error = np.linalg.norm(pairs[:, 0] - pairs[:, 1] - [17, 9], axis=1)
                    self.assertLess(np.quantile(error, .9), .2)
                    self.assertTrue(clear.all())
                    self.assertTrue((pairs[:, 1, 1] >= 200).all())
                    self.assertEqual(reports[0]['crop_xyxy'],
                                     [[30, 170, 630, 470], [10, 200, 630, 470]])


if __name__ == '__main__':
    unittest.main()
