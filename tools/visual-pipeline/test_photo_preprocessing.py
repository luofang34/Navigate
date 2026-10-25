"""Cropped evidence must retain original coordinates, support and drift rejection."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from photo_preprocessing import WindowMatcher, select_preprocessing
from photo_render_check import alignment


class ScaledFeatureMatcher:
    color = True
    patch_refinement = False

    def match(self, reference, query, refinement=False):
        features = []
        for path in [reference, query]:
            image = cv2.imread(str(path), 0)
            scale = 256 / max(image.shape)
            small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
            detector = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.008, edgeThreshold=20)
            points, descriptors = detector.detectAndCompute(small, None)
            pixels = (np.asarray([p.pt for p in points]).reshape(-1, 2) + .5) / scale - .5
            features.append((pixels, descriptors))
        (a, da), (b, db) = features
        pairs = []
        if da is not None and db is not None and len(db) >= 2:
            neighbors = cv2.BFMatcher().knnMatch(da, db, k=2)
            pairs = [[a[m.queryIdx], b[m.trainIdx]] for m, n in neighbors if m.distance < .7 * n.distance]
        return {'pairs': np.asarray(pairs).reshape(-1, 2, 2), 'backend': 'scaled-sift'}


class PhotoPreprocessingTests(unittest.TestCase):
    def test_crop_recovers_features_and_still_rejects_displacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.zeros((480, 640, 3), np.uint8)
            image[210:470, 90:630] = cv2.GaussianBlur(np.random.default_rng(4).integers(
                0, 256, (260, 540, 3), np.uint8), (3, 3), .6)
            mask = np.zeros(image.shape[:2], np.uint8); mask[210:470, 90:630] = 255
            reference = root / 'reference.png'; query = root / 'query.png'
            cv2.imwrite(str(reference), image); cv2.imwrite(str(query), image)
            matcher, selected = select_preprocessing(ScaledFeatureMatcher(), reference, query,
                                                     mask, mask, root / 'selection')
            self.assertTrue(selected['selected'].startswith('crop'), selected)
            good = next(r for r in selected['attempts'] if r['variant'] == selected['selected'])
            self.assertTrue(good['passed'], good)
            self.assertGreater(good['pairs'], selected['attempts'][0]['pairs'])
            with np.load(root / 'selection' / (selected['selected'] + '.npz')) as saved:
                self.assertTrue((saved['pairs'][:, :, 0] >= 90).all())
                self.assertTrue((saved['pairs'][:, :, 1] >= 210).all())
            self.assertLess(good['image_support_fraction'], .46)
            cv2.imwrite(str(reference), cv2.warpAffine(image, np.array([[1., 0, 16.875], [0, 1., 0]]), (640, 480)))
            wrong, _, _ = alignment(matcher.match(reference, query)['pairs'], mask, mask)
            self.assertGreater(wrong['pairs'], 30)
            self.assertFalse(wrong['passed'], wrong)
            self.assertGreater(wrong['median_px'], 10)

    def test_grayscale_uses_the_same_decoder_as_the_roma_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'color.png'
            image = np.random.default_rng(6).integers(0, 256, (160, 200, 3), np.uint8)
            cv2.imwrite(str(path), image)
            matcher = WindowMatcher(ScaledFeatureMatcher(), 'crop-gray', [20, 30, 180, 150])
            matcher.match(path, path)
            actual = cv2.imread(str(root / 'crop-gray/reference.png'), 0)
            expected = cv2.imread(str(path), 0)[30:150, 20:180]
            np.testing.assert_array_equal(actual, expected)

    def test_empty_visible_area_does_not_invent_support(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); empty = np.zeros((48, 64), np.uint8)
            matcher, result = select_preprocessing(ScaledFeatureMatcher(), root / 'absent.png',
                root / 'absent.png', empty, empty, root)
            self.assertEqual(result['attempts'], [])
            self.assertEqual(matcher.variant, 'full-color')

    def test_small_crop_cannot_claim_whole_image_support(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'photo.png'
            image = np.random.default_rng(8).integers(0, 256, (480, 640, 3), np.uint8)
            cv2.imwrite(str(path), image)
            mask = np.zeros((480, 640), np.uint8); mask[200:240, 100:160] = 255
            matcher = WindowMatcher(ScaledFeatureMatcher(), 'crop-color', [100, 200, 160, 240])
            result, _, _ = alignment(matcher.match(path, path)['pairs'], mask, mask)
            self.assertGreater(result['pairs'], 30)
            self.assertLess(result['median_px'], .01)
            self.assertLess(result['image_support_fraction'], .01)
            self.assertFalse(result['passed'], result)

    def test_cli_rejects_search_combined_with_forced_grayscale(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('photo_refine.py')),
            'profile.json', 'output', '--catalog', 'catalog.json', '--run', 'seed',
            '--backend', 'roma', '--preprocessing-search', '--grayscale'],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires color RoMa', result.stderr)

    def test_pipeline_exposes_context_and_automatic_preprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); recipe = root / 'recipe.json'
            recipe.write_text(json.dumps({'source':'photos','dataset':'dataset','worker':'worker',
                'segmentation_model':'segmentation','loftr_model':'loftr','onnx_runtime':'runtime',
                'reference_catalog':'catalog.json','roma_model':'roma','roma_encoder':'encoder',
                'passes':[{'id':'seed'}, {'id':'refine','kind':'refine','parents':['seed'],
                    'backend':'roma','context_catalog':'wide.json','preprocessing_search':True}]}))
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('photo_pipeline.py')),
                str(recipe), '--run-id', 'trial', '--dry-run'], capture_output=True, text=True, check=True)
            actions = json.loads(result.stdout)
            command = next(command for name, command in actions if name == 'match/refine')
            self.assertIn('--preprocessing-search', command)
            self.assertEqual(command[command.index('--context-catalog') + 1], str((root / 'wide.json').resolve()))


if __name__ == '__main__':
    unittest.main()
