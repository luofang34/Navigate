"""Label composition, speck removal, input colour modes, and exclusion scores."""
import unittest

import numpy as np

from airframe_labels import parse_object
from segment_finetune import (AIRFRAME, GROUND, PROPELLER, SKY, Score, compose_labels, drop_specks,
                              normalize, split_name)
from video_masks import AIRFRAME_CLASSES, PropellerDetector, class_masks


class LabelTests(unittest.TestCase):
    def test_airframe_and_propeller_override_sky(self):
        sky = np.zeros((4, 4), bool)
        sky[:2] = True
        airframe = np.zeros_like(sky)
        airframe[1, :2] = True
        propeller = np.zeros_like(sky)
        propeller[1, 0] = True
        labels = compose_labels(sky, airframe, propeller)
        self.assertEqual(labels[0, 0], SKY)
        self.assertEqual(labels[1, 1], AIRFRAME)
        self.assertEqual(labels[1, 0], PROPELLER)
        self.assertEqual(labels[3, 3], GROUND)

    def test_small_prompt_leaks_are_removed(self):
        mask = np.zeros((100, 100), bool)
        mask[10:40, 10:40] = True
        mask[80, 80] = True
        kept = drop_specks(mask)
        self.assertTrue(kept[20, 20])
        self.assertFalse(kept[80, 80])

    def test_photo_split_suffix(self):
        self.assertEqual(split_name('photos-test'), ('photos', 'test'))
        self.assertEqual(split_name('v5730'), ('v5730', ''))

    def test_box_prompt_is_separated_from_points(self):
        self.assertEqual(parse_object([['box', 1, 2, 3, 4], [5, 6, 1]]), ([1, 2, 3, 4], [[5, 6, 1]]))
        self.assertEqual(parse_object([[5, 6, 0]]), (None, [[5, 6, 0]]))


class InputTests(unittest.TestCase):
    def test_gray_input_has_equal_channels(self):
        rgb = np.random.default_rng(1).integers(0, 255, (2, 8, 8, 3), dtype=np.uint8)
        x = normalize(rgb, np.zeros(3, np.float32), np.ones(3, np.float32), 'gray')
        self.assertEqual(x.shape, rgb.shape)
        self.assertTrue(np.allclose(x[..., 0], x[..., 2]))
        colour = normalize(rgb, np.zeros(3, np.float32), np.ones(3, np.float32), 'randgray')
        self.assertFalse(np.allclose(colour[..., 0], colour[..., 2]))


class ScoreTests(unittest.TestCase):
    def test_exclusion_rates_count_any_non_ground_prediction(self):
        labels = np.array([[AIRFRAME, AIRFRAME, GROUND, GROUND]])
        prediction = np.array([[SKY, GROUND, GROUND, AIRFRAME]])
        score = Score()
        score.add(prediction != GROUND, labels, prediction)
        result = score.result()
        self.assertEqual(result['airframe_excluded'], .5)
        self.assertEqual(result['ground_wrongly_excluded'], .5)
        self.assertEqual(result['iou']['ground'], round(1 / 3, 3))


class ClassMaskTests(unittest.TestCase):
    def test_only_ground_class_is_usable(self):
        frame = np.full((40, 40, 3), 180, np.uint8)
        labels = np.full((40, 40), AIRFRAME_CLASSES.index('ground'), np.uint8)
        labels[:10] = AIRFRAME_CLASSES.index('sky')
        labels[30:] = AIRFRAME_CLASSES.index('airframe')
        masks = class_masks(frame, labels, PropellerDetector())
        self.assertEqual(masks['sky'][5, 5], 255)
        self.assertEqual(masks['ground'][20, 20], 255)
        self.assertEqual(masks['ground'][35, 5], 0)
        self.assertEqual(masks['sky'][35, 5], 0)


if __name__ == '__main__':
    unittest.main()
