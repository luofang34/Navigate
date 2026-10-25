"""Aircraft evidence completes border supports without masking similar ground lines."""
import unittest

import cv2
import numpy as np

from photo_airframe import border_support_mask
from photo_masks import surface_mask


class AirframeLineTests(unittest.TestCase):
    def scene(self):
        image = np.random.default_rng(1).integers(20, 180, (480, 640, 3), dtype=np.uint8)
        cv2.fillConvexPoly(image, np.array([[150, 479], [250, 479], [639, 150], [639, 50]]), (185, 185, 185))
        labels = np.full((480, 640), 54, np.uint8)
        confidence = np.ones(labels.shape, np.float32)
        return image, labels, confidence

    def test_partial_airplane_label_completes_a_border_support(self):
        image, labels, confidence = self.scene()
        labelled = np.all(image == 185, axis=2) & (np.indices(labels.shape)[0] > 360)
        labels[labelled] = 90
        mask = border_support_mask(image, labels, confidence)
        self.assertEqual(mask[300, 420], 255)
        self.assertEqual(mask[300, 40], 0)
        self.assertEqual(mask[430, 610], 0)
        semantic = surface_mask(labels, confidence, policy='airframe-occlusions')
        completed = surface_mask(labels, confidence, policy='airframe-lines', image=image)
        self.assertEqual(semantic[300, 420], 255)
        self.assertEqual(completed[300, 420], 0)

    def test_ground_lines_and_uncertain_airplane_labels_do_not_expand(self):
        image, labels, confidence = self.scene()
        self.assertFalse(border_support_mask(image, labels, confidence).any())
        labelled = np.all(image == 185, axis=2) & (np.indices(labels.shape)[0] > 360)
        labels[labelled] = 90
        confidence[labelled] = .1
        self.assertFalse(border_support_mask(image, labels, confidence).any())

    def test_misaligned_inputs_are_rejected(self):
        image, labels, confidence = self.scene()
        with self.assertRaises(ValueError):
            border_support_mask(image[:20], labels, confidence)

    def test_parallel_ground_edge_does_not_widen_the_support(self):
        image, labels, confidence = self.scene()
        support = np.all(image == 185, axis=2)
        image[~support] = 70
        cv2.line(image, (350, 479), (639, 180), (130, 130, 130), 2)
        labels[support & (np.indices(labels.shape)[0] > 360)] = 90
        mask = border_support_mask(image, labels, confidence)
        self.assertEqual(mask[300, 420], 255)
        self.assertEqual(mask[300, 500], 0)


if __name__ == '__main__':
    unittest.main()
