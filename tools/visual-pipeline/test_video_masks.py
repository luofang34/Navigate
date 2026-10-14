"""Transient dark blades are masked; static dark terrain and the sky are kept."""
import unittest

import cv2
import numpy as np

from video_masks import PropellerDetector, frame_masks


def scene():
    image = np.random.default_rng(3).integers(150, 230, (270, 480, 3), dtype=np.uint8)
    # Shadowed terrain is as dark as a blade but does not change between frames.
    image[200:, :120] = 40
    return image


def blade(image, angle):
    frame = image.copy()
    centre = np.array([240, 260])
    tip = centre + 230 * np.array([np.cos(np.radians(angle)), -np.sin(np.radians(angle))])
    cv2.line(frame, tuple(centre), tuple(tip.round().astype(int)), (30, 20, 20), 18)
    return frame


class PropellerTests(unittest.TestCase):
    def test_moving_blade_is_found_and_static_dark_terrain_is_not(self):
        detector = PropellerDetector()
        image = scene()
        for angle in [20, 50, 80, 110, 140, 170]:
            detector.update(blade(image, angle))
        mask = detector.update(blade(image, 65))
        tip = (240 + round(150 * np.cos(np.radians(65))), 260 - round(150 * np.sin(np.radians(65))))
        self.assertEqual(mask[tip[1], tip[0]], 255)
        self.assertFalse(mask[230:, :100].any())
        self.assertFalse(mask[:, 400:].any())

    def test_first_frames_do_not_claim_a_blade(self):
        detector = PropellerDetector()
        image = scene()
        for angle in [20, 50, 80]:
            self.assertFalse(detector.update(blade(image, angle)).any())

    def test_blade_is_removed_from_sky_and_ground(self):
        detector = PropellerDetector()
        image = scene()
        labels = np.full(image.shape[:2], 2, np.uint8)
        labels[135:] = 94
        confidence = np.ones(labels.shape, np.float32)
        for angle in [20, 50, 80, 110, 140, 170]:
            detector.update(blade(image, angle))
        masks = frame_masks(blade(image, 90), labels, confidence, detector)
        self.assertEqual(masks['propeller'][80, 240], 255)
        self.assertEqual(masks['sky'][80, 240], 0)
        self.assertEqual(masks['propeller'][180, 240], 255)
        self.assertEqual(masks['ground'][180, 240], 0)
        self.assertEqual(masks['sky'][40, 400], 255)
        self.assertEqual(masks['ground'][230, 400], 255)


if __name__ == '__main__':
    unittest.main()
