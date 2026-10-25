#!/usr/bin/env python3
"""Exercise visual registration through real feature extraction and matching."""
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).with_name("prepare-video-overlay.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class RegistrationTest(unittest.TestCase):
    def test_known_perspective_motion_round_trips_to_the_reference(self):
        cv2.setRNGSeed(1)
        image = np.random.default_rng(1).integers(0, 256, (360, 640), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0.6)
        transform = np.array([[0.99, 0.02, 15], [-0.01, 1.01, 8], [0.00001, -0.00002, 1.]])
        moved = cv2.warpPerspective(image, transform, (640, 360))
        detector = cv2.SIFT_create(nfeatures=1600)
        result = prepare.registration(detector.detectAndCompute(moved, None),
                                      detector.detectAndCompute(image, None), cv2.BFMatcher())
        self.assertIsNotNone(result)
        recovered, inliers, residual = result
        self.assertGreater(inliers, 100)
        points = np.float32([[[100, 100]], [[500, 100]], [[500, 260]], [[100, 260]]])
        round_trip = cv2.perspectiveTransform(cv2.perspectiveTransform(points, transform), recovered)
        self.assertLess(float(np.max(np.abs(round_trip - points))), 0.5)
        self.assertLess(residual, 1.)

    def test_textureless_frames_have_no_fabricated_pose(self):
        detector = cv2.SIFT_create()
        blank = detector.detectAndCompute(np.zeros((360, 640), np.uint8), None)
        self.assertIsNone(prepare.registration(blank, blank, cv2.BFMatcher()))

    def test_north_and_east_headings_rotate_ground_footprints(self):
        points = np.float32([[320, 180], [320, 0], [640, 180]])
        north = prepare.geographic(points, 640, 360, 400, 100, 40, -74, 0)
        east = prepare.geographic(points, 640, 360, 400, 100, 40, -74, 90)
        np.testing.assert_allclose(north[0], [-74, 40])
        self.assertGreater(north[1, 1], north[0, 1])
        self.assertGreater(north[2, 0], north[0, 0])
        self.assertGreater(east[1, 0], east[0, 0])
        self.assertLess(east[2, 1], east[0, 1])


if __name__ == "__main__":
    unittest.main()
