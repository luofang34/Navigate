"""Joint proposals recover shared optics while retaining an untouched holdout."""
import unittest

import cv2
import numpy as np

from photo_line import query_line, reference_line
from photo_surface import attitude
from photo_water_pair import epipolar, solve


class WaterPairTests(unittest.TestCase):
    def test_fragmented_road_strips_preserve_centres_and_exclude_uncovered_pixels(self):
        for scale in [2, 4]:
            with self.subTest(scale=scale):
                height, width = 240 * scale, 320 * scale
                image = np.full((height, width, 3), (15, 60, 15), np.uint8)
                valid = np.full((height, width), 255, np.uint8)
                for x in range(0, width, 58 * scale):
                    cv2.rectangle(image, (x, 140 * scale),
                                  (min(width - 1, x + 50 * scale), 146 * scale),
                                  (160, 160, 160), -1)
                valid[:, 120 * scale:126 * scale] = 0
                road = reference_line(image, valid)
                self.assertIsNotNone(road)
                self.assertGreater(np.ptp(road[:, 0]), width * .8)
                self.assertLess(np.max(abs(road[:, 1] - 143 * scale)), scale)
                xy = road.round().astype(int)
                self.assertTrue((valid[xy[:, 1], xy[:, 0]] > 0).all())
                self.assertIsNone(reference_line(image, np.zeros_like(valid)))

    def test_line_proposals_work_above_the_image_midpoint(self):
        image = np.full((480, 640, 3), (20, 70, 25), np.uint8)
        cv2.line(image, (20, 90), (620, 110), (150, 150, 150), 10)
        valid = np.full((480, 640), 255, np.uint8)
        reference = reference_line(image, valid)
        self.assertIsNotNone(reference)
        self.assertLess(reference[:, 1].max(), 130)
        observed = query_line(image, valid, reference)
        self.assertIsNotNone(observed)
        self.assertLess(observed[:, 1].max(), 130)
        self.assertIsNone(query_line(image, np.zeros_like(valid), reference))

    def test_zero_baseline_cannot_score_as_perfect_epipolar_geometry(self):
        eye = np.zeros(3)
        camera = (eye, eye, np.eye(3), np.eye(3))
        errors = epipolar([camera, camera], np.zeros((10, 2, 2)))
        self.assertTrue((errors > 100).all())

    def pair_scene(self):
        rng = np.random.default_rng(31)
        K = np.array([[480., 0, 319.5], [0, 480., 239.5], [0, 0, 1]])
        actual = K.copy(); actual[0, 0] = actual[1, 1] = 400
        angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        shore = np.column_stack([-2500 + 300 * np.cos(angle),
                                  600 * np.sin(angle) * (1 + .2 * np.cos(angle)), np.zeros(100)])
        road = np.column_stack([np.full(80, -1200), np.linspace(-900, 900, 80), np.zeros(80)])
        scene = rng.uniform([-3800, -1000, 0], [-1200, 1000, 100], (700, 3))
        data, pairs = [], []
        for eye, heading in [(np.array([0., 0., 1000.]), 270), (np.array([80., 60., 1000.]), 267)]:
            rotation = attitude(heading, 20, 2)
            matrix = np.diag([1, -1, -1]) @ rotation.T
            vector = cv2.Rodrigues(matrix)[0].ravel()
            project = lambda points: cv2.projectPoints(points - eye, vector, np.zeros(3), actual, None)[0].reshape(-1, 2)
            observed_road = project(road)
            line = cv2.fitLine(observed_road.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
            horizon = np.linalg.inv(actual).T @ matrix @ np.array([0, 0, 1.])
            roll = np.degrees(np.arctan2(-horizon[0], horizon[1]))
            data.append({'eye': eye, 'rotation': rotation, 'vector': vector, 'K': K,
                'size': (640, 480), 'shore_world': shore, 'shore_pixels': project(shore),
                'road_world': road, 'road_origin': line[2:], 'road_normal': [-line[1], line[0]],
                'up': np.array([0, 0, 1.]), 'horizon': {'roll_deg': roll}, 'sigma_m': 30})
            pairs.append(project(scene))
        return data, np.stack(pairs, axis=1), actual

    def test_joint_fit_recovers_shared_focal_from_near_and_far_geometry(self):
        data, pairs, actual = self.pair_scene()
        _, report = solve(data, pairs)
        self.assertTrue(report['optimizer_converged'])
        self.assertAlmostEqual(report['focal_factor'], 400 / 480, delta=.02)
        self.assertLess(report['held_epipolar_median_px'], .5)
        self.assertFalse(report['geometry_accepted'])
        self.assertFalse(report['holdout_filtered_by_fitted_geometry'])

    def test_fixed_calibration_recovers_poses_without_changing_intrinsics(self):
        data, pairs, actual = self.pair_scene()
        expected = [view['eye'].copy() for view in data]
        for view in data:
            view['K'] = actual.copy()
            view['eye'] = view['eye'] + [5, -8, 3]
            view['sigma_m'] = 300
        cameras, report = solve(data, pairs, fixed_focal=True)
        self.assertTrue(report['optimizer_converged'])
        self.assertEqual(report['focal_model'], 'fixed_seed')
        self.assertEqual(report['focal_factor'], 1.)
        self.assertLess(report['held_epipolar_median_px'], .2)
        for camera, eye in zip(cameras, expected):
            np.testing.assert_array_equal(camera[3], actual)
            self.assertLess(np.linalg.norm(camera[0] - eye), 3)
        self.assertFalse(report['geometry_accepted'])


if __name__ == '__main__':
    unittest.main()
