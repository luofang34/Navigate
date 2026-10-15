"""Rotation tracking meets both anchors and recovers a known camera rotation."""
import unittest

import cv2
import numpy as np

from video_track import dense_rotations, interpolate_eyes, kabsch, rays, relative_rotation, slerp


def rotation(axis, degrees):
    return cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(degrees))[0]


def angle(a, b):
    return np.degrees(np.linalg.norm(cv2.Rodrigues(a.T @ b)[0]))


class RotationTests(unittest.TestCase):
    def test_known_rotation_is_recovered_with_outliers(self):
        rng = np.random.default_rng(1)
        K = np.array([[800., 0, 479.5], [0, 800., 359.5], [0, 0, 1]])
        a = rays(np.column_stack([rng.uniform(0, 960, 300), rng.uniform(0, 720, 300)]), K)
        M = rotation([.2, 1, .1], 1.5)
        b = (M @ a.T).T
        b[:60] = rays(np.column_stack([rng.uniform(0, 960, 60), rng.uniform(0, 720, 60)]), K)
        found, inliers = relative_rotation(a, b, rng)
        self.assertGreaterEqual(inliers, 240)
        self.assertLess(angle(found, M), .01)

    def test_kabsch_is_a_proper_rotation(self):
        a = np.eye(3)
        self.assertAlmostEqual(np.linalg.det(kabsch(a, a @ rotation([0, 0, 1], 30).T)), 1.)

    def test_blend_meets_both_anchors_and_removes_linear_drift(self):
        truth = [rotation([0, 1, 0], 2 * i) for i in range(11)]
        # Each tracked step carries the same bias, as parallax from forward motion does.
        bias = rotation([1, 0, 0], .05)
        steps = [bias @ truth[i + 1].T @ truth[i] for i in range(10)]
        result = dense_rotations({0: truth[0], 10: truth[10]}, steps, 11)
        self.assertLess(angle(result[0], truth[0]), 1e-9)
        self.assertLess(angle(result[10], truth[10]), 1e-9)
        self.assertLess(max(angle(r, t) for r, t in zip(result, truth)), .05)

    def test_failed_step_leaves_unreachable_frames_without_pose(self):
        steps = [np.eye(3), None, np.eye(3)]
        result = dense_rotations({0: np.eye(3)}, steps, 4)
        self.assertIsNotNone(result[1])
        self.assertIsNone(result[2])
        self.assertIsNone(result[3])

    def test_slerp_midpoint_and_positions(self):
        self.assertLess(angle(slerp(np.eye(3), rotation([0, 0, 1], 40), .5), rotation([0, 0, 1], 20)), 1e-9)
        eyes = interpolate_eyes({0: np.zeros(3), 4: np.array([4., 0, 8])}, 6)
        np.testing.assert_allclose(eyes[2], [2, 0, 4])
        np.testing.assert_allclose(eyes[5], [4, 0, 8])



class RegistrationTests(unittest.TestCase):
    def test_isolated_registration_failures_and_outliers_take_neighbour_values(self):
        from video_track import smooth_shifts
        shifts = [(2., 1.)] * 4 + [None, (40., -30.)] + [(2., 1.)] * 4
        smooth = smooth_shifts(shifts)
        self.assertTrue(np.allclose(smooth, [[2., 1.]] * 10))

    def test_correction_moves_a_projected_point_by_the_measured_shift(self):
        from video_track import corrected
        f = 800.
        R = rotation([0, 1, 0], 10)
        point = R @ np.array([0., 0., -1000.])  # straight ahead of the camera
        def project(rot):
            cam = point @ rot
            return np.array([f * cam[0] / -cam[2], -f * cam[1] / -cam[2]])
        moved = project(corrected(R, (5., -3.), f)) - project(R)
        # Turning by the measured shift moves the terrain the opposite way in the image.
        np.testing.assert_allclose(moved, [-5., 3.], atol=.05)


class FocalTests(unittest.TestCase):
    def test_shared_focal_recovers_the_true_focal_from_anchor_correspondences(self):
        from video_track import shared_focal
        rng = np.random.default_rng(2)
        true_ratio, size = .84, np.array([640, 480])
        states = {}
        for frame, heading in [(0, 0), (5, 3), (10, 6)]:
            R = rotation([0, 1, 0], heading) @ rotation([1, 0, 0], -20)
            eye = np.array([frame * 2., 0., 1000.])
            dirs = rays(np.column_stack([rng.uniform(0, 640, 200), rng.uniform(240, 480, 200)]),
                        np.array([[true_ratio * 640, 0, 319.5], [0, true_ratio * 640, 239.5], [0, 0, 1]]))
            world = (R @ dirs.T).T
            depth = -eye[2] / world[:, 2]
            points = eye + world * depth[:, None]
            cam = (points - eye) @ R
            f = true_ratio * 640
            pixels = np.column_stack([f * cam[:, 0] / -cam[:, 2] + 319.5, -f * cam[:, 1] / -cam[:, 2] + 239.5])
            # A per-anchor focal that never got refined, as a stopped solve leaves it.
            K = np.array([[.75 * 640, 0, 319.5], [0, .75 * 640, 239.5], [0, 0, 1]])
            states[frame] = {'K': K, 'size': size, 'rotation': R, 'eye': eye,
                             'points': points, 'pixels': pixels, 'inliers': np.ones(200, bool)}
        self.assertAlmostEqual(shared_focal(states), true_ratio, delta=.01)


if __name__ == '__main__':
    unittest.main()
