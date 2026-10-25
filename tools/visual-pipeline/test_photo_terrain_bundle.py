"""Exercise terrain-envelope errors, image refinement and held-out bundle verification."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from photo_bundle_fit import solve, support
from photo_bundle_pairs import refine_flow, selected_pixels
from photo_pair_depth import forest_check
from photo_pose import spatial_holdout
from photo_reference import tile
from photo_surface import Surface, frame, reframe_camera
from photo_terrain_bundle import inputs
from photo_terrain_envelope import envelope_residual, project
from photo_testset import digest


class TerrainBundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name); z = 15; x, y = tile(41, -75, z)
        lon = (x + .5) / 2**z * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + .5) / 2**z))))
        tiles = []
        for kind, color in [('imagery', (20, 80, 120)), ('dem', (0, 100, 128))]:
            path = root / (kind + '.png')
            self.assertTrue(cv2.imwrite(str(path), np.full((256, 256, 3), color, np.uint8)))
            tiles.append({'kind': kind, 'xyz': [z, x, y], 'path': path.name, 'sha256': digest(path)})
        catalog = root / 'catalog.json'
        catalog.write_text(json.dumps({'complete': True, 'configuration': {
            'bbox': [lon - .01, lat - .01, lon + .01, lat + .01]}, 'tiles': tiles}))
        self.surface = Surface(catalog)
        center = self.surface.local(lat, lon, 100)
        rng = np.random.default_rng(41)
        self.points = center + rng.uniform([-200, -160, 5], [200, 160, 55], (600, 3))
        K = np.array([[480., 0, 319.5], [0, 480., 239.5], [0, 0, 1]])
        rotation = np.diag([1., -1., -1.]); vector = cv2.Rodrigues(rotation)[0].ravel()
        self.cameras = [(center + [x, y, 800], vector, rotation, K)
                        for x, y in [(-70, 0), (0, 40), (70, 0)]]
        self.pixels = [cv2.projectPoints(self.points - eye, vector, np.zeros(3), K, None)[0].reshape(-1, 2)
                       for eye, _, _, _ in self.cameras]
        self.edges = [(a, b, np.stack([self.pixels[a], self.pixels[b]], axis=1), np.ones(600, bool))
                      for a, b in [(1, 0), (2, 0), (1, 2)]]
        self.views = [{'eye': eye + [4, -3, 6], 'rotation': rotation.T @ np.diag([1, -1, -1]),
            'K': K, 'size': np.array([640, 480]), 'points': self.points, 'pixels': pixels,
            'inliers': np.ones(600, bool), 'direct_map_anchor': i > 0,
            'prior': {'focal_35mm_mm': 480 * np.hypot(36, 24) / 800}}
            for i, ((eye, _, rotation, K), pixels) in enumerate(zip(self.cameras, self.pixels))]

    def test_envelope_separates_valid_canopy_from_wrong_baseline(self):
        _, _, pairs, _ = self.edges[0]
        cameras = [self.cameras[1], self.cameras[0]]
        residual, valid = envelope_residual(cameras, pairs, self.surface)
        self.assertTrue(valid.all())
        self.assertLess(np.median(np.linalg.norm(residual, axis=1)), .02)
        wrong = list(cameras); eye, vector, rotation, K = wrong[1]
        wrong[1] = (eye + [-70, 0, 0], vector, rotation, K)
        residual, _ = envelope_residual(wrong, pairs, self.surface)
        self.assertGreater(np.median(np.linalg.norm(residual, axis=1)), 10)
        outside = [(eye + [10000, 0, 0], vector, rotation, K) for eye, vector, rotation, K in cameras]
        residual, valid = envelope_residual(outside, pairs, self.surface)
        self.assertFalse(valid.any()); self.assertTrue((residual == 100).all())

    def test_bundle_recovers_poses_but_does_not_promote_them(self):
        cameras, report = solve(self.views, self.edges, self.surface)
        self.assertTrue(report['optimizer_converged'])
        self.assertFalse(report['geometry_accepted']); self.assertFalse(report['absolute_pose_verified'])
        for actual, expected in zip(cameras, self.cameras):
            self.assertLess(np.linalg.norm(actual[0] - expected[0]), 3)
        for held in report['held'].values():
            self.assertLess(held['median_epipolar_px'], .1)
            self.assertGreater(held['clear_terrain']['within_bounds_fraction'], .95)

    def test_renderer_does_not_project_photos_onto_dem_coverage_edges(self):
        eye, _, rotation, K = self.cameras[0]
        small = K.copy(); small[:2] *= .1
        small[0, 2] = (K[0, 2] + .5) * .1 - .5
        small[1, 2] = (K[1, 2] + .5) * .1 - .5
        pose = rotation.T @ np.diag([1, -1, -1])
        _, depth, _ = self.surface.render(eye, pose, small, (64, 48))
        self.assertGreater(np.count_nonzero(depth), 2000)
        _, outside, _ = self.surface.render(eye + [10000, 0, 0], pose, small, (64, 48))
        self.assertEqual(np.count_nonzero(outside), 0)

    def test_two_views_need_two_direct_anchors_and_recover_both_poses(self):
        views = self.views[1:]
        edges = [(0, 1, self.edges[2][2], self.edges[2][3])]
        cameras, report = solve(views, edges, self.surface)
        self.assertTrue(report['optimizer_converged'])
        for actual, expected in zip(cameras, self.cameras[1:]):
            self.assertLess(np.linalg.norm(actual[0] - expected[0]), 3)
        self.assertLess(report['held']['0-1']['median_epipolar_px'], .1)
        self.assertFalse(report['geometry_accepted'])
        views = [dict(view) for view in views]; views[0]['direct_map_anchor'] = False
        with self.assertRaisesRegex(ValueError, 'at least two direct map anchors'):
            solve(views, edges, self.surface)

    def test_optimistic_gps_accuracy_does_not_override_image_geometry(self):
        views = [dict(view, prior=dict(view['prior'])) for view in self.views]
        for view, camera in zip(views, self.cameras):
            lat, lon, altitude = self.surface.geodetic(camera[0] + [60, -45, 0])
            view['prior']['position'] = {'latitude': float(lat), 'longitude': float(lon),
                'altitude_m': float(altitude), 'reported_horizontal_error_m': 1.}
        cameras, report = solve(views, self.edges, self.surface)
        self.assertTrue(report['optimizer_converged'])
        for actual, expected in zip(cameras, self.cameras):
            self.assertLess(np.linalg.norm(actual[0] - expected[0]), 3)
        self.assertFalse(report['geometry_accepted'])

    def test_held_corruption_cannot_enter_training_consensus(self):
        training, _, _ = support(self.views, self.edges)
        changed = [(a, b, pairs.copy(), mask) for a, b, pairs, mask in self.edges]
        for a, b, pairs, _ in changed:
            held = spatial_holdout(pairs[:, 1], self.views[b]['size'])
            pairs[held, 0] += [30, 0]
        other, validation, _ = support(self.views, changed)
        for (_, _, a), (_, _, b) in zip(training, other):
            np.testing.assert_array_equal(a, b)
        a, b, _, clear = validation[0]
        self.assertEqual(forest_check([self.cameras[a], self.cameras[b]], clear, self.surface)['status'],
                         'terrain_conflict')

    def test_native_flow_respects_resized_pixel_centers(self):
        rng = np.random.default_rng(7)
        image = cv2.GaussianBlur(rng.integers(0, 256, (320, 320), np.uint8), (5, 5), 1)
        translation = np.array([12.5, -4.75])
        other = cv2.warpAffine(image, np.c_[np.eye(2), translation], (320, 320))
        pixels = np.array([(x, y) for x in range(20, 100, 8) for y in range(20, 100, 8)], float)
        pairs = np.stack([pixels, pixels + [4.5, -1.5]], axis=1)
        refined = refine_flow([image, other], [np.array([128, 128])] * 2, pairs)
        self.assertGreater(len(refined), len(pairs) // 2)
        self.assertLess(np.median(np.linalg.norm(refined[:, 1] - refined[:, 0] - translation / 2.5, axis=1)), .1)
        bad = np.array([[[1, 1], [1, 1]], [[-1, 1], [1, 1]], [[1, 1], [np.nan, 1]]])
        np.testing.assert_array_equal(selected_pixels(bad, [np.ones((8, 8))] * 2), [True, False, False])

    def test_foreign_seeds_preserve_map_projections_and_require_source_catalog(self):
        root = self.surface.root; catalog = root / 'catalog.json'
        source = root / 'source.json'
        source.write_text(json.dumps({'configuration': {'bbox': [-76, 40, -75, 41]}}))
        source_frame = frame(json.loads(source.read_text()))
        destination = (self.surface.basis, self.surface.origin)
        run = root / 'seeds'; run.mkdir()
        (run / 'run.json').write_text(json.dumps({'catalog_sha256': digest(source)}))
        image = root / 'image.png'; mask = root / 'mask.png'
        self.assertTrue(cv2.imwrite(str(image), np.zeros((480, 640, 3), np.uint8)))
        self.assertTrue(cv2.imwrite(str(mask), np.full((480, 640), 255, np.uint8)))
        cases = []
        for i, view in enumerate(self.views):
            identifier = f'q-{i}'; directory = run / identifier; directory.mkdir()
            eye, rotation = reframe_camera(view['eye'], view['rotation'], destination, source_frame)
            points = (view['points'] @ destination[0].T + destination[1] - source_frame[1]) @ source_frame[0]
            np.savez_compressed(directory / 'candidate.npz', **{
                **{k: view[k] for k in ['K', 'size', 'pixels', 'inliers']},
                'eye': eye, 'rotation': rotation, 'points': points})
            (run / (identifier + '.json')).write_text(json.dumps({'id': identifier,
                'status': 'candidate_requires_review', 'candidate': {'score': {
                    'geometry_accepted': True, 'inliers': 600, 'holdout_median_px': 1.,
                    'image_support_fraction': .3, 'holdout_inlier_fraction': .9}}}))
            cases.append({'id': identifier, 'image': image.name, 'valid_mask': mask.name,
                'image_sha256': digest(image), 'mask_sha256': digest(mask),
                'prior': {**view['prior'], 'camera_model': 'test', 'lens_model': 'fixed'}})
        identifiers = [case['id'] for case in cases]
        (root / 'manifest.json').write_text(json.dumps({'flights': [{'photos': identifiers}]}))
        profile = root / 'profile.json'
        profile.write_text(json.dumps({'profile': 'with-prior', 'root': '.', 'cases': cases}))
        args = SimpleNamespace(profile=profile, catalog=catalog, seed_run=[run],
                               source_catalog=[], case=identifiers)
        with self.assertRaisesRegex(ValueError, 'explicit source catalog'):
            inputs(args)
        args.source_catalog = [source]
        _, _, converted, hashes = inputs(args)
        self.assertEqual(hashes[str(source)], digest(source))
        for expected, actual in zip(self.views, converted):
            np.testing.assert_allclose(actual['eye'], expected['eye'], atol=1e-8)
            np.testing.assert_allclose(actual['points'], expected['points'], atol=1e-8)
            rotation = np.diag([1, -1, -1]) @ actual['rotation'].T
            vector = cv2.Rodrigues(rotation)[0].ravel()
            camera = (actual['eye'], vector, rotation, actual['K'])
            expected_rotation = np.diag([1, -1, -1]) @ expected['rotation'].T
            expected_camera = (expected['eye'], cv2.Rodrigues(expected_rotation)[0].ravel(),
                               expected_rotation, expected['K'])
            np.testing.assert_allclose(project(camera, actual['points']),
                                       project(expected_camera, expected['points']), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
