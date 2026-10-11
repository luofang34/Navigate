"""Blind acquisition excludes leaked references and checks actual rendered pixels."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from photo_acquisition import acquire, focal_beam
from photo_bank import read_profile, rank_references, cached_features, pooled_descriptors
from photo_pose import fit
from photo_reference import tile
from photo_surface import Surface
from photo_testset import digest
from photo_tracking import load_anchors
from test_photo_render_check import ImageMatcher


class PhotoAcquisitionTests(unittest.TestCase):
    def test_blind_profile_and_reference_partition_prevent_metadata_and_capture_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'profile.json'
            case = {'id': 'q-query', 'image_sha256': 'pixels'}
            path.write_text(json.dumps({'profile': 'no-prior', 'root': '.', 'cases': [case]}))
            self.assertEqual(read_profile(path, blind=True)[0], {'q-query': case})
            for hidden in [{'prior': {'position': {'latitude': 41}}}, {'focal_35mm_mm': 26}, {'capture_time': 'no'}]:
                path.write_text(json.dumps({'profile': 'no-prior', 'root': '.', 'cases': [{**case, **hidden}]}))
                with self.assertRaisesRegex(ValueError, 'metadata-free'):
                    read_profile(path, blind=True)
            references = {'q-query': case, 'q-copy': {'image_sha256': 'pixels'},
                          'q-neighbor': {'image_sha256': 'neighbor'}, 'q-other': {'image_sha256': 'other'}}
            features = {q: np.array([[1., 0.]]) for q in references}
            flights = {'q-query': 'one', 'q-copy': 'two', 'q-neighbor': 'one', 'q-other': 'two'}
            ranked = rank_references(case, np.array([[1., 0.]]), references, features, flights)
            self.assertEqual([r['anchor'] for r in ranked], ['q-other'])
            descendants = {'q-other': {'lineage': [{'id': 'q-neighbor'}]}}
            self.assertEqual(rank_references(case, np.array([[1., 0.]]), references,
                                             features, flights, descendants), [])
            copies = {'q-other': {'lineage': [{'id': 'q-copy'}]}}
            ranked = rank_references(case, np.array([[1., 0.]]), references,
                                     {'q-other': features['q-other']}, anchors=copies)
            self.assertEqual(ranked, [])
            with self.assertRaisesRegex(ValueError, 'every query and reference'):
                rank_references(case, np.array([[1., 0.]]), references, features, {'q-query': 'one'})

    def test_masked_feature_cache_detects_changed_pixels_and_cached_arrays(self):
        class Pixels:
            def extract(self, image, mask):
                return pooled_descriptors(image.astype(np.float32), mask > 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.full((40, 60, 3), (100, 10, 0), np.uint8)
            image[:, :30] = [0, 0, 255]
            mask = np.full((40, 60), 255, np.uint8)
            mask[:, :30] = 0
            case = self.save_case(root, 'q-query', image, mask)
            cache = root / 'cache'
            cache.mkdir()
            with self.assertRaisesRegex(ValueError, 'preprocessed obstruction masks'):
                cached_features({**case, 'mask_status': 'unreviewed_all_pixels'}, root,
                                Pixels(), cache, {'model': 'pixels'})
            features = cached_features(case, root, Pixels(), cache, {'model': 'pixels'})
            self.assertTrue(np.all(features[:, 2] == 0), 'masked obstruction cannot enter retrieval')
            np.testing.assert_array_equal(features, cached_features(case, root, Pixels(), cache, {'model': 'pixels'}))
            array = next(cache.glob('*.npz'))
            array.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                cached_features(case, root, Pixels(), cache, {'model': 'pixels'})
            (root / case['image']).write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                cached_features(case, root, Pixels(), cache, {'model': 'pixels'})

    def test_planar_camera_family_keeps_distinct_focals_for_terrain_verification(self):
        x, y = np.meshgrid(np.linspace(-280, 280, 13), np.linspace(-180, 180, 11))
        points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        camera = np.array([[480., 0, 319.5], [0, 480., 239.5], [0, 0, 1.]])
        pixels = cv2.projectPoints(points, np.array([math.pi, 0, 0]), np.array([0., 0, 500.]), camera, None)[0].reshape(-1, 2)
        seeds = []
        for equivalent in [26, 52, 85]:
            intrinsics = camera.copy()
            intrinsics[0, 0] = intrinsics[1, 1] = equivalent * 800 / math.hypot(36, 24)
            solved = fit(points, pixels, intrinsics, (640, 480))
            self.assertIsNotNone(solved)
            self.assertTrue(solved[2]['geometry_accepted'], solved[2])
            seeds.append({'focal_35mm_mm': equivalent, 'score': solved[2]})
        self.assertEqual([s['focal_35mm_mm'] for s in focal_beam(seeds + seeds)], [26, 52, 85])

    def test_acquisition_matches_real_images_and_exports_a_fresh_render_checked_camera(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface = self.surface(root)
            z, x, y = json.loads((root / 'catalog.json').read_text())['tiles'][0]['xyz']
            lon = (x + .5) / 2**z * 360 - 180
            lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + .5) / 2**z))))
            eye = surface.local(lat, lon, 200)
            rotation = surface.camera_rotation(lat, lon, 0, 90)
            K = np.array([[480., 0, 319.5], [0, 480., 239.5], [0, 0, 1.]])
            size = (640, 480)
            source_catalog = json.loads((root / 'catalog.json').read_text())
            source_catalog['configuration']['bbox'] = [-75.2, 40.9, -74.9, 41.1]
            source_path = root / 'source-catalog.json'
            source_path.write_text(json.dumps(source_catalog))
            source = Surface(source_path)
            reference_eye = source.local(lat, lon, 200)
            reference_rotation = source.camera_rotation(lat, lon, 0, 90)
            reference, depth, _ = source.render(reference_eye, reference_rotation, K, size)
            query, query_depth, _ = surface.render(eye + [8, 5, 10], rotation, K, size)
            ref = self.save_case(root, 'q-reference', reference, (depth > 0).astype(np.uint8) * 255)
            case = self.save_case(root, 'q-query', query, (query_depth > 0).astype(np.uint8) * 255)
            pose, report = root / 'reference.npz', root / 'reference.json'
            np.savez_compressed(pose, eye=reference_eye, rotation=reference_rotation, K=K, size=size)
            report.write_text(json.dumps({'source': 'synthetic textured terrain camera'}))
            anchors = {'q-reference': {'path': pose, 'report': report, 'catalog_sha256': 'surface'}}
            directory = root / 'result'
            result = acquire(case, root, directory, {'q-reference': ref}, root, anchors,
                             [{'anchor': 'q-reference', 'similarity': 1.}], surface,
                             lambda key: source, ImageMatcher(), [20, 26, 35], 1, 0)
            self.assertEqual(result['status'], 'candidate_requires_review', result)
            self.assertFalse(result['query_metadata_used'])
            self.assertTrue(result['candidate']['fixed_render']['passed'])
            self.assertFalse(result['candidate']['fixed_render']['pose_refitted'])
            self.assertEqual(result['candidate']['state_sha256'], digest(directory / 'candidate.npz'))
            result_path = root / 'q-query.json'
            result_path.write_text(json.dumps(result))
            (root / 'run.json').write_text(json.dumps({'catalog_sha256': digest(root / 'catalog.json')}))
            (root / 'q-query').mkdir()
            (root / 'q-query/candidate.npz').write_bytes((directory / 'candidate.npz').read_bytes())
            library = load_anchors([root], digest(root / 'catalog.json'))
            self.assertEqual(library['q-query']['lineage'], [{'id': 'q-reference',
                'pose_sha256': digest(pose), 'report_sha256': digest(report)}])
            with np.load(directory / 'candidate.npz') as saved:
                output, _, _ = surface.render(saved['eye'], saved['rotation'], saved['K'], size)
            self.assertLess(float(np.mean(np.abs(output.astype(float) - query))), 8.)

    @staticmethod
    def save_case(root, identifier, image, mask):
        cv2.imwrite(str(root / (identifier + '.png')), image)
        cv2.imwrite(str(root / (identifier + '-mask.png')), mask)
        return {'id': identifier, 'image': identifier + '.png', 'valid_mask': identifier + '-mask.png',
                'image_sha256': digest(root / (identifier + '.png')),
                'mask_sha256': digest(root / (identifier + '-mask.png'))}

    @staticmethod
    def surface(root):
        z = 17
        x, y = tile(41, -75, z)
        rng = np.random.default_rng(7)
        texture = cv2.GaussianBlur(rng.integers(0, 256, (256, 256, 3), np.uint8), (3, 3), .5)
        tiles = []
        for kind, image in [('imagery', texture), ('dem', np.full((256, 256, 3), (0, 100, 128), np.uint8))]:
            path = root / (kind + '.png')
            cv2.imwrite(str(path), image)
            tiles.append({'kind': kind, 'xyz': [z, x, y], 'path': path.name, 'sha256': digest(path)})
        path = root / 'catalog.json'
        path.write_text(json.dumps({'complete': True,
            'configuration': {'bbox': [-75.1, 40.9, -74.9, 41.1]}, 'tiles': tiles}))
        return Surface(path)


if __name__ == '__main__':
    unittest.main()
