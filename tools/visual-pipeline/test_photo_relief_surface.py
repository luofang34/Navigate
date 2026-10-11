"""Height corrections affect ray intersections without changing base tiles."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from photo_reference import tile
from photo_relief_surface import ReliefSurface, region_mask, save_correction
from photo_surface import Surface, ContextSurface
from photo_testset import digest


class ReliefSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.z = 17
        self.x, self.y = tile(41, -75, self.z)
        tiles = []
        for kind, color in [('imagery', (20, 80, 120)), ('dem', (0, 100, 128))]:
            path = self.root / (kind + '.png')
            image = np.full((256, 256, 3), color, np.uint8)
            if kind == 'imagery':
                image[96:160, 96:160] = (200, 200, 200)
            self.assertTrue(cv2.imwrite(str(path), image))
            tiles.append({'kind': kind, 'xyz': [self.z, self.x, self.y],
                          'path': path.name, 'sha256': digest(path)})
        self.catalog = self.root / 'catalog.json'
        self.catalog.write_text(json.dumps({'complete': True,
            'configuration': {'bbox': [-75.1, 40.9, -74.9, 41.1]}, 'tiles': tiles}))
        self.base = Surface(self.catalog)
        self.mask = np.full((64, 64), 255, np.uint8)
        self.path = save_correction(self.root, self.mask, [96, 96, 160, 160],
                                   [self.z, self.x, self.y], 15, digest(self.catalog))

    def geo(self, x, y):
        lon = (self.x + (np.asarray(x) + .5) / 256) / 2**self.z * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 *
            (self.y + (np.asarray(y) + .5) / 256) / 2**self.z))))
        return lat, lon

    def test_ray_cast_hits_raised_roof_and_base_stays_unchanged(self):
        surface = ReliefSurface(self.base, self.path, digest(self.catalog))
        lat, lon = self.geo(128, 128)
        eye = surface.local(lat, lon, 1000)
        rotation = surface.camera_rotation(lat, lon, 0, 90)
        K = np.array([[20., 0, 1], [0, 20., 1], [0, 0, 1]])
        _, depth, world = surface.render(eye, rotation, K, (3, 3))
        self.assertAlmostEqual(depth[1, 1], 885, delta=.02)
        self.assertAlmostEqual(float(surface.geodetic(world[1, 1])[2]), 115, delta=.02)
        self.assertEqual(float(self.base.sample('dem', lat, lon)[0]), 100)
        lat, lon = self.geo([20, 128, 220], [20, 128, 220])
        heights, covered = surface.sample('dem', lat, lon)
        np.testing.assert_array_equal(heights, [100, 115, 100])
        self.assertTrue(covered.all())
        np.testing.assert_array_equal(surface.sample('imagery', lat, lon)[0],
                                      self.base.sample('imagery', lat, lon)[0])

    def test_seeded_segmentation_finds_roof_without_ground(self):
        points = [self.base.local(*self.geo(x, y), 100)
                  for x in [108, 128, 148] for y in [108, 128, 148]]
        result = region_mask(self.base, points)
        self.assertIsNotNone(result)
        mask, (left, top, right, bottom), _ = result
        expected = np.zeros((256, 256), np.uint8)
        expected[96:160, 96:160] = 255
        np.testing.assert_array_equal(mask, expected[top:bottom, left:right])
        self.assertIsNone(region_mask(self.base, np.empty((0, 3))))

    def test_dem_edges_do_not_report_nonfinite_covered_heights(self):
        lat, lon = self.geo([0, 128, 255, 255, 256, -1], [0, 255, 128, 255, 128, 128])
        heights, covered = self.base.sample('dem', lat, lon)
        np.testing.assert_array_equal(covered, [True, True, True, True, False, False])
        np.testing.assert_array_equal(heights[covered], [100, 100, 100, 100])

    def test_context_fills_uncovered_rays_without_replacing_local_samples(self):
        tiles = []
        for kind, color in [('imagery', (200, 20, 10)), ('dem', (0, 200, 128))]:
            path = self.root / ('context-' + kind + '.png')
            self.assertTrue(cv2.imwrite(str(path), np.full((256, 256, 3), color, np.uint8)))
            for x in [self.x, self.x + 1]:
                tiles.append({'kind': kind, 'xyz': [self.z, x, self.y],
                              'path': path.name, 'sha256': digest(path)})
        catalog = self.root / 'context.json'
        catalog.write_text(json.dumps({**self.base.catalog, 'tiles': tiles}))
        surface = ContextSurface(self.base, Surface(catalog))
        lat, lon = self.geo([128, 384, 640], [128, 128, 128])
        heights, covered = surface.sample('dem', lat, lon)
        np.testing.assert_array_equal(covered, [True, True, False])
        np.testing.assert_array_equal(heights[:2], [100, 200])
        colors, _ = surface.sample('imagery', lat, lon)
        np.testing.assert_array_equal(colors[:2], [[200, 200, 200], [200, 20, 10]])
        eye = surface.local(lat[1], lon[1], 1000)
        rotation = surface.camera_rotation(lat[1], lon[1], 0, 90)
        K = np.array([[20., 0, 1], [0, 20., 1], [0, 0, 1]])
        image, depth, _ = surface.render(eye, rotation, K, (3, 3))
        self.assertAlmostEqual(depth[1, 1], 800, delta=.02)
        np.testing.assert_array_equal(image[1, 1], [200, 20, 10])

        corrected = ReliefSurface(self.base, self.path, digest(self.catalog))
        combinations = [ContextSurface(corrected, Surface(catalog)),
                        ReliefSurface(surface, self.path, digest(self.catalog))]
        for combined in combinations:
            with self.subTest(outer=type(combined).__name__):
                values, covered = combined.sample('dem', lat, lon)
                np.testing.assert_array_equal(covered, [True, True, False])
                np.testing.assert_array_equal(values[:2], [115, 200])
                for index, expected in [(0, 885), (1, 800)]:
                    eye = combined.local(lat[index], lon[index], 1000)
                    rotation = combined.camera_rotation(lat[index], lon[index], 0, 90)
                    _, depth, _ = combined.render(eye, rotation, K, (3, 3))
                    self.assertAlmostEqual(depth[1, 1], expected, delta=.02)
        np.testing.assert_array_equal(self.base.sample('dem', lat, lon)[0][:1], [100])

    def test_changed_mask_or_catalog_cannot_be_replayed(self):
        with self.assertRaisesRegex(ValueError, 'exact base catalog'):
            ReliefSurface(self.base, self.path, 'another catalog')
        self.mask[0, 0] = 0
        cv2.imwrite(str(self.root / 'relief-mask.png'), self.mask)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            ReliefSurface(self.base, self.path, digest(self.catalog))

    def test_invalid_windows_heights_and_paths_are_rejected(self):
        original = json.loads(self.path.read_text())
        for update in [{'window': [-1, 96, 63, 160]}, {'window': [96., 96, 160, 160]},
                       {'height_offset_m': float('nan')}, {'height_offset_m': 100},
                       {'xyz_origin': [17, 0, 0]}, {'mask': '../relief-mask.png'}]:
            with self.subTest(update=update):
                self.path.write_text(json.dumps({**original, **update}))
                with self.assertRaises(ValueError):
                    ReliefSurface(self.base, self.path, digest(self.catalog))


if __name__ == '__main__':
    unittest.main()
