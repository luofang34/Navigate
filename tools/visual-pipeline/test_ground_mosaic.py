"""Exercise real composition with unsupported projections and overlapping samples."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from ground_mosaic import mosaic

CAMERA = {"size": [64, 36], "focal": [50, 50], "principal": [31.5, 17.5]}


def frame(image, accepted=True, sigma=5):
    return {"image": image, "corners": [[-74.001, 40.001], [-74.0, 40.001], [-74., 40.], [-74.001, 40.]],
            "navigation": {"pose": {"position": [0, 0, 100], "rotation": [0, 0, 0, 1]},
                           "camera_agl": [100, sigma], "map_observation_accepted": accepted}}


class MosaicTests(unittest.TestCase):
    def test_grazing_and_tracking_only_frames_cannot_expand_map_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configuration.json").write_text(json.dumps({"camera": CAMERA}))
            cv2.imwrite(str(root / "good.png"), np.full((36, 64, 3), [20, 80, 180], np.uint8))
            good = frame("good.png")
            grazing = copy.deepcopy(good)
            angle = math.radians(80)/2
            grazing["navigation"]["pose"]["rotation"] = [math.sin(angle), 0, 0, math.cos(angle)]
            grazing["corners"] = [[-75, 41], [-74, 41], [-74, 40], [-75, 40]]
            conditional = copy.deepcopy(grazing)
            conditional["navigation"]["pose"]["rotation"] = [0, 0, 0, 1]
            conditional["navigation"]["map_observation_accepted"] = False
            result = mosaic(root, [good, grazing, conditional])
            self.assertEqual(result["corners"], good["corners"])
            self.assertEqual(result["source_frames"], 1)
            self.assertEqual(result["rejected_frames"], 2)
            self.assertGreater(result["coverage_fraction"], .99)

    def test_projection_support_controls_admission_even_for_map_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configuration.json").write_text(json.dumps({"camera": CAMERA}))
            cv2.imwrite(str(root / "good.png"), np.full((36, 64, 3), [20, 80, 180], np.uint8))
            good = frame("good.png", accepted=False)
            good["navigation"]["support"] = {"ground_projection": True}
            rejected = frame("good.png", accepted=True)
            rejected["navigation"]["support"] = {"ground_projection": False}
            rejected["corners"] = [[-75, 41], [-74, 41], [-74, 40], [-75, 40]]
            result = mosaic(root, [good, rejected])
            self.assertEqual(result["source_frames"], 1)
            self.assertEqual(result["corners"], good["corners"])

    def test_later_lower_quality_image_does_not_replace_better_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configuration.json").write_text(json.dumps({"camera": CAMERA}))
            cv2.imwrite(str(root / "good.png"), np.full((36, 64, 3), [20, 80, 180], np.uint8))
            cv2.imwrite(str(root / "weak.png"), np.full((36, 64, 3), [200, 10, 10], np.uint8))
            mosaic(root, [frame("good.png"), frame("weak.png", sigma=40)])
            image = cv2.imread(str(root / "mosaic.png"))
            np.testing.assert_array_equal(image[image.shape[0]//2, image.shape[1]//2], [20, 80, 180])

    def test_bare_earth_projection_places_ground_where_the_camera_sees_it(self):
        # A hill under one feature moves its ray intersection by metres. Four
        # corner intersections define one plane and cannot follow the hill.
        class Hill:
            east_per_degree = 111320 * math.cos(math.radians(40))
            north_per_degree = 110540

            def height(self, east, north):
                return 18 * np.exp(-((east - 72) ** 2 + (north - 62) ** 2) / 300)

            def ground(self, lat, lon):
                east = (np.asarray(lon) + 74) * self.east_per_degree
                north = (np.asarray(lat) - 40) * self.north_per_degree
                return np.stack((east, north, self.height(east, north)), axis=-1), np.ones(east.shape, bool)

            def geodetic(self, east, north):
                return [-74 + east / self.east_per_degree, 40 + north / self.north_per_degree]

        surface, eye = Hill(), np.array([40.0, 50.0, 100.0])

        def hit(u, v):
            ray = np.array([(u - 31.5) / 50, (17.5 - v) / 50, -1])
            near, far = 0.0, 200.0
            for _ in range(60):
                t = (near + far) / 2
                point = eye + t * ray
                near, far = (t, far) if point[2] > surface.height(*point[:2]) else (near, t)
            return eye + near * ray

        def place(surface_model):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "configuration.json").write_text(json.dumps({"camera": CAMERA}))
                picture = np.zeros((36, 64, 3), np.uint8)
                picture[10, 50] = 255
                cv2.imwrite(str(root / "dot.png"), picture)
                dot = frame("dot.png")
                dot["navigation"]["pose"]["position"] = eye.tolist()
                dot["corners"] = [surface.geodetic(*hit(u, v)[:2])
                                  for u, v in [[-.5, -.5], [63.5, -.5], [63.5, 35.5], [-.5, 35.5]]]
                result = mosaic(root, [dot], surface_model)
                image = cv2.imread(str(root / "mosaic.png"))
            rows, columns = np.nonzero(image[:, :, 0] > 128)
            (west, north), (east, _), (_, south) = result["corners"][:3]
            lon = west + (columns.mean() + .5) / image.shape[1] * (east - west)
            lat = north - (rows.mean() + .5) / image.shape[0] * (north - south)
            found = np.array([(lon + 74) * surface.east_per_degree, (lat - 40) * surface.north_per_degree])
            return np.linalg.norm(found - hit(50, 10)[:2]), result["composition"]

        error, composition = place(surface)
        self.assertLess(error, .3)
        self.assertIn("bare-earth", composition)
        corner_error, _ = place(None)
        self.assertGreater(corner_error, 3)

if __name__ == "__main__":
    unittest.main()
