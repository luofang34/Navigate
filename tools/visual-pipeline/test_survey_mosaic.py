"""Exercise shared ground composition using two actual source images and camera poses."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from ground_mosaic import mosaic
from survey_mosaic import sources


class FlatGround:
    def ground(self, latitude, longitude):
        return np.stack((longitude * 1000, latitude * 1000, np.zeros_like(latitude)), axis=-1), np.ones_like(latitude, dtype=bool)


class MosaicTests(unittest.TestCase):
    def test_combined_ground_grid_preserves_both_flights_and_rejects_unsupported_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = {"size": [16, 16], "focal": [16, 16], "principal": [7.5, 7.5]}
            media = []
            for index, color in enumerate([(30, 60, 120), (180, 150, 90)]):
                folder = root / str(index)
                folder.mkdir()
                cv2.imwrite(str(folder / "frame.png"), np.full((16, 16, 3), color, np.uint8))
                (folder / "mosaic.png").write_bytes(b"individual product")
                configuration = {"camera":camera, "anchor":[0,0], "map_sha256":"terrain"}
                (folder / "configuration.json").write_text(json.dumps(configuration))
                x = index * 10
                corners = [[(x-5)/1000,.005],[(x+5)/1000,.005],[(x+5)/1000,-.005],[(x-5)/1000,-.005]]
                navigation = {"pose":{"position":[x,0,10],"rotation":[0,0,0,1]},
                              "camera_agl":[10,1],"support":{"ground_projection":True}}
                frame = {"image":"frame.png","corners":corners,"navigation":navigation}
                rejected = {**frame, "navigation":{**navigation,"support":{"ground_projection":False}}}
                (folder / "replay.json").write_text(json.dumps({"frames":[frame,rejected]}))
                media.append({"id":str(index),"replay":f"{index}/replay.json"})
            document = {"units":[{"id":"UAV-1","media":media}]}
            frames, references, configuration = sources(root, document)
            result = mosaic(root, frames, FlatGround(), camera=configuration["camera"],
                            image_name="survey-mosaic.png", max_size=64)
            image = cv2.imread(str(root / result["image"]), cv2.IMREAD_UNCHANGED)
            self.assertEqual(result["source_frames"], 2)
            self.assertEqual(result["rejected_frames"], 2)
            self.assertEqual(len(references), 2)
            np.testing.assert_array_equal(image[16,16], [30,60,120,255])
            np.testing.assert_array_equal(image[16,48], [180,150,90,255])
            for index in range(2):
                self.assertEqual((root / str(index) / "mosaic.png").read_bytes(), b"individual product")
            configuration["anchor"] = [1,0]
            (root / "1/configuration.json").write_text(json.dumps(configuration))
            with self.assertRaisesRegex(ValueError, "local frame"):
                sources(root, document)


if __name__ == "__main__":
    unittest.main()
