"""Rotation/scale retrieval only returns image evidence for geometric verification."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from navigation_retrieval import Retrieval


class RetrievalTests(unittest.TestCase):
    def test_changed_view_recovers_correspondences_and_blank_view_stays_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(21)
            image = np.zeros((360, 640), np.uint8)
            for _ in range(250):
                x, y = int(rng.integers(20, 620)), int(rng.integers(20, 340))
                cv2.circle(image, (x, y), int(rng.integers(3, 12)), int(rng.integers(20, 100)), -1)
            transform = cv2.getRotationMatrix2D((320, 180), 25, .8)
            query = cv2.warpAffine(image, transform, (640, 360))
            cv2.imwrite(str(root / 'reference.png'), image)
            cv2.imwrite(str(root / 'query.png'), query)
            cv2.imwrite(str(root / 'blank.png'), np.zeros_like(image))
            retrieval = Retrieval(root)
            retrieval.add(0, 'reference.png', {'position': [0, 0, 100], 'rotation': [0, 0, 0, 1]})
            candidates = retrieval.candidates('query.png')
            self.assertEqual(len(candidates), 1)
            pairs = np.asarray(candidates[0]['pairs'])
            self.assertGreaterEqual(len(pairs), 20)
            expected = pairs[:, 0] @ transform[:, :2].T + transform[:, 2]
            self.assertLess(np.median(np.linalg.norm(expected - pairs[:, 1], axis=1)), 1)
            self.assertEqual(retrieval.candidates('blank.png'), [])


if __name__ == '__main__':
    unittest.main()
