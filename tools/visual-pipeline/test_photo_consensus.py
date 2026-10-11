"""Association consensus cannot transform a displaced view into an aligned one."""
import unittest

import numpy as np

from photo_consensus import consensus_alignment
from photo_render_check import alignment


class ConsensusTests(unittest.TestCase):
    def pairs(self, offset):
        rng = np.random.default_rng(19)
        reference = rng.uniform([30, 30], [550, 400], (200, 2))
        good = np.stack([reference, reference + offset + rng.normal(0, .25, reference.shape)], axis=1)
        unrelated = rng.uniform([10, 10], [620, 460], (800, 2, 2))
        return np.concatenate([good, unrelated])

    def test_consensus_reports_alignment_without_relaxing_raw_acceptance(self):
        mask = np.ones((480, 640), np.uint8)
        report, _, _ = alignment(self.pairs([0, 0]), mask, mask)
        self.assertFalse(report['passed'])
        self.assertTrue(report['consensus']['aligned'], report)
        self.assertGreater(report['consensus']['held_pairs'], 6)
        self.assertLess(report['consensus']['held_median_px'], 1)

    def test_coherent_camera_displacement_retains_its_original_error(self):
        report = consensus_alignment(self.pairs([20, 12]), (640, 480))
        self.assertFalse(report['aligned'])
        self.assertGreater(report['pairs'], 180)
        self.assertGreater(report['held_median_px'], 20)
        self.assertFalse(report['selection_uses_identity_error'])
        self.assertFalse(report['pose_refitted'])
        self.assertEqual(report, consensus_alignment(self.pairs([20, 12]), (640, 480)))

    def test_dominant_displaced_associations_are_not_replaced_by_identity_subset(self):
        pairs = self.pairs([30, 0])[:200]
        identity = self.pairs([0, 0])[:40]
        report = consensus_alignment(np.concatenate([pairs, identity]), (640, 480))
        self.assertFalse(report['aligned'])
        self.assertGreater(report['median_px'], 25)

    def test_insufficient_or_collinear_evidence_cannot_pass(self):
        self.assertFalse(consensus_alignment([], (640, 480))['aligned'])
        points = np.column_stack([np.linspace(10, 600, 100), np.full(100, 200.)])
        self.assertFalse(consensus_alignment(np.stack([points, points], axis=1), (640, 480))['aligned'])


if __name__ == '__main__':
    unittest.main()
