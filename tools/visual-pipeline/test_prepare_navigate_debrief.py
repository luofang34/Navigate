"""The debrief's map search starts from the prior track's pose before the first fix, keeps the
session's own relocation candidates first afterwards, and sweeps heights only without a track."""
import argparse
import importlib.util
from pathlib import Path
import unittest

import prior_track

spec = importlib.util.spec_from_file_location(
    "prepare_navigate_debrief", Path(__file__).with_name("prepare-navigate-debrief.py"))
debrief = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debrief)


class FlatMap:
    def local(self, lat, lon, altitude):
        return [lon, lat, altitude]


class StubNavigation:
    def __init__(self, relocations):
        self.relocations = relocations

    def call(self, request, _prefix=""):
        assert request == {"relocate": True}, request
        return self.relocations


PRIOR = {"position": [0.0, 0.0, 200.0], "rotation": [0, 0, 0, 1]}
SESSION = [{"position": [5.0, 5.0, 205.0], "rotation": [0, 0, 0, 1]},
           {"position": [6.0, 6.0, 206.0], "rotation": [0, 0, 0, 1]}]
TRACKED = {"position": [-74.5, 40.5, 230.0], "rotation": [0, 0, 0.0, 1.0]}


def preparation(track, relocations, pose, candidate_limit=0):
    instance = debrief.Preparation.__new__(debrief.Preparation)
    instance.args = argparse.Namespace(prior_track_gap_s=5.0, candidate_limit=candidate_limit)
    instance.map = FlatMap()
    instance.prior = PRIOR
    instance.track = track
    instance.navigation = StubNavigation(relocations)
    instance.pose = pose
    return instance


def track():
    frames = [{"video_time_s": 7.0, "latitude": 40.5, "longitude": -74.5, "agl_m": 110.0, "heading_deg": 0.0}]
    return prior_track.PriorTrack(frames, 120.0)


def observation(time_s):
    return {"media_ns": round(time_s * 1e9)}


class HypothesesTests(unittest.TestCase):
    def test_without_a_track_the_seed_is_the_sweep_base(self):
        poses, base = preparation(None, None, None).hypotheses(observation(7.0))
        self.assertEqual(poses, [PRIOR])
        self.assertEqual(base, PRIOR)
        self.assertIsNot(base, PRIOR)

    def test_before_the_first_fix_the_track_pose_leads_and_the_height_is_known(self):
        poses, base = preparation(track(), None, None).hypotheses(observation(7.4))
        self.assertEqual(poses, [TRACKED])
        self.assertIsNone(base)

    def test_before_the_first_fix_the_track_pose_leads_the_session_candidates(self):
        poses, base = preparation(track(), SESSION, None).hypotheses(observation(7.4))
        self.assertEqual(poses, [TRACKED, *SESSION])
        self.assertEqual(base, SESSION[0])

    def test_once_located_the_session_candidates_lead_and_the_track_follows(self):
        poses, base = preparation(track(), SESSION, SESSION[0]).hypotheses(observation(7.4))
        self.assertEqual(poses, [*SESSION, TRACKED])
        self.assertEqual(base, SESSION[0])

    def test_the_candidate_limit_bounds_the_session_candidates_only(self):
        poses, _ = preparation(track(), SESSION, SESSION[0], candidate_limit=1).hypotheses(observation(7.4))
        self.assertEqual(poses, [SESSION[0], TRACKED])

    def test_a_frame_far_from_every_track_entry_falls_back_to_the_seed(self):
        poses, base = preparation(track(), None, None).hypotheses(observation(40.0))
        self.assertEqual(poses, [PRIOR])
        self.assertEqual(base, PRIOR)

    def test_the_terrain_sample_follows_the_track_pose_before_the_first_fix(self):
        instance = preparation(track(), None, None)
        self.assertEqual(instance.tracked_pose(7.0), TRACKED)
        self.assertIsNone(instance.tracked_pose(40.0))


if __name__ == "__main__":
    unittest.main()
