"""The prior track serves the entry nearest a video frame as a map search pose at the
launch terrain height plus the recorded height above launch."""
import math
import unittest

import prior_track


class FlatMap:
    """A pack frame that keeps geodetic degrees as metres; enough to check the plumbing."""

    def local(self, lat, lon, altitude):
        return [lon, lat, altitude]


def acquisition(frames, terrain_m=120.0, **track_overrides):
    track = {"format": prior_track.FORMAT, "version": prior_track.VERSION, "frames": frames}
    track.update(track_overrides)
    return {"pose": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1]},
            "live_replay": {"terrain_m": terrain_m}, "prior_track": track}


def entry(video_time_s, lat=40.5, lon=-74.5, agl_m=110.0, heading_deg=0.0):
    return {"video_time_s": video_time_s, "latitude": lat, "longitude": lon,
            "agl_m": agl_m, "heading_deg": heading_deg}


class PriorTrackTests(unittest.TestCase):
    def test_a_seed_without_a_track_has_none(self):
        self.assertIsNone(prior_track.PriorTrack.from_acquisition({"pose": {}}))

    def test_another_format_or_version_is_refused(self):
        with self.assertRaises(ValueError):
            prior_track.PriorTrack.from_acquisition(acquisition([], format="other"))
        with self.assertRaises(ValueError):
            prior_track.PriorTrack.from_acquisition(acquisition([], version=2))

    def test_a_track_without_the_launch_terrain_is_refused(self):
        document = acquisition([entry(0.0)])
        del document["live_replay"]["terrain_m"]
        with self.assertRaises(ValueError):
            prior_track.PriorTrack.from_acquisition(document)

    def test_the_nearest_entry_serves_within_the_gap(self):
        track = prior_track.PriorTrack.from_acquisition(
            acquisition([entry(4.0, lat=40.504), entry(1.0, lat=40.501), entry(2.0, lat=40.502)]))
        self.assertEqual(len(track), 3)
        self.assertEqual(track.nearest(2.4, 5.0)["latitude"], 40.502)
        self.assertEqual(track.nearest(0.0, 5.0)["latitude"], 40.501)
        self.assertIsNone(track.nearest(9.5, 5.0))
        self.assertIsNone(track.nearest(2.4, 0.3))

    def test_the_pose_stands_at_launch_terrain_plus_height_above_launch(self):
        track = prior_track.PriorTrack.from_acquisition(
            acquisition([entry(3.0, lat=40.51, lon=-74.49, agl_m=110.0, heading_deg=90.0)], terrain_m=120.0))
        pose = track.pose(3.2, FlatMap(), 5.0)
        self.assertEqual(pose["position"], [-74.49, 40.51, 230.0])
        angle = math.radians(-90.0) / 2
        self.assertEqual(pose["rotation"], [0, 0, math.sin(angle), math.cos(angle)])
        self.assertIsNone(track.pose(30.0, FlatMap(), 5.0))

    def test_an_empty_track_serves_nothing(self):
        track = prior_track.PriorTrack.from_acquisition(acquisition([]))
        self.assertEqual(len(track), 0)
        self.assertIsNone(track.pose(0.0, FlatMap(), 5.0))


if __name__ == "__main__":
    unittest.main()
