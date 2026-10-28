"""The live prior seeds one acquisition pose from the registered frame nearest the video start
and writes the per-frame prior track inside the video span."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("live_prior", Path(__file__).with_name("live_prior.py"))
live_prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_prior)
capture_clock = live_prior.load_capture_clock()

ANCHOR = [40.5, -74.5]
REPLAY_UNIX_MS = 1_754_258_174_000


def atom(kind, body):
    return struct.pack(">I4s", 8 + len(body), kind) + body


def write_video(path, unix_ms, duration_s):
    seconds = unix_ms // 1000 + live_prior.SECONDS_1904_TO_1970
    mvhd = bytes(4) + struct.pack(">IIII", seconds, seconds, 1000, int(duration_s * 1000)) + bytes(80)
    path.write_bytes(atom(b"ftyp", b"mp42" + bytes(8)) + atom(b"moov", atom(b"mvhd", mvhd)))


def north_up_corners(lat, lon, half_m=20.0):
    dlat = half_m / 111_195
    dlon = half_m / (111_195 * math.cos(math.radians(lat)))
    return [[lon - dlon, lat + dlat], [lon + dlon, lat + dlat], [lon + dlon, lat - dlat], [lon - dlon, lat - dlat]]


def east_up_corners(lat, lon, half_m=20.0):
    north = north_up_corners(lat, lon, half_m)
    # The image's upward axis points east: rotate the corner order a quarter turn.
    return [north[1], north[2], north[3], north[0]]


def write_replay(root, frames):
    (root / "frames").mkdir()
    records = []
    for index, (time_s, position, corners) in enumerate(frames):
        name = f"frames/{index:06}.png"
        (root / name).write_bytes(b"\x89PNG")
        records.append({"time_s": time_s, "image": name, "corners": corners, "position": position,
                        "inliers": 0, "rms_px": None})
    document = {"version": 1, "name": "live", "duration_s": 10.0, "width": 640, "height": 360, "fps": 1,
                "agl_m": 110.0, "origin": [ANCHOR[1], ANCHOR[0]], "heading_deg": 0.0,
                "registration": "Flight-controller pose; nadir assumption; no visual registration",
                "time_anchor": {"at": 0, "unix_ms": REPLAY_UNIX_MS, "source_scale": "utc"},
                "frames": records, "mosaic": {"image": "live-mosaic.png", "corners": records[0]["corners"]}}
    (root / "replay.json").write_text(json.dumps(document))
    return document


class Workspace:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.map = self.root / "map.json"
        self.map.write_text(json.dumps({"pack_id": "a" * 64, "anchor_lat_lon": ANCHOR, "release_id": "r"}))
        self.replay = self.root / "live"
        self.replay.mkdir()

    def seed(self, video_unix_ms, offset_s=0.0, duration_s=100.0, max_gap_s=5.0):
        video = self.root / "DJI_0001.MP4"
        write_video(video, video_unix_ms, duration_s)
        return live_prior.seed(self.replay, video, self.map, offset_s, [640, 360], 82.1, 30.0, max_gap_s,
                               capture_clock)


class LivePriorTests(unittest.TestCase):
    def test_the_frame_nearest_the_offset_video_start_seeds_the_pose(self):
        space = Workspace()
        frames = [(float(t), [ANCHOR[1] + 0.0001 * t, ANCHOR[0]], north_up_corners(ANCHOR[0], ANCHOR[1] + 0.0001 * t))
                  for t in (0, 2, 4, 6)]
        frames[3] = (6.0, frames[3][1], east_up_corners(ANCHOR[0], frames[3][1][0]))
        write_replay(space.replay, frames)
        # The movie clock has one-second resolution; the video starts 4 s after the replay anchor.
        acquisition, summary = space.seed(REPLAY_UNIX_MS + 4000)
        self.assertEqual(summary["frame_time_s"], 4.0)
        self.assertEqual(summary["frames_inside_video"], 2, "the frames at 4 s and 6 s fall inside the 100 s video")
        self.assertEqual(summary["registered_frames"], 4)
        lon, lat = frames[2][1]
        self.assertEqual(acquisition["initial_user_prior"]["longitude"], lon)
        expected = live_prior.local_position(ANCHOR, lat, lon, 140.0)
        self.assertEqual(acquisition["pose"]["position"], expected)
        self.assertAlmostEqual(acquisition["pose"]["rotation"][2], 0.0, places=6)
        self.assertAlmostEqual(acquisition["pose"]["rotation"][3], 1.0, places=6)
        self.assertEqual(acquisition["map_manifest"], str(space.map.resolve()))
        self.assertEqual(acquisition["live_replay"]["offset_s"], 0.0)

        acquisition, summary = space.seed(REPLAY_UNIX_MS + 4000, offset_s=2.0)
        self.assertEqual(summary["frame_time_s"], 6.0)
        self.assertAlmostEqual(acquisition["live_replay"]["heading_deg"], 90.0, places=3)
        self.assertAlmostEqual(acquisition["pose"]["rotation"][2], math.sin(math.radians(-45)), places=6)
        self.assertEqual(summary["video_start_utc"], "2025-08-03T21:56:18+00:00")

    def test_the_camera_model_matches_the_debrief_calibration(self):
        camera = live_prior.camera_model(640, 360, 82.1)
        self.assertAlmostEqual(camera["focal"][0], 459.3371747579101, places=9)
        self.assertEqual(camera["principal"], [319.5, 179.5])
        self.assertEqual(camera["size"], [640, 360])

    def test_the_local_frame_is_metres_east_and_north_of_the_pack_anchor(self):
        east = live_prior.local_position(ANCHOR, ANCHOR[0], ANCHOR[1] + 0.001, 5.0)
        north = live_prior.local_position(ANCHOR, ANCHOR[0] + 0.001, ANCHOR[1], 5.0)
        self.assertAlmostEqual(east[0], 2 * math.pi * live_prior.RADIUS * math.cos(math.radians(ANCHOR[0])) / 360_000, places=6)
        self.assertLess(abs(east[1]), 1e-6)
        self.assertLess(abs(north[0]), 1e-6)
        self.assertAlmostEqual(north[1], 111.195, delta=0.3)
        self.assertEqual(east[2], 5.0)

    def test_missing_frames_clocks_and_distant_starts_fail_closed(self):
        space = Workspace()
        write_replay(space.replay, [(0.0, None, None), (1.0, None, None)])
        with self.assertRaises(ValueError):
            space.seed(REPLAY_UNIX_MS)

        space = Workspace()
        write_replay(space.replay, [(0.0, [ANCHOR[1], ANCHOR[0]], north_up_corners(*ANCHOR))])
        with self.assertRaises(ValueError):
            space.seed(REPLAY_UNIX_MS + 60_000)

        space = Workspace()
        document = write_replay(space.replay, [(0.0, [ANCHOR[1], ANCHOR[0]], north_up_corners(*ANCHOR))])
        document["time_anchor"]["source_scale"] = "media_pts"
        (space.replay / "replay.json").write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            space.seed(REPLAY_UNIX_MS)

        space = Workspace()
        write_replay(space.replay, [(0.0, [ANCHOR[1], ANCHOR[0]], north_up_corners(*ANCHOR))])
        (space.replay / "frames" / "000000.png").unlink()
        with self.assertRaises(ValueError):
            space.seed(REPLAY_UNIX_MS)

    def test_the_track_lists_registered_frames_inside_the_video_span_in_video_time(self):
        space = Workspace()
        # Frames at 0, 2, 4 and 6 s; the one at 6 s has its upward axis to the east. The frame
        # at 8 s is unregistered and never enters the track.
        frames = [(float(t), [ANCHOR[1] + 0.0001 * t, ANCHOR[0]], north_up_corners(ANCHOR[0], ANCHOR[1] + 0.0001 * t))
                  for t in (0, 2, 4, 6)]
        frames[3] = (6.0, frames[3][1], east_up_corners(ANCHOR[0], frames[3][1][0]))
        frames.append((8.0, None, None))
        write_replay(space.replay, frames)
        # The video starts 3 s after the anchor and lasts 2.5 s: only the frame at 4 s is
        # inside without an offset.
        acquisition, summary = space.seed(REPLAY_UNIX_MS + 3000, duration_s=2.5)
        track = acquisition["prior_track"]
        self.assertEqual(track["format"], "navigate-prior-track")
        self.assertEqual(track["version"], 1)
        self.assertEqual(summary["track_frames"], 1)
        self.assertEqual([entry["replay_time_s"] for entry in track["frames"]], [4.0])
        self.assertAlmostEqual(track["frames"][0]["video_time_s"], 1.0)
        self.assertEqual(track["video"]["start_unix_ms"], REPLAY_UNIX_MS + 3000)
        self.assertEqual(track["video"]["duration_s"], 2.5)
        self.assertEqual(track["video"]["offset_s"], 0.0)

        # An offset of 1 s moves the video start to 4 s; the frames at 4 s and 6 s are inside,
        # in video time order, each with its own heading and corners.
        acquisition, summary = space.seed(REPLAY_UNIX_MS + 3000, offset_s=1.0, duration_s=2.5)
        track = acquisition["prior_track"]
        self.assertEqual(summary["track_frames"], 2)
        self.assertEqual([entry["replay_time_s"] for entry in track["frames"]], [4.0, 6.0])
        self.assertEqual([round(entry["video_time_s"], 6) for entry in track["frames"]], [0.0, 2.0])
        self.assertAlmostEqual(track["frames"][0]["heading_deg"], 0.0, places=3)
        self.assertAlmostEqual(track["frames"][1]["heading_deg"], 90.0, places=3)
        self.assertEqual(track["frames"][1]["corners"], frames[3][2])
        self.assertEqual(track["frames"][1]["longitude"], frames[3][1][0])
        self.assertEqual(track["frames"][1]["agl_m"], 110.0)
        self.assertEqual(track["video"]["offset_s"], 1.0)
        self.assertTrue(all(entry["replay_time_s"] != 8.0 for entry in track["frames"]))

    def test_the_command_writes_the_track_beside_the_acquisition(self):
        space = Workspace()
        write_replay(space.replay, [(0.0, [ANCHOR[1], ANCHOR[0]], north_up_corners(*ANCHOR)),
                                    (2.0, [ANCHOR[1] + 0.0002, ANCHOR[0]], north_up_corners(ANCHOR[0], ANCHOR[1] + 0.0002))])
        video = space.root / "DJI_0003.MP4"
        write_video(video, REPLAY_UNIX_MS, 100.0)
        output = space.root / "out" / "acquisition.json"
        output.parent.mkdir()
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("live_prior.py")), str(space.replay), str(video),
             str(output), "--map", str(space.map), "--size", "640x360", "--terrain-m", "30"],
            capture_output=True, text=True, check=True)
        self.assertIn("2 per-frame priors inside the video span", result.stdout)
        track = json.loads((output.parent / "prior_track.json").read_text())
        acquisition = json.loads(output.read_text())
        self.assertEqual(track, acquisition["prior_track"])
        self.assertEqual(len(track["frames"]), 2)

    def test_a_video_without_a_movie_clock_is_refused(self):
        space = Workspace()
        write_replay(space.replay, [(0.0, [ANCHOR[1], ANCHOR[0]], north_up_corners(*ANCHOR))])
        video = space.root / "DJI_0002.MP4"
        video.write_bytes(atom(b"ftyp", b"mp42" + bytes(8)))
        with self.assertRaises(ValueError):
            live_prior.seed(space.replay, video, space.map, 0.0, [640, 360], 82.1, 30.0, 5.0, capture_clock)


if __name__ == "__main__":
    unittest.main()
