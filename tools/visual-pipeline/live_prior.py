#!/usr/bin/env python3
"""Seed a Navigate acquisition from a Sokoly live survey replay.

The live survey records each video frame it registered from the flight
controller's pose. This script finds the registered frame nearest to the
start of the original SD-card video and writes `acquisition.json` in the
form `prepare-navigate-debrief.py --acquisition` reads: an initial position
prior, the camera model, a search pose in the map pack frame, and the map
manifest.

The camera clock and the host clock are different clocks. `--offset-s` is
the correlation the operator confirms; the script does not infer it. The
seed is one initial pose for retrieval.

The script also writes the per-frame prior track: every registered live
frame whose time falls inside the video span after the offset, in video
time order, with its position, heading and corners, and the survey's one
recorded height above launch (the live recorder keeps a single height; a
flight that changes altitude needs a per-frame height there). The track goes
to `--track` and, under the `prior_track` key, into `acquisition.json`,
where `prepare-navigate-debrief.py` reads the entry nearest each video frame
as that frame's map search pose.
"""

import argparse
import datetime
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

from prior_track import rotation_from_heading

RADIUS = 6371008.8
SECONDS_1904_TO_1970 = 2082844800


def load_capture_clock():
    spec = importlib.util.spec_from_file_location(
        "capture_clock", Path(__file__).with_name("capture_clock.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mercator(lat, lon):
    return ((lon + 180) / 360, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2)


def local_position(anchor, lat, lon, altitude):
    """East, north and altitude in the pack frame, as the reference renderer defines them."""
    scale = 2 * math.pi * RADIUS * math.cos(math.radians(anchor[0]))
    ox, oy = mercator(*anchor)
    x, y = mercator(lat, lon)
    return [(x - ox) * scale, -(y - oy) * scale, float(altitude)]


def heading_from_corners(anchor, corners):
    """Heading of the image's upward axis, degrees clockwise from north."""
    top_left = local_position(anchor, corners[0][1], corners[0][0], 0)
    bottom_left = local_position(anchor, corners[3][1], corners[3][0], 0)
    east = top_left[0] - bottom_left[0]
    north = top_left[1] - bottom_left[1]
    if math.hypot(east, north) < 1e-6:
        return None
    return math.degrees(math.atan2(east, north)) % 360


def camera_model(width, height, sensor_fov_deg):
    focal = width * 1.25 / (2 * math.tan(math.radians(sensor_fov_deg) / 2))
    return {"size": [width, height], "focal": [focal, focal],
            "principal": [(width - 1) / 2, (height - 1) / 2]}


def movie_duration_s(video, atoms):
    """Movie duration from the QuickTime header, seconds; `None` when absent."""
    with video.open("rb") as stream:
        data = stream.read()
    for kind, body in atoms(data):
        if kind != b"moov":
            continue
        for inner, header in atoms(body):
            if inner != b"mvhd" or len(header) < 12:
                continue
            if header[0] == 1:
                timescale, duration = struct.unpack_from(">IQ", header, 20)
            else:
                timescale, duration = struct.unpack_from(">II", header, 12)
            return duration / timescale if timescale else None
    return None


def replay_unix_ms(document, time_s):
    anchor = document.get("time_anchor")
    if not anchor or anchor.get("source_scale") not in ("utc", "unix"):
        raise ValueError("live replay needs a utc or unix time anchor")
    return anchor["unix_ms"] + time_s * 1000 - anchor["at"]


def registered_frames(replay_dir, document):
    frames = []
    for frame in document["frames"]:
        if frame.get("corners") is None or frame.get("position") is None:
            continue
        image = Path(frame["image"])
        if image.is_absolute() or ".." in image.parts:
            raise ValueError(f"frame image escapes the live replay: {frame['image']}")
        if not (replay_dir / image).is_file():
            raise ValueError(f"live frame is missing: {frame['image']}")
        frames.append(frame)
    if not frames:
        raise ValueError("the live replay has no registered frame")
    return frames


def seed(replay_dir, video, map_manifest, offset_s, size, sensor_fov_deg, terrain_m, max_gap_s,
         capture_clock):
    """Build the acquisition and its correlation summary."""
    document = json.loads((replay_dir / "replay.json").read_text())
    if document.get("version") != 1:
        raise ValueError("live replay must be version 1")
    manifest = json.loads(map_manifest.read_text())
    anchor = manifest["anchor_lat_lon"]
    video_anchor = capture_clock.capture_anchor(video)
    if video_anchor is None:
        raise ValueError("the video has no nonzero movie clock")
    video_start_ms = video_anchor["unix_ms"] + offset_s * 1000
    frames = registered_frames(replay_dir, document)
    times = [(abs(replay_unix_ms(document, f["time_s"]) - video_start_ms), f) for f in frames]
    gap_ms, frame = min(times, key=lambda item: item[0])
    if gap_ms > max_gap_s * 1000:
        raise ValueError(f"nearest registered live frame is {gap_ms / 1000:.1f} s from the video start")
    duration_s = movie_duration_s(video, capture_clock.atoms)
    inside = sum(1 for f in frames
                 if duration_s is not None
                 and 0 <= replay_unix_ms(document, f["time_s"]) - video_start_ms <= duration_s * 1000)
    lon, lat = frame["position"]
    heading = heading_from_corners(anchor, frame["corners"])
    if heading is None:
        heading = document["heading_deg"]
    position = local_position(anchor, lat, lon, terrain_m + document["agl_m"])
    track = build_track(document, frames, anchor, video_start_ms, duration_s, offset_s)
    acquisition = {
        "initial_user_prior": {
            "longitude": lon, "latitude": lat,
            "source": "Sokoly live survey: flight-controller position of the registered frame "
                      f"nearest the video start; frame {frame['image']} at {frame['time_s']} s; "
                      f"camera clock offset {offset_s} s confirmed by the operator",
        },
        "camera": camera_model(size[0], size[1], sensor_fov_deg),
        "pose": {"position": position, "rotation": rotation_from_heading(heading)},
        "map_manifest": str(map_manifest.resolve()),
        "live_replay": {
            "directory": str(replay_dir.resolve()), "frame": frame["image"],
            "frame_time_s": frame["time_s"], "replay_anchor_unix_ms": document["time_anchor"]["unix_ms"],
            "video_start_unix_ms": video_start_ms, "offset_s": offset_s,
            "terrain_m": terrain_m, "height_above_launch_m": document["agl_m"],
            "heading_deg": heading,
        },
        "prior_track": track,
    }
    summary = {
        "video_start_utc": iso(video_start_ms - offset_s * 1000),
        "replay_anchor_utc": iso(document["time_anchor"]["unix_ms"]),
        "offset_s": offset_s,
        "frame_time_s": frame["time_s"], "frame_position": [lon, lat],
        "gap_s": gap_ms / 1000, "video_duration_s": duration_s,
        "frames_inside_video": inside, "registered_frames": len(frames),
        "track_frames": len(track["frames"]),
    }
    return acquisition, summary


def build_track(document, frames, anchor, video_start_ms, duration_s, offset_s):
    """The per-frame prior track inside the video span, in video time order."""
    entries = []
    for frame in frames:
        video_time_s = (replay_unix_ms(document, frame["time_s"]) - video_start_ms) / 1000
        if duration_s is None or not 0 <= video_time_s <= duration_s:
            continue
        heading = heading_from_corners(anchor, frame["corners"])
        if heading is None:
            heading = document["heading_deg"]
        lon, lat = frame["position"]
        entries.append({
            "video_time_s": video_time_s, "replay_time_s": frame["time_s"],
            "longitude": lon, "latitude": lat, "agl_m": document["agl_m"],
            "heading_deg": heading, "corners": frame["corners"],
        })
    entries.sort(key=lambda entry: entry["video_time_s"])
    return {
        "format": "navigate-prior-track", "version": 1,
        "video": {"start_unix_ms": video_start_ms, "duration_s": duration_s, "offset_s": offset_s,
                  "replay_anchor_unix_ms": document["time_anchor"]["unix_ms"]},
        "source": "Sokoly live survey: flight-controller pose per registered frame; nadir assumption, "
                  "no visual registration; camera clock offset confirmed by the operator",
        "frames": entries,
    }


def iso(unix_ms):
    return datetime.datetime.fromtimestamp(unix_ms / 1000, datetime.timezone.utc).isoformat()


def parse_size(value):
    width, height = value.lower().split("x")
    return [int(width), int(height)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("live_replay", type=Path, help="Directory with the live survey replay.json")
    parser.add_argument("video", type=Path, help="The original SD-card recording")
    parser.add_argument("output", type=Path, help="Where to write acquisition.json")
    parser.add_argument("--map", type=Path, required=True, help="Navigate pack manifest")
    parser.add_argument("--offset-s", type=float, default=0.0,
                        help="Host clock minus camera clock, seconds; added to the movie timestamp")
    parser.add_argument("--size", type=parse_size, required=True,
                        help="Width x height of the frames the debrief processes, for example 640x360")
    parser.add_argument("--sensor-fov", type=float, default=82.1,
                        help="Diagonal field of view of the 4:3 sensor, degrees")
    parser.add_argument("--terrain-m", type=float, required=True,
                        help="Bare-earth elevation at the launch point in the pack datum, metres")
    parser.add_argument("--max-gap-s", type=float, default=5.0,
                        help="Largest accepted distance between the video start and a live frame")
    parser.add_argument("--track", type=Path,
                        help="Where to write the per-frame prior track; default prior_track.json beside the output")
    args = parser.parse_args()
    try:
        acquisition, summary = seed(args.live_replay, args.video, args.map, args.offset_s, args.size,
                                    args.sensor_fov, args.terrain_m, args.max_gap_s, load_capture_clock())
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(acquisition, indent=2) + "\n")
    track_path = args.track or args.output.with_name("prior_track.json")
    track_path.write_text(json.dumps(acquisition["prior_track"], indent=2) + "\n")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote {args.output}; one initial pose for retrieval")
    print(f"wrote {track_path}; {summary['track_frames']} per-frame priors inside the video span")


if __name__ == "__main__":
    sys.exit(main())
