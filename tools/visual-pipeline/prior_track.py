"""The per-frame prior track `live_prior.py` writes inside `acquisition.json`.

Each entry is the flight controller's pose for one registered live frame,
placed on the SD-card video's time axis. The debrief reads the entry
nearest a video frame as a map search pose: a position in the pack frame at
the operator's launch terrain height plus the recorded height above launch
(one value for the survey, as the live recorder writes it), and a rotation
from the recorded heading. The track is a prior; every frame still needs
its own map evidence.
"""
import math

FORMAT = "navigate-prior-track"
VERSION = 1


def rotation_from_heading(heading_deg):
    """A nadir camera whose image-up axis points `heading_deg` clockwise from north."""
    angle = math.radians(-heading_deg) / 2
    return [0, 0, math.sin(angle), math.cos(angle)]


class PriorTrack:
    def __init__(self, frames, terrain_m):
        self.frames = sorted(frames, key=lambda entry: entry["video_time_s"])
        self.terrain_m = float(terrain_m)

    @classmethod
    def from_acquisition(cls, acquisition):
        """The track in `acquisition`, or `None` when the seed carries none.

        Raises `ValueError` on a track of another format or version, or one
        without the launch terrain height its heights are measured from.
        """
        track = acquisition.get("prior_track")
        if track is None:
            return None
        if track.get("format") != FORMAT or track.get("version") != VERSION:
            raise ValueError("prior track is not a version 1 navigate-prior-track")
        terrain_m = acquisition.get("live_replay", {}).get("terrain_m")
        if terrain_m is None:
            raise ValueError("prior track needs the live replay's launch terrain height")
        return cls(track.get("frames", []), terrain_m)

    def __len__(self):
        return len(self.frames)

    def nearest(self, time_s, max_gap_s):
        """The entry nearest `time_s` on the video axis, or `None` beyond `max_gap_s`."""
        if not self.frames:
            return None
        entry = min(self.frames, key=lambda entry: abs(entry["video_time_s"] - time_s))
        if abs(entry["video_time_s"] - time_s) > max_gap_s:
            return None
        return entry

    def pose(self, time_s, reference_map, max_gap_s):
        """The search pose for the video frame at `time_s`, or `None` without a near entry."""
        entry = self.nearest(time_s, max_gap_s)
        if entry is None:
            return None
        position = reference_map.local(entry["latitude"], entry["longitude"],
                                       self.terrain_m + entry["agl_m"])
        return {"position": position, "rotation": rotation_from_heading(entry["heading_deg"])}
