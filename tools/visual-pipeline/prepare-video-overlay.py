#!/usr/bin/env python3
"""Build a local, approximate ground-registration replay from a nadir video.

Requires opencv-python-headless and numpy. The native viewer does not use Python.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import cv2
import numpy as np

from ground_mosaic import mosaic
from capture_clock import capture_anchor


def metadata(path):
    """Read only the QuickTime header; these tags describe the recording start."""
    with path.open("rb") as source:
        while header := source.read(8):
            size, kind = struct.unpack(">I4s", header)
            body = 8
            if size == 1:
                size = struct.unpack(">Q", source.read(8))[0]
                body = 16
            if size < body:
                break
            if kind == b"moov":
                if size > 64 * 1024 * 1024:
                    raise ValueError("QuickTime header is too large")
                data = source.read(size - body)
                result = {}
                for name in ("xyz", "fyw", "gyw", "gpt"):
                    at = data.find(b"\xa9" + name.encode())
                    if at >= 0 and at + 8 <= len(data):
                        count = struct.unpack(">H", data[at + 4:at + 6])[0]
                        result[name] = data[at + 8:at + 8 + count].decode().rstrip("\0")
                return result
            source.seek(size - body, 1)
    return {}


def registration(current, reference, matcher):
    current_points, current_descriptors = current
    reference_points, reference_descriptors = reference
    if current_descriptors is None or reference_descriptors is None:
        return None
    pairs = matcher.knnMatch(current_descriptors, reference_descriptors, k=2)
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.72 * b.distance]
    if len(good) < 30:
        return None
    a = np.float32([current_points[m.queryIdx].pt for m in good])
    b = np.float32([reference_points[m.trainIdx].pt for m in good])
    h, mask = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
    if h is None or mask is None:
        return None
    selected = mask.ravel().astype(bool)
    count = int(selected.sum())
    if count < 25 or count / len(good) < 0.45:
        return None
    span = np.ptp(a[selected], axis=0)
    if span[0] < 150 or span[1] < 80:
        return None
    projected = cv2.perspectiveTransform(a[selected, None, :], h)[:, 0, :]
    rms = float(np.sqrt(np.mean(np.sum((projected - b[selected]) ** 2, axis=1))))
    if rms > 2.0:
        return None
    return h, count, rms


def geographic(points, width, height, focal, agl, latitude, longitude, heading):
    x = (points[:, 0] - width / 2) * agl / focal
    y = -(points[:, 1] - height / 2) * agl / focal
    angle = math.radians(heading)
    east = x * math.cos(angle) + y * math.sin(angle)
    north = -x * math.sin(angle) + y * math.cos(angle)
    lat = latitude + np.degrees(north / 6378137)
    lon = longitude + np.degrees(east / (6378137 * math.cos(math.radians(latitude))))
    return np.column_stack((lon, lat))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--agl", type=float, required=True, help="Assumed height above local ground, metres")
    parser.add_argument("--sensor-fov", type=float, default=82.1, help="Diagonal FOV of the assumed 4:3 sensor")
    parser.add_argument("--fps", type=float, default=5)
    args = parser.parse_args()
    if not (1 <= args.agl <= 10_000 and 10 <= args.sensor_fov <= 170 and 0.1 <= args.fps <= 60):
        parser.error("AGL must be 1..10000 m, sensor FOV 10..170 degrees and sample rate 0.1..60 fps")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty")
    tags = metadata(args.video)
    import re
    match = re.fullmatch(r"([+-][0-9.]+)([+-][0-9.]+)", tags.get("xyz", ""))
    if not match:
        parser.error("video has no supported starting GPS position")
    latitude, longitude = map(float, match.groups())
    heading = (float(tags.get("fyw", 0)) + float(tags.get("gyw", 0))) % 360
    if abs(float(tags.get("gpt", 0)) + 90) > 15:
        parser.error("this planar experiment requires a near-nadir camera")
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if not capture.isOpened() or fps <= 0:
        parser.error("video decoder could not open the input")
    if args.fps > fps:
        parser.error("sample rate cannot exceed the video frame rate")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir()
    width, height = 640, round(640 * capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    focal = width * 1.25 / (2 * math.tan(math.radians(args.sensor_fov) / 2))
    detector = cv2.SIFT_create(nfeatures=1600)
    matcher = cv2.BFMatcher()
    frames, reference, reference_h, reference_time = [], None, np.eye(3), 0.0
    index, next_time = 0, 0.0
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height], [width / 2, height / 2]])
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        time = index / fps
        index += 1
        if time + 1e-6 < next_time:
            continue
        next_time += 1 / args.fps
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        features = detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        result = registration(features, reference, matcher) if reference is not None else (np.eye(3), 0, 0.0)
        h = None
        if result is not None and (reference is None or time - reference_time < 3):
            relative, inliers, rms = result
            proposed = reference_h @ relative
            transformed = cv2.perspectiveTransform(corners[:, None, :], proposed)[:, 0, :]
            area = abs(cv2.contourArea(transformed[:4]))
            if np.isfinite(transformed).all() and 0.15 * width * height < area < 8 * width * height:
                h = proposed
                reference, reference_h, reference_time = features, h, time
        file = f"frames/{len(frames):06}.png"
        cv2.imwrite(str(args.output / file), frame, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        record = {"time_s": time, "image": file, "corners": None, "position": None,
                  "inliers": 0, "rms_px": None}
        if h is not None:
            points = geographic(transformed, width, height, focal, args.agl, latitude, longitude, heading)
            record.update(corners=points[:4].tolist(), position=points[4].tolist(), inliers=inliers, rms_px=rms)
        frames.append(record)
        if len(frames) % 50 == 0:
            print(f"{time:.1f}s: {sum(f['corners'] is not None for f in frames)}/{len(frames)} registered", flush=True)
    capture.release()
    if not frames or frames[0]["corners"] is None:
        parser.error("no usable frames were decoded")
    with args.video.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    document = {"version": 1, "name": args.video.name, "duration_s": count / fps,
                "width": width, "height": height, "fps": args.fps, "agl_m": args.agl,
                "origin": [longitude, latitude], "heading_deg": heading,
                "source_sha256": digest, "time_anchor": capture_anchor(args.video),
                "registration": "Approximate planar visual registration. Initial GPS and heading; assumed AGL and FOV. Drift, terrain, roofs and trees are not corrected.",
                "frames": frames, "mosaic": mosaic(args.output, frames)}
    (args.output / "replay.json").write_text(json.dumps(document, indent=2))
    print(f"Prepared {len(frames)} frames; {sum(f['corners'] is not None for f in frames)} registered; {args.output / 'replay.json'}")


if __name__ == "__main__":
    main()
