"""Decode sampled video frames with their actual presentation times."""

import json, subprocess
import cv2, numpy as np


def frames(path, camera, sample_period, max_frames):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is not None:
        yield 0, 0, resize(image, camera)
        return
    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,width,height",
        "-of",
        "json",
        str(path),
    ]
    probe = subprocess.run(args, capture_output=True, text=True, check=True)
    timeline = json.loads(probe.stdout)["frames"]
    if not timeline:
        raise ValueError("Input has no video frames")
    width, height = camera["width"], camera["height"]
    source = timeline[0]
    if abs(source["width"] / source["height"] - width / height) > 0.005:
        raise ValueError("Video aspect ratio does not match camera calibration")
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-noautorotate",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={width}:{height}",
            "-vsync",
            "0",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    previous = None
    start = None
    next_sample = 0
    count = 0
    try:
        for index, record in enumerate(timeline):
            timestamp = float(record["best_effort_timestamp_time"])
            if (
                not np.isfinite(timestamp)
                or previous is not None
                and timestamp <= previous
            ):
                raise ValueError("Video presentation times must increase")
            previous = timestamp
            if start is None:
                start = timestamp
            relative = timestamp - start
            raw = decoder.stdout.read(width * height)
            if len(raw) != width * height:
                raise ValueError("Video decoding ended before its timeline")
            if relative + 1e-6 < next_sample:
                continue
            yield (
                index,
                round(relative * 1e9),
                np.frombuffer(raw, np.uint8).reshape(height, width),
            )
            count += 1
            next_sample = relative + sample_period
            if max_frames is not None and count >= max_frames:
                return
        if decoder.stdout.read(1):
            raise ValueError("Video has frames beyond its timeline")
        if decoder.wait() != 0:
            raise RuntimeError("FFmpeg could not decode the complete input")
    finally:
        if decoder.poll() is None:
            decoder.terminate()
        try:
            decoder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            decoder.kill()
            decoder.wait()
        decoder.stdout.close()


def resize(image, camera):
    width, height = camera["width"], camera["height"]
    if abs(image.shape[1] / image.shape[0] - width / height) > 0.005:
        raise ValueError("Image aspect ratio does not match camera calibration")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
