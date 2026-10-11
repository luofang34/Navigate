"""Compose original supported camera frames on one shared ground model."""
import json
import math
from pathlib import Path

from ground_mosaic import mosaic, suitability
from navigation_reference import ReferenceMap


def sources(root, document):
    root = root.resolve()
    frames, references, configuration = [], [], None
    for unit in document["units"]:
        for media in unit["media"]:
            path = root / media["replay"]
            replay = json.loads(path.read_text())
            current = json.loads((path.parent / replay.get("configuration", "configuration.json")).read_text())
            if configuration is not None and any(current[key] != configuration[key]
                    for key in ("camera", "anchor", "map_sha256")):
                raise ValueError("survey mosaics require a common calibration, local frame and terrain package")
            configuration = current
            admitted = 0
            for frame in replay["frames"]:
                frame = {**frame, "image": str((path.parent / frame["image"]).resolve().relative_to(root))}
                frames.append(frame)
                admitted += suitability(frame, current["camera"]) is not None
            if admitted:
                references.append({"unit": unit["id"], "media": media["id"]})
    if not references:
        raise ValueError("no supported source frames for the survey mosaic")
    return frames, references, configuration


def compose(root, document, manifest):
    root = root.resolve()
    frames, references, configuration = sources(root, document)
    package = json.loads(manifest.read_text())
    if package["pack_id"] != configuration["map_sha256"] or package["anchor_lat_lon"] != configuration["anchor"]:
        raise ValueError("terrain package does not match the navigation coordinate frame")
    points = [point for frame in frames if suitability(frame, configuration["camera"]) is not None
              for point in frame["corners"]]
    west, east = min(p[0] for p in points), max(p[0] for p in points)
    south, north = min(p[1] for p in points), max(p[1] for p in points)
    lat, lon = (south+north)/2, (west+east)/2
    radius = max(north-south, (east-west)*math.cos(math.radians(lat))) * 111_320 / 2 + 500
    surface = ReferenceMap(manifest, [lat, lon], radius_m=radius)
    result = mosaic(root, frames, surface, camera=configuration["camera"],
                    image_name="survey-mosaic.png", max_size=4096)
    return {**result, "sources": references}
