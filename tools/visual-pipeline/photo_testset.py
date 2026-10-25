"""Prepare local, content-addressed photograph inputs and separate prior profiles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from photo_metadata import inspect, imagemagick

STILLS = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".tif", ".tiff"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(path, source, output, edge):
    sha = digest(path)
    identifier = "q-" + sha[:20]
    cached = output / "metadata" / (identifier + ".json")
    if cached.exists():
        entry = json.loads(cached.read_text())
        if entry["source_sha256"] != sha or entry["preprocess"]["long_edge"] != edge:
            raise ValueError(f"preparation settings differ for {path}")
        if digest(output / entry["image"]) != entry["image_sha256"]:
            raise ValueError(f"prepared image changed for {path}")
        if entry.get("metadata_version") != 2:
            entry.update(inspect(path))
            write_json(cached, entry)
        return entry
    metadata = inspect(path)
    image = f"images/{identifier}.png"
    subprocess.run([*imagemagick(), str(path), "-auto-orient", "-colorspace", "sRGB",
                    "-resize", f"{edge}x{edge}>", "-strip", "+set", "orientation",
                    "PNG24:" + str(output / image)], check=True, capture_output=True)
    picture = cv2.imread(str(output / image))
    if picture is None:
        raise ValueError(f"cannot decode prepared image {image}")
    height, width = picture.shape[:2]
    mask = f"masks/{identifier}.png"
    if not cv2.imwrite(str(output / mask), np.full((height, width), 255, np.uint8)):
        raise OSError(f"cannot write {mask}")
    thumbnail = cv2.resize(picture, (320, round(height * 320 / width)), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(output / "thumbnails" / (identifier + ".jpg")), thumbnail):
        raise OSError(f"cannot write thumbnail for {path}")
    companions = sorted(p for p in path.parent.iterdir()
                        if p.is_file() and p != path and p.stem.casefold() == path.stem.casefold())
    entry = {"id": identifier, "source": str(path.relative_to(source)),
             "source_sha256": sha, "image": image, "image_sha256": digest(output / image),
             "size": [width, height], "valid_mask": mask,
             "mask_status": "unreviewed_all_pixels", "mask_sha256": digest(output / mask), "companions": [
                 {"source": str(p.relative_to(source)), "sha256": digest(p)} for p in companions],
             "preprocess": {"version": 1, "long_edge": edge,
                            "orientation": "decoder_then_auto_orient", "color": "sRGB"},
             **metadata}
    write_json(cached, entry)
    return entry


def group_flights(entries, gap_minutes):
    def key(entry):
        timestamp = entry["claims"]["capture_time"]
        date = datetime.fromisoformat(timestamp) if timestamp else None
        clock = date.astimezone(timezone.utc).isoformat() if date and date.tzinfo else timestamp
        return (clock or "9999", entry["id"])
    flights = []
    previous = None
    for entry in sorted(entries, key=key):
        timestamp = entry["claims"]["capture_time"]
        date = datetime.fromisoformat(timestamp) if timestamp else None
        compatible = (previous is not None and date is not None
                      and (date.tzinfo is None) == (previous.tzinfo is None))
        difference = (date - previous).total_seconds() if compatible else None
        if difference is None or difference < 0 or difference > gap_minutes * 60:
            flights.append({"id": f"flight-{len(flights)+1:03}", "status": "provisional",
                            "basis": "EXIF time gap; requires route review", "photos": []})
        flights[-1]["photos"].append(entry["id"])
        previous = date
    return flights


def cases(entries, with_prior):
    result = []
    for entry in sorted(entries, key=lambda v: v["id"]):
        case = {"id": entry["id"], "image": entry["image"], "size": entry["size"],
                "valid_mask": entry["valid_mask"], "mask_status": entry["mask_status"],
                "image_sha256": entry["image_sha256"]}
        if entry.get("mask_sha256"):
            case["mask_sha256"] = entry["mask_sha256"]
        if entry.get("semantic_sha256"):
            case.update({key: entry[key] for key in ["semantic_labels", "semantic_sha256"]})
        if with_prior:
            case["prior"] = entry["claims"]
        result.append(case)
    return result


def contact_sheets(output, entries):
    ordered = sorted(entries, key=lambda e: (e["claims"]["capture_time"] or "9999", e["id"]))
    for start in range(0, len(ordered), 24):
        page = np.full((6 * 240, 4 * 320, 3), 24, np.uint8)
        for index, entry in enumerate(ordered[start:start+24]):
            tile = cv2.imread(str(output / "thumbnails" / (entry["id"] + ".jpg")))
            scale = min(320 / tile.shape[1], 190 / tile.shape[0])
            tile = cv2.resize(tile, (round(tile.shape[1] * scale), round(tile.shape[0] * scale)))
            x, y = index % 4 * 320, index // 4 * 240
            page[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
            for dy, text in [(205, entry["source"]), (225, entry["claims"]["capture_time"] or "time unknown")]:
                cv2.putText(page, text[:42], (x+5, y+dy), cv2.FONT_HERSHEY_SIMPLEX, .4, (240,240,240), 1)
        if not cv2.imwrite(str(output / f"contact-{start//24+1:02}.jpg"), page):
            raise OSError("cannot write contact sheet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--long-edge", type=int, default=1600)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--flight-gap-minutes", type=float, default=90)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source):
        parser.error("output must be outside the source tree")
    if args.long_edge < 64 or args.workers < 1 or args.flight_gap_minutes <= 0:
        parser.error("invalid image size, worker count or flight gap")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in STILLS)
    if not files:
        parser.error("no still images found")
    if (output / "manifest.json").exists():
        existing=json.loads((output / "manifest.json").read_text())
        if existing["source_root"]!=str(source):
            parser.error("output belongs to a different source directory")
    for folder in ["images", "masks", "metadata", "thumbnails", "profiles"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        entries = list(pool.map(lambda p: prepare(p, source, output, args.long_edge), files))
    unique = {entry["id"]: entry for entry in entries}
    if len(unique) != len(entries):
        raise ValueError("duplicate source content; resolve aliases before case generation")
    version = subprocess.run([*imagemagick(), "-version"], capture_output=True, text=True, check=True).stdout
    manifest = {"format": "photo-terrain-testset", "version": 1, "source_root": str(source),
                "tools": {"imagemagick": version.splitlines()[0], "opencv": cv2.__version__},
                "flight_gap_minutes": args.flight_gap_minutes,
                "flights": group_flights(entries, args.flight_gap_minutes), "photos": entries}
    captures = {}
    for entry in entries:
        key = (entry["claims"]["capture_time"] or entry["id"], entry["claims"]["camera_model"],
               Path(entry["source"]).parent.as_posix())
        captures.setdefault(key, []).append(entry["id"])
    manifest["capture_groups"] = [{"photos": ids, "basis": "source folder, camera and exposure time"}
                                  for ids in captures.values()]
    write_json(output / "manifest.json", manifest)
    for name, prior in [("with-prior", True), ("no-prior", False)]:
        write_json(output / "profiles" / (name + ".json"), {
            "format": "photo-terrain-cases", "version": 1, "root": "..",
            "profile": name, "reference_catalog": None, "cases": cases(entries, prior)})
    contact_sheets(output, entries)
    print(json.dumps({"photos": len(entries), "flights": len(manifest["flights"]),
                      "gps": sum(e["claims"]["position"] is not None for e in entries),
                      "output": str(output)}))


if __name__ == "__main__":
    main()
