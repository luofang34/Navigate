#!/usr/bin/env python3
"""Group prepared flights on one reported UTC clock, preserving real gaps and relative asset paths."""
import argparse
import json
from pathlib import Path

SIDC = "130401000012030000080100000000"


def compose(output, paths, name, unit):
    root = output.parent.resolve()
    inputs = []
    for path in paths:
        path = path.resolve()
        relative = path.relative_to(root)
        replay = json.loads(path.read_text())
        anchor = replay.get("time_anchor")
        if anchor is None or anchor.get("source_scale") not in ("utc", "unix"):
            raise ValueError(f"{path}: an explicit civil time anchor is required; no file-date inference")
        inputs.append((anchor["unix_ms"] - anchor["at"], relative, replay))
    origin = min(start for start, _, _ in inputs)
    activities, media = [], []
    for number, (start, path, replay) in enumerate(sorted(inputs, key=lambda item: item[0]), 1):
        start_ms = start - origin
        duration_ms = round(replay["duration_s"] * 1000)
        activity = f"flight-{number}"
        activities.append({"id": activity, "title": f"Flight {number}", "entity": unit,
                           "kind": "flight", "recorded": {"start": start_ms, "end": start_ms + duration_ms}})
        media.append({"id": f"{activity}-camera", "activity": activity, "label": replay["name"],
                      "replay": str(path), "start_s": start_ms / 1000})
    return {"format": "sokoly-debrief", "version": 1, "name": name, "activities": activities,
            "units": [{"id": unit, "sidc": SIDC, "media": media}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("replays", type=Path, nargs="+")
    parser.add_argument("--name", default="DJI survey")
    parser.add_argument("--unit", default="UAV-1")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--map", type=Path, help="Pinned Navigate terrain package for the combined survey mosaic")
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        parser.error("output already exists; use --replace to update the catalog")
    document = compose(args.output, args.replays, args.name, args.unit)
    if args.map is not None:
        from survey_mosaic import compose as mosaic
        document["mosaic"] = mosaic(args.output.parent, document, args.map)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Prepared {len(document['activities'])} flights on one reported UTC clock: {args.output}")


if __name__ == "__main__":
    main()
