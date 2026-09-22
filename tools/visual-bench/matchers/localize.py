"""Render candidate views, match real images, and write checked camera estimates."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2

sys.dont_write_bytecode = True
from superglue import SuperGlueMatcher
from demo.geometry import match_reference


def run_command(command, log):
    with log.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=stream, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed with status {result.returncode}; see {log}")


def localize_record(record, directory, output, binary, matcher):
    name = Path(record["image"]).stem
    image_path = directory / record["image"]
    candidate = directory / record["candidate"]
    prior = directory / record["prior"]
    package = directory / record["package"]
    camera = json.loads(candidate.read_text())["camera"]
    source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise ValueError(f"Cannot decode {image_path}")
    query = cv2.resize(
        source, (camera["width"], camera["height"]), interpolation=cv2.INTER_AREA
    )
    query_path = output / f"{name}-query.png"
    if not cv2.imwrite(str(query_path), query):
        raise OSError(f"Cannot save {query_path}")
    reference = output / f"{name}-reference.png"
    started = time.monotonic()
    run_command(
        [str(binary), "render", str(package), str(candidate), str(reference)],
        output / f"{name}-render.log",
    )
    payload, metrics = match_reference(
        matcher, reference, query_path, reference.with_suffix(".depth.bin"), camera
    )
    matches_path = output / f"{name}-matches.json"
    matches_path.write_text(json.dumps(payload))
    (output / f"{name}-matches.metrics.json").write_text(json.dumps(metrics, indent=2))
    estimate = output / f"{name}-estimate.jsonl"
    run_command(
        [
            str(binary),
            "refine",
            str(package),
            str(query_path),
            "--prior",
            str(prior),
            "--reference-prior",
            str(candidate),
            "--matches",
            str(matches_path),
            "--output",
            str(estimate),
        ],
        output / f"{name}-refine.log",
    )
    result = json.loads(estimate.read_text())
    result.update(
        image=record["image"],
        matching=metrics,
        pipeline_seconds=time.monotonic() - started,
        camera_intrinsics=camera,
        candidate_source=record.get("candidate_source", "supplied"),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON records with image, package, prior and candidate paths relative to the manifest",
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    names = [Path(record["image"]).stem for record in manifest["records"]]
    if len(set(names)) != len(names):
        raise ValueError("Image stems must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    matcher = SuperGlueMatcher(args.model_directory, args.device)
    reports = []
    for record in manifest["records"]:
        reports.append(
            localize_record(
                record,
                args.manifest.parent,
                args.output,
                args.binary.resolve(),
                matcher,
            )
        )
        (args.output / "estimates.json").write_text(json.dumps(reports, indent=2))
    summary = {
        "images": len(reports),
        "accepted": sum(r["accepted"] for r in reports),
        "output": str(args.output),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
