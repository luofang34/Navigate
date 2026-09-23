"""Locate an image or sampled video from a rough prior for the WASM preview."""

import argparse, json, math, sys
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).parent / "matchers"))
from superglue import SuperGlueMatcher
from demo.map_data import MapData
from demo.search import RegionalSearch
from demo.media import frames
from demo.worker import Worker
from demo.prior import resolve
from demo.pipeline import process_frame, pose


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    location = p.add_mutually_exclusive_group(required=True)
    location.add_argument(
        "--prior",
        type=Path,
        help="Calibrated camera and independent rough pose bounds, in visual-bench prior format",
    )
    location.add_argument(
        "--coordinate",
        help="Rough latitude,longitude. Uses explicit altitude and camera assumptions.",
    )
    p.add_argument("--radius-m", type=float, default=500.0)
    p.add_argument("--agl-m", type=float, default=110.0)
    p.add_argument(
        "--fov-deg",
        type=float,
        default=82.1,
        help="Assumed 4:3 sensor diagonal FOV for --coordinate",
    )
    p.add_argument(
        "--width",
        type=int,
        default=640,
        help="Processing width; camera intrinsics scale with it",
    )
    p.add_argument("--model-directory", type=Path, required=True)
    p.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parent / "target/release/visual-bench",
    )
    p.add_argument("--output", type=Path, required=True, help="New run directory")
    p.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    p.add_argument(
        "--sample-period",
        type=float,
        default=1.0,
        help="Minimum seconds between video observations",
    )
    p.add_argument("--max-frames", type=int)
    p.add_argument("--no-view", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if not math.isfinite(a.sample_period) or a.sample_period <= 0:
        p.error("--sample-period must be positive")
    if a.max_frames is not None and a.max_frames < 1:
        p.error("--max-frames must be positive")
    if not 64 <= a.width <= 1920:
        p.error("--width must be between 64 and 1920")
    return a


def main():
    a = arguments()
    a.output = a.output.resolve()
    a.package = a.package.resolve()
    a.prior = a.prior.resolve() if a.prior else None
    a.binary = a.binary.resolve()
    map_data = MapData(a.package)
    config, prior_source = resolve(a, map_data)
    camera = config["camera"]
    prior = config["prior"]
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "prior.json").write_text(json.dumps(config, indent=2))
    (a.output / "prior-source.json").write_text(json.dumps(prior_source, indent=2))
    coarse = SuperGlueMatcher(a.model_directory, a.device, keypoints=1024)
    search = RegionalSearch(map_data, coarse, camera, prior)
    matcher = SuperGlueMatcher(a.model_directory, a.device, keypoints=2048)
    worker = Worker(
        a.binary, a.package, a.output / "prior.json", a.output / "worker.log"
    )
    results = []
    previous_candidates = []
    try:
        for sequence, stamp, image in frames(
            a.input, camera, a.sample_period, a.max_frames
        ):
            result = process_frame(
                worker,
                search,
                matcher,
                image,
                sequence,
                stamp,
                prior,
                camera,
                a.output,
                previous_candidates,
            )
            results.append(result)
            surviving = [
                pose(h) for h in result["candidate_hypotheses"] if h["accepted"]
            ]
            if surviving:
                previous_candidates = surviving
            (a.output / "estimates.json").write_text(json.dumps(results, indent=2))
            print(
                json.dumps(
                    {
                        key: result.get(key)
                        for key in [
                            "sequence",
                            "capture_time_ns",
                            "accepted",
                            "latitude_deg",
                            "longitude_deg",
                            "reason",
                            "pipeline_seconds",
                        ]
                    }
                ),
                flush=True,
            )
    finally:
        worker.close()
    viewer = dict(package=str(a.package), camera=camera, prior=prior, frames=results)
    manifest = a.output / "view.json"
    manifest.write_text(json.dumps(viewer, indent=2))
    (a.output / "run.json").write_text(
        json.dumps(
            dict(
                input=str(a.input.resolve()),
                prior_source=prior_source,
                device=a.device,
                sample_period=a.sample_period,
                max_frames=a.max_frames,
                frames=len(results),
                accepted=sum(r["accepted"] for r in results),
                scope="Separate visual observations. Shared evidence and correlations are not quantified. No inertial fusion or measured absolute accuracy.",
            ),
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "saved_view": str(manifest),
                "preview": "Use serve.py to open the WASM interface.",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
