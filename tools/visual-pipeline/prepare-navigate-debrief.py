#!/usr/bin/env python3
"""Prepare a Navigate-backed recording. Matching and navigation run in Rust.

Requires the example's OpenCV/numpy environment, a pinned Navigate map package,
its LoFTR model, and an ONNX Runtime shared library. Inputs remain local.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import cv2

from navigation_reference import ReferenceMap, rotation
from navigation_retrieval import Retrieval
from ground_mosaic import mosaic as compose_mosaic, suitability


class Worker:
    def __init__(self, args):
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, env={**os.environ, "ORT_DISABLE_TELEMETRY": "1"})

    def call(self, data, prefix=""):
        self.process.stdin.write(prefix + json.dumps(data, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(f"worker stopped: {self.process.wait()}")
        return json.loads(response)

    def close(self):
        self.process.stdin.close()
        try:
            result = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("worker exceeded shutdown timeout")
        if result:
            raise RuntimeError(f"worker failed: {result}")


class Preparation:
    def __init__(self, args):
        self.args = args
        self.source = json.loads(args.replay.read_text())
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=args.resume)
        for name in ["frames", "gray", "references"]:
            (self.output / name).mkdir(exist_ok=args.resume)
        acquisition = json.loads(args.acquisition.read_text()) if args.acquisition else None
        if acquisition:
            prior = acquisition["initial_user_prior"]
            lon, lat = prior["longitude"], prior["latitude"]
            if json.loads(Path(acquisition["map_manifest"]).read_text())["pack_id"] != json.loads(args.map.read_text())["pack_id"]:
                raise ValueError("acquisition and processing maps differ")
            (self.output / "acquisition.json").write_text(json.dumps(acquisition, indent=2))
        else:
            lon, lat = self.source["origin"]
        self.map = ReferenceMap(args.map, [lat, lon])
        self.camera = acquisition["camera"] if acquisition else self.calibration()
        position = self.map.local(lat, lon, 0)
        if acquisition:
            self.prior = acquisition["pose"]
        else:
            position[2] = self.map.terrain(position) + self.source["agl_m"]
            angle = math.radians(-self.source["heading_deg"]) / 2
            self.prior = {"position": position, "rotation": [0, 0, math.sin(angle), math.cos(angle)]}
        digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
        configuration = {"camera": self.camera, "anchor": self.map.anchor,
            "map_release": self.map.manifest["release_id"], "map_sha256": self.map.manifest["pack_id"],
            "prior": self.prior, "position_radius_m": 1500, "map_sigma_m": [3, 5], "object_height_m": 20,
            "capture_error_ns": 33366667,
            "clock_description": "Media-derived recording-local clock; absolute exposure time unavailable.",
            "matcher": f"Navigate LoFTR {digest} / CPU map matches; cpu-pyramidal-lk-v1 relative tracking"}
        configuration_file = self.output / "configuration.json"
        if configuration_file.exists() and json.loads(configuration_file.read_text()) != configuration:
            raise ValueError("resume configuration differs from the recorded configuration")
        configuration_file.write_text(json.dumps(configuration, indent=2))
        self.matcher = Worker([str(args.worker), "match-worker", str(args.model), str(args.runtime)])
        self.navigation = Worker([str(args.worker), "navigate-worker", str(self.output), str(self.output / "configuration.json")])
        self.retrieval = Retrieval(self.output) if args.night_retrieval else None
        self.previous = None
        self.pose = None
        self.search_pose = None
        self.frames = []
        self.last_map_check = -math.inf
        self.last_revision = 0

    def calibration(self):
        width, height = self.source["width"], self.source["height"]
        focal = width * 1.25 / (2 * math.tan(math.radians(self.args.sensor_fov) / 2))
        return {"size": [width, height], "focal": [focal, focal], "principal": [(width-1)/2, (height-1)/2]}

    def matches(self, reference, query, dense):
        result = self.matcher.call({"reference": str(self.output / reference),
                                    "query": str(self.output / query), "dense": dense})
        return result["pairs"]

    def reference(self, name, pose):
        return self.map.render(self.output, "references/" + name, self.camera, pose)[0]

    def acquire(self, observation):
        candidates = []
        if self.retrieval:
            for hit in self.retrieval.candidates(observation["image"]):
                reference = self.reference(f"retrieval-{observation['sequence']:06}-{hit['sequence']}", hit["pose"])
                proposal_input = {**observation, "map_reference": reference, "map_matches": hit["pairs"]}
                proposal = self.navigation.call(proposal_input, "propose ")
                if proposal:
                    candidates.append(proposal)
        ordinary = self.navigation.call({"relocate": True})
        base = copy.deepcopy(ordinary[0] if ordinary else self.prior)
        ordinary = ordinary or [base]
        if self.args.candidate_limit:
            ordinary = ordinary[:self.args.candidate_limit]
        candidates.extend(ordinary)
        if self.pose is None and not self.retrieval:
            for height in [150, 90, 210]:
                candidate = copy.deepcopy(base)
                candidate["position"][2] = self.map.terrain(candidate["position"]) + height
                candidates.append(candidate)
        for index, pose in enumerate(candidates):
            for refinement in range(2):
                reference = self.reference(f"map-{observation['sequence']:06}-{index}-{refinement}", pose)
                observation["map_reference"] = reference
                observation["map_matches"] = self.matches(reference["image"], observation["image"], True)
                accepted = self.navigation.call(observation, "assess ")
                if accepted:
                    return True
                pose = self.navigation.call(observation, "propose ")
                if pose is None:
                    break
        return False

    def revisits(self):
        """Evidence that the preceding frame sees an earlier keyframe again; Navigate verifies it."""
        for request in self.navigation.call({"revisits": True})[:1]:
            later, earlier = request["later"], request["earlier"]
            name = f"revisit-{later:06}-{earlier:06}"
            reference = self.reference(name, request["earlier_pose"])
            heading = [rotation(request[k]["rotation"])[:2, 0] for k in ("earlier_pose", "later_pose")]
            # The matcher is not rotation invariant; a return pass is turned half a circle.
            turned = float(heading[0] @ heading[1]) < 0
            query = f"gray/{later:06}.png"
            if turned:
                image = cv2.imread(str(self.output / query), cv2.IMREAD_GRAYSCALE)
                query = f"references/{name}-turned.png"
                cv2.imwrite(str(self.output / query), cv2.rotate(image, cv2.ROTATE_180))
            pairs = self.matches(f"gray/{earlier:06}.png", query, True)
            if turned:
                width, height = self.camera["size"]
                pairs = [[a, [width - 1 - b[0], height - 1 - b[1]]] for a, b in pairs]
            return [{"later": later, "earlier": earlier,
                     "earlier_capture_ns": round(self.source["frames"][earlier]["time_s"] * 1e9),
                     "earlier_image": f"gray/{earlier:06}.png", "earlier_reference": reference,
                     "matches": pairs}]
        return []

    def prepare(self, index, source):
        image = f"frames/{index:06}.png"
        original = self.args.replay.parent / source["image"]
        target = self.output / image
        picture = cv2.imread(str(original))
        if picture is None:
            raise ValueError(f"cannot decode {original}")
        if list(reversed(picture.shape[:2])) != self.camera["size"]:
            picture = cv2.resize(picture, self.camera["size"], interpolation=cv2.INTER_AREA)
            if target.exists() and not (cv2.imread(str(target)) == picture).all():
                raise ValueError("saved source frame changed")
            cv2.imwrite(str(target), picture, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        elif target.exists():
            if not os.path.samefile(original, target) and original.read_bytes() != target.read_bytes():
                raise ValueError("saved source frame changed")
        else:
            try:
                os.link(original, target)
            except OSError:
                shutil.copyfile(original, target)
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        gray_name = f"gray/{index:06}.png"
        cv2.imwrite(str(self.output / gray_name), gray)
        timestamp = round(source["time_s"] * 1e9)
        observation = {"sequence": index, "capture_ns": timestamp, "media_ns": timestamp,
            "image": gray_name, "map_reference": None, "map_matches": [],
            "tracking_reference": None, "tracking_matches": [],
            "terrain_m": self.map.terrain((self.pose or self.prior)["position"])}
        if self.previous is not None and self.pose is not None:
            observation["tracking_reference"] = self.reference(f"depth-{index:06}", self.pose)
            observation["tracking_matches"] = self.matches(self.previous, gray_name, False)
            if self.args.revisit_interval and index % self.args.revisit_interval == 0:
                observation["revisits"] = self.revisits()
        if source["time_s"] - self.last_map_check >= self.args.map_interval:
            self.acquire(observation)
            self.last_map_check = source["time_s"]
        self.previous = gray_name
        return image, observation

    def restore(self):
        if not self.args.resume:
            return 0
        events = [json.loads(line) for line in (self.output / "events.jsonl").read_text().splitlines()]
        observations = [json.loads(line) for line in (self.output / "observations.jsonl").read_text().splitlines()][:len(events)]
        for observation, event in zip(observations, events):
            result = self.navigation.call(observation)
            event["result"].setdefault("map_observation_accepted", result["map_observation_accepted"])
            if result != event["result"]:
                raise ValueError("recording reconstruction changed during resume")
            index = observation["sequence"]
            source = self.source["frames"][index]
            self.frames.append({"time_s": source["time_s"], "image": f"frames/{index:06}.png", "corners": None,
                                "position": None, "inliers": 0, "rms_px": None})
            self.previous = observation["image"]
            self.pose = result["pose"]
            if self.retrieval and result["map_observation_accepted"] and self.pose:
                self.retrieval.add(index, observation["image"], self.pose)
            self.search_pose = self.pose or self.search_pose
            self.last_revision = result["revision"]
            if observation["map_reference"] is not None:
                self.last_map_check = source["time_s"]
        for event in events:
            event["source_time_ns"] = event.pop("available_at_ns", event.get("source_time_ns"))
            event["view"] = "reconstructed"
        (self.output / "events.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))
        (self.output / "observations.jsonl").write_text("".join(json.dumps(o, separators=(",", ":")) + "\n" for o in observations))
        print(f"Resumed {len(observations)} verified observations", flush=True)
        return len(observations)

    def run(self):
        start = self.restore()
        with (self.output / "observations.jsonl").open("a") as evidence, (self.output / "events.jsonl").open("a") as events:
            for index, source in enumerate(self.source["frames"]):
                if index < start:
                    continue
                if self.args.duration and source["time_s"] >= self.args.duration:
                    break
                image, observation = self.prepare(index, source)
                evidence.write(json.dumps(observation, separators=(",", ":")) + "\n")
                result = self.navigation.call(observation)
                self.pose = result["pose"]
                if self.retrieval and result["map_observation_accepted"] and self.pose:
                    self.retrieval.add(index, observation["image"], self.pose)
                self.search_pose = self.pose or self.search_pose
                record = {"sequence": index, "source_time_ns": observation["media_ns"], "view": "reconstructed",
                          "kind": "visual_result", "result": result}
                if result["revision"] != self.last_revision:
                    record["trajectory"] = self.navigation.call({"finish": True})
                    self.last_revision = result["revision"]
                events.write(json.dumps(record, separators=(",", ":")) + "\n")
                events.flush()
                self.frames.append({"time_s": source["time_s"], "image": image, "corners": None,
                                    "position": None, "inliers": 0, "rms_px": None})
                if index % 25 == 0:
                    print(f"{source['time_s']:.1f}s: located={self.pose is not None}, revision={result['revision']}, {result['diagnostics']}", flush=True)
        results = self.navigation.call({"finish": True})
        for result in results:
            if result["pose"]:
                self.navigation.call([result["sequence"], self.map.terrain(result["pose"]["position"])], "terrain ")
        results = self.navigation.call({"finish": True})
        for frame, result in zip(self.frames, results):
            frame["navigation"] = result
            if result["pose"]:
                support = result["support"]
                lat, lon, _ = result["geodetic"]
                frame["position"] = [lon, lat] if support["navigation"] else None
                frame["corners"] = self.map.footprint(self.camera, result["pose"]) if support["ground_projection"] else None
                if suitability(frame, self.camera) is None:
                    frame["corners"] = None
        self.write()

    def write(self):
        registered = sum(frame["corners"] is not None for frame in self.frames)
        if registered:
            mosaic = compose_mosaic(self.output, self.frames, self.map)
        else:
            mosaic = None
        asset = copy.deepcopy(self.source.get("source"))
        if asset:
            original = self.args.replay.parent / asset["path"]
            asset["path"] = str(Path("originals") / original.name)
            target = self.output / asset["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not os.path.samefile(original, target):
                    with target.open("rb") as saved:
                        if hashlib.file_digest(saved, "sha256").hexdigest() != asset["sha256"]:
                            raise ValueError("saved media source changed")
            else:
                try:
                    os.link(original, target)
                except OSError:
                    shutil.copyfile(original, target)
        origin = next((frame["position"] for frame in self.frames if frame["position"]), None)
        document = {**self.source, "version": 3, "frames": self.frames, "mosaic": mosaic,
            "origin": origin, "agl_m": None, "source": asset,
            "duration_s": min(self.source["duration_s"], self.args.duration or math.inf),
            "registration": "Navigate map anchors, revisit closures between passes, independent two-view ground-normal constraints and conditional map-depth tracking. Evidence-gated revised camera trajectory, projected onto the bare-earth DEM; unknown lens calibration and mount, declared map-error budget. Projection over roofs and trees remains approximate.",
            "evidence": "observations.jsonl", "events": "events.jsonl", "configuration": "configuration.json",
            "map_attribution": self.map.manifest["attribution"]}
        (self.output / "replay.json").write_text(json.dumps(document, indent=2))
        archive = {"format": "sokoly-debrief", "version": 1, "name": f"{self.source['name']} · survey debrief",
            "activities": [{"id": "flight-1", "title": "Flight 1", "entity": "UAV-1", "kind": "flight",
                            "recorded": {"start": 0, "end": round(document["duration_s"]*1000)}}],
            "units": [{"id": "UAV-1", "sidc": "130401000012030000080100000000", "media": [
                {"id": "camera-1", "activity": "flight-1", "label": self.source["name"], "replay": "replay.json", "start_s": 0.0}]}]}
        (self.output / "debrief.json").write_text(json.dumps(archive, indent=2))
        print(f"Prepared {len(self.frames)} frames; {registered} projected; {self.output / 'debrief.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("map", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--acquisition", type=Path,
                        help="Verified initial search candidate, including camera and map identity; all frames still require session admission")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path(__file__).resolve().parents[2] / "target/release/navigate-recording-worker")
    parser.add_argument("--sensor-fov", type=float, default=82.1,
                        help="Diagonal FOV of an assumed 4:3 sensor, cropped vertically to the video")
    parser.add_argument("--night-retrieval", action="store_true",
                        help="Use anchored camera frames to propose retrieval poses; require independent map verification")
    parser.add_argument("--candidate-limit", type=int, default=0,
                        help="Limit ordinary map-render hypotheses per acquisition; 0 uses the library policy")
    parser.add_argument("--map-interval", type=float, default=5)
    parser.add_argument("--revisit-interval", type=int, default=5,
                        help="Frames between revisit checks of the preceding frame; 0 disables them")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.sensor_fov <= 170 or args.map_interval <= 0 or args.revisit_interval < 0 or args.candidate_limit < 0 \
            or (args.duration is not None and args.duration <= 0):
        parser.error("invalid FOV, interval or duration")
    preparation = Preparation(args)
    try:
        preparation.run()
    finally:
        preparation.navigation.close()
        preparation.matcher.close()


if __name__ == "__main__":
    main()
