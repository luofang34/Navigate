"""Run one local inference process at a time and retain reproducible inputs."""

import json, math, os, signal, subprocess, sys, threading, uuid
from concurrent.futures import ThreadPoolExecutor
from .packages import materialize


class Jobs:
    def __init__(self, config, regions):
        self.config = config
        self.regions = regions
        self.records = {}
        self.stopping = threading.Event()
        self.process = None
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="visual-inference"
        )

    def start(self, value):
        if self.stopping.is_set():
            raise ValueError("Local inference service is stopping")
        region = self.regions[value["region_id"]]
        manifest = region["manifest"]
        if value["pack_id"] != manifest["pack_id"]:
            raise ValueError("Selected package is stale")
        upload = value["upload_id"]
        if len(upload) != 32 or not all(c in "0123456789abcdef" for c in upload):
            raise ValueError("Invalid upload ID")
        files = list((self.config.state / "uploads").glob(upload + ".*"))
        if len(files) != 1:
            raise ValueError("Upload not found")
        lat, lon, radius, agl, fov = [
            float(value[k])
            for k in ("latitude", "longitude", "radius_m", "agl_m", "fov_deg")
        ]
        west, south, east, north = region["bounds"]
        if (
            not all(math.isfinite(x) for x in (lat, lon, radius, agl, fov))
            or not west <= lon <= east
            or not south <= lat <= north
        ):
            raise ValueError("Prior center leaves prepared coverage")
        if not 20 <= radius <= 2000 or not 10 <= agl <= 2000 or not 10 < fov < 170:
            raise ValueError("Prior bounds are outside the demo range")
        period = float(value.get("sample_period", 1))
        max_frames = int(value.get("max_frames", 10))
        if not math.isfinite(period) or period < 0.1 or not 1 <= max_frames <= 120:
            raise ValueError("Invalid video sampling limits")
        job = uuid.uuid4().hex
        root = self.config.state / "jobs" / job
        root.mkdir()
        (root / "request.json").write_text(json.dumps(value, indent=2))
        with self.lock:
            self.records[job] = dict(
                id=job, status="queued", pack_id=manifest["pack_id"], frames=[]
            )
        self.executor.submit(
            self.run,
            job,
            root,
            files[0],
            manifest,
            (lat, lon, radius, agl, fov, period, max_frames),
        )
        return self.read(job)

    def run(self, job, root, source, manifest, prior):
        try:
            with self.lock:
                self.records[job]["status"] = "running"
            subprocess.run(
                [
                    str(self.config.binary),
                    "verify-pack",
                    str(self.config.state / "packs" / (manifest["pack_id"] + ".json")),
                    str(self.config.state),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            materialize(manifest, self.config.state, root / "map")
            lat, lon, radius, agl, fov, period, max_frames = prior
            command = [
                sys.executable,
                str(self.config.tool / "demo.py"),
                "--input",
                str(source),
                "--package",
                str(root / "map"),
                "--coordinate",
                f"{lat},{lon}",
                "--radius-m",
                str(radius),
                "--agl-m",
                str(agl),
                "--fov-deg",
                str(fov),
                "--sample-period",
                str(period),
                "--max-frames",
                str(max_frames),
                "--model-directory",
                str(self.config.models),
                "--binary",
                str(self.config.binary),
                "--device",
                self.config.device,
                "--output",
                str(root / "result"),
                "--no-view",
            ]
            with (root / "inference.log").open("w") as log:
                self.execute(command, log)
            view = json.loads((root / "result/view.json").read_text())
            with self.lock:
                self.records[job].update(
                    status="complete", view=view, frames=view["frames"]
                )
        except Exception as error:
            with self.lock:
                self.records[job].update(status="failed", error=str(error))
        finally:
            (root / "job.json").write_text(json.dumps(self.read(job), indent=2))

    def execute(self, command, log):
        with self.lock:
            if self.stopping.is_set():
                raise RuntimeError(
                    "Local inference service stopped before the job started"
                )
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            self.process = process
        try:
            status = process.wait(timeout=1800)
            if status:
                raise RuntimeError(
                    f"Visual inference process failed with status {status}; see inference.log"
                )
        finally:
            self.stop_process(process)
            with self.lock:
                self.process = None

    def stop_process(self, process):
        if process.poll() is not None:
            return
        sys.stderr.write(f"Stopping visual-inference process {process.pid}\n")
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sys.stderr.write(
                f"Aborting visual-inference process {process.pid} after timeout\n"
            )
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def close(self):
        self.stopping.set()
        with self.lock:
            process = self.process
        if process is not None:
            self.stop_process(process)
        self.executor.shutdown(wait=False, cancel_futures=True)

    def read(self, job):
        if len(job) != 32 or not all(c in "0123456789abcdef" for c in job):
            raise ValueError("Invalid job ID")
        with self.lock:
            if job not in self.records:
                root = self.config.state / "jobs" / job
                if (root / "job.json").is_file():
                    return json.loads((root / "job.json").read_text())
                if (root / "result/view.json").is_file():
                    view = json.loads((root / "result/view.json").read_text())
                    request = json.loads((root / "request.json").read_text())
                    return dict(
                        id=job,
                        status="complete",
                        pack_id=request["pack_id"],
                        view=view,
                        frames=view["frames"],
                    )
                raise ValueError("Job not found")
            record = dict(self.records[job])
        path = self.config.state / "jobs" / job / "result/estimates.json"
        if record["status"] == "running" and path.exists():
            try:
                record["frames"] = json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        return record
