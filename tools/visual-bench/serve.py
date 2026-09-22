"""Serve the offline WASM preview and a local visual inference worker."""

import argparse, json, mimetypes, os, sys, uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote

sys.dont_write_bytecode = True
from webserver.packages import build_region
from webserver.jobs import Jobs
from webserver.downloads import Downloads
from webserver.coverage import plan


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--catalog",
        type=Path,
        help="JSON list of {id,label,path} prepared source regions",
    )
    p.add_argument(
        "--state",
        type=Path,
        required=True,
        help="Persistent local packages, uploads and results",
    )
    p.add_argument("--models", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    p.add_argument("--port", type=int, default=0)
    a = p.parse_args()
    a.tool = Path(__file__).resolve().parent
    a.state = a.state.resolve()
    if bool(a.models) != bool(a.binary):
        p.error("Supply both --models and --binary for the optional native API")
    if a.models:
        a.models = a.models.resolve()
        a.binary = a.binary.resolve()
    return a


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write((fmt % args) + "\n")

    def send_bytes(self, body, kind="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def json(self, value, code=200):
        self.send_bytes(json.dumps(value).encode(), code=code)

    def do_GET(self):
        try:
            path = unquote(urlsplit(self.path).path)
            if path == "/api/catalog":
                return self.json(
                    [
                        {k: v for k, v in r.items() if k != "manifest"}
                        for r in list(self.server.regions.values())
                    ]
                )
            if path.startswith("/api/downloads/"):
                return self.json(self.server.downloads.read(path.split("/")[-1]))
            if path.startswith("/api/jobs/"):
                if self.server.jobs is None:
                    raise ValueError("Native inference API is disabled")
                return self.json(self.server.jobs.read(path.split("/")[-1]))
            prefix = path.split("/")[1]
            root = (
                self.server.config.state
                if prefix in ("chunks", "packs", "catalog", "jobs")
                else self.server.config.tool / "webapp"
            )
            relative = path.lstrip("/") or "index.html"
            file = (root / relative).resolve()
            if not file.is_relative_to(root.resolve()) or not file.is_file():
                return self.json({"error": "Not found"}, 404)
            if prefix == "jobs" and not (
                "/result/" in path and file.suffix in (".json", ".png")
            ):
                return self.json({"error": "Not found"}, 404)
            kind = (
                "application/wasm"
                if file.suffix == ".wasm"
                else mimetypes.guess_type(file)[0] or "application/octet-stream"
            )
            self.send_bytes(file.read_bytes(), kind)
        except (ValueError, OSError, KeyError, TypeError) as error:
            self.json({"error": str(error)}, 400)

    def do_POST(self):
        try:
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                return self.json({"error": "Cross-site write rejected"}, 403)
            origin = self.headers.get("Origin")
            expected = f"http://127.0.0.1:{self.server.server_port}"
            if origin and origin != expected:
                return self.json({"error": "Origin rejected"}, 403)
            size = int(self.headers.get("Content-Length", "0"))
            if self.path == "/api/uploads":
                if self.server.jobs is None:
                    raise ValueError("Native inference API is disabled")
                return self.upload(size)
            if not 0 < size <= 65536:
                raise ValueError("Invalid request size")
            value = json.loads(self.rfile.read(size))
            if self.path == "/api/offline-plan":
                return self.json(self.server.regions[value["region_id"]]["manifest"])
            if self.path == "/api/coverage-plan":
                return self.json(plan(value))
            if self.path == "/api/coverage-download":
                return self.json(self.server.downloads.start(value),202)
            if self.path == "/api/localize":
                if self.server.jobs is None:
                    raise ValueError("Native inference API is disabled")
                return self.json(self.server.jobs.start(value), 202)
            self.json({"error": "Not found"}, 404)
        except (ValueError, OSError, KeyError, TypeError) as error:
            self.json({"error": str(error)}, 400)

    def upload(self, size):
        if not 0 < size <= 256 * 1024 * 1024:
            raise ValueError("Input must be between 1 byte and 256 MB")
        suffix = Path(self.headers.get("X-File-Name", "")).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".mp4", ".mov", ".mkv", ".webm"):
            raise ValueError("Unsupported image or video type")
        name = uuid.uuid4().hex
        path = self.server.config.state / "uploads" / (name + suffix)
        try:
            with path.open("xb") as output:
                left = size
                while left:
                    block = self.rfile.read(min(left, 1024 * 1024))
                    if not block:
                        raise ValueError("Incomplete input upload")
                    output.write(block)
                    left -= len(block)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self.json({"upload_id": name, "size": size})


def main():
    a = arguments()
    for folder in ("chunks", "packs", "catalog", "uploads", "jobs"):
        (a.state / folder).mkdir(parents=True, exist_ok=True)
    regions = {}
    for value in (json.loads(a.catalog.read_text()) if a.catalog else []):
        key = value["id"]
        if not key or not all(c.isalnum() or c in "-_" for c in key):
            raise ValueError("Invalid region ID")
        regions[key] = build_region(Path(value["path"]), a.state, key, value["label"])
    for path in (a.state / "catalog").glob("naip-*.json"):
        region = json.loads(path.read_text())
        regions[region["id"]] = region
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    server.config = a
    server.regions = regions
    server.jobs = Jobs(a, regions) if a.models else None
    server.downloads = Downloads(a.state, regions)
    url = f"http://127.0.0.1:{server.server_port}"
    (a.state / "server.json").write_text(
        json.dumps(dict(url=url, pid=os.getpid()), indent=2)
    )
    print(url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if server.jobs is not None:
            server.jobs.close()
        server.downloads.close()


if __name__ == "__main__":
    main()
