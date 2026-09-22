"""Keep the renderer process resident and bind matches to each reference."""

import json, subprocess


class Worker:
    def __init__(self, binary, package, prior, log):
        self.log = log.open("x")
        try:
            self.process = subprocess.Popen(
                [str(binary), "worker", str(package), str(prior)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
            )
        except Exception:
            self.log.close()
            raise

    def request(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Renderer worker stopped; inspect worker.log")
        result = json.loads(line)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        self.process.stdin.close()
        try:
            status = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("Renderer worker did not stop within ten seconds")
        finally:
            self.process.stdout.close()
            self.log.close()
        if status:
            raise RuntimeError(f"Renderer worker failed with status {status}")
