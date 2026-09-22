"""Serve the browser regression and save its result on loopback only."""
import argparse
import json
import sys
from pathlib import Path
from wsgiref.simple_server import make_server
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webserver.service import Config, PackageService

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--port', type=int, default=0)
    args = p.parse_args()
    config = Config(args.state, Path(__file__).resolve().parents[1] / 'webapp', 'http://127.0.0.1', args.catalog)
    service = PackageService(config)
    def application(env, respond):
        if env['PATH_INFO'] == '/__test__/report' and env['REQUEST_METHOD'] == 'POST':
            count = int(env.get('CONTENT_LENGTH') or 0)
            if not 0 < count < 262144:
                respond('400 Bad Request', []); return [b'Invalid report']
            result = json.loads(env['wsgi.input'].read(count))
            args.result.write_text(json.dumps(result, indent=2))
            respond('200 OK', [('Content-Type','application/json')]); return [b'{}']
        return service(env, respond)
    with make_server('127.0.0.1', args.port, application) as server:
        object.__setattr__(config, 'public_origin', f'http://127.0.0.1:{server.server_port}')
        print(config.public_origin, flush=True)
        try:
            server.serve_forever()
        finally:
            service.close()

if __name__ == '__main__':
    main()
