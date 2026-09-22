"""WSGI package service. Visual observations remain in the browser."""

import atexit
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit
from wsgiref.util import FileWrapper

from .coverage import plan
from .downloads import Downloads
from .packages import build_region


@dataclass(frozen=True)
class Config:
    state: Path
    webapp: Path
    public_origin: str
    catalog: Path | None = None

    def validate(self):
        origin = urlsplit(self.public_origin)
        if origin.scheme not in ('https', 'http') or not origin.netloc or origin.path:
            raise ValueError('Set a public origin with scheme and host only')
        if origin.scheme != 'https' and origin.hostname not in ('localhost', '127.0.0.1'):
            raise ValueError('A deployed browser app requires an HTTPS origin')


class PackageService:
    def __init__(self, config):
        config.validate()
        self.config = config
        for folder in ('chunks', 'packs', 'catalog'):
            (config.state / folder).mkdir(parents=True, exist_ok=True)
        self.regions = {}
        for item in json.loads(config.catalog.read_text()) if config.catalog else []:
            region = build_region(Path(item['path']), config.state, item['id'], item['label'])
            self.regions[region['id']] = region
        for file in (config.state / 'catalog').glob('naip-*.json'):
            region = json.loads(file.read_text())
            self.regions[region['id']] = region
        self.downloads = Downloads(config.state, self.regions)

    def close(self):
        self.downloads.close()

    def __call__(self, environ, start_response):
        try:
            status, headers, body = self.respond(environ)
        except (ValueError, KeyError, TypeError, OSError) as error:
            status, headers, body = self.json({'error': str(error)}, '400 Bad Request')
        start_response(status, headers + [('X-Content-Type-Options', 'nosniff')])
        return body

    def json(self, value, status='200 OK'):
        body = json.dumps(value).encode()
        return status, [('Content-Type', 'application/json'), ('Content-Length', str(len(body))), ('Cache-Control', 'no-store')], [body]

    def respond(self, environ):
        path = unquote(environ.get('PATH_INFO', '/'))
        method = environ['REQUEST_METHOD']
        if method == 'GET':
            if path == '/api/health':
                return self.json({'status': 'ready', 'processing': 'browser-worker', 'native_inference': False})
            if path == '/api/catalog':
                return self.json([{k: v for k, v in r.items() if k != 'manifest'} for r in list(self.regions.values())])
            if path.startswith('/api/downloads/'):
                return self.json(self.downloads.read(path.split('/')[-1]))
            return self.file(path)
        if method != 'POST':
            return self.json({'error': 'Method not allowed'}, '405 Method Not Allowed')
        if environ.get('HTTP_SEC_FETCH_SITE') == 'cross-site' or environ.get('HTTP_ORIGIN') not in (None, self.config.public_origin):
            return self.json({'error': 'Origin rejected'}, '403 Forbidden')
        length = int(environ.get('CONTENT_LENGTH') or 0)
        if not 0 < length <= 65536:
            raise ValueError('Invalid request size')
        body = environ['wsgi.input'].read(length)
        if len(body) != length:
            raise ValueError('Incomplete request')
        value = json.loads(body)
        if path == '/api/offline-plan':
            return self.json(self.regions[value['region_id']]['manifest'])
        if path == '/api/coverage-plan':
            return self.json(plan(value))
        if path == '/api/coverage-download':
            return self.json(self.downloads.start(value), '202 Accepted')
        return self.json({'error': 'Not found'}, '404 Not Found')

    def file(self, path):
        prefix = path.split('/')[1]
        if prefix in ('api', 'jobs', 'uploads'):
            return self.json({'error': 'Not found'}, '404 Not Found')
        root = self.config.state if prefix in ('chunks', 'packs', 'catalog') else self.config.webapp
        file = (root / (path.lstrip('/') or 'index.html')).resolve()
        if not file.is_relative_to(root.resolve()) or not file.is_file():
            return self.json({'error': 'Not found'}, '404 Not Found')
        kind = 'application/wasm' if file.suffix == '.wasm' else mimetypes.guess_type(file)[0] or 'application/octet-stream'
        headers = [('Content-Type', kind), ('Content-Length', str(file.stat().st_size)), ('Cache-Control', 'no-cache')]
        return '200 OK', headers, FileWrapper(file.open('rb'), 1024 * 1024)


def create_app():
    config = Config(
        state=Path(os.environ['NAVIGATE_DATA']).resolve(),
        webapp=Path(__file__).resolve().parents[1] / 'webapp',
        public_origin=os.environ['NAVIGATE_ORIGIN'],
        catalog=Path(os.environ['NAVIGATE_CATALOG']) if os.environ.get('NAVIGATE_CATALOG') else None,
    )
    app = PackageService(config)
    atexit.register(app.close)
    return app
