import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webserver.service import Config, PackageService


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.web = self.root / 'web'
        self.web.mkdir()
        (self.web / 'index.html').write_text('browser worker app')
        self.app = PackageService(Config(self.root / 'state', self.web, 'https://navigate.example'))

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def request(self, path, method='GET', value=None, origin=None):
        body = json.dumps(value).encode() if value is not None else b''
        env = dict(PATH_INFO=path, REQUEST_METHOD=method, CONTENT_LENGTH=str(len(body)))
        env['wsgi.input'] = io.BytesIO(body)
        if origin:
            env['HTTP_ORIGIN'] = origin
        result = {}
        response = self.app(env, lambda status, headers: result.update(status=status, headers=dict(headers)))
        try:
            result['body'] = b''.join(response)
        finally:
            if hasattr(response, 'close'):
                response.close()
        return result

    def test_browser_service_has_no_native_inference_or_upload_path(self):
        self.assertEqual(self.request('/')['body'], b'browser worker app')
        self.assertFalse(json.loads(self.request('/api/health')['body'])['native_inference'])
        for path in ('/api/uploads', '/api/localize'):
            self.assertEqual(self.request(path, 'POST', {}, 'https://navigate.example')['status'], '404 Not Found')

    def test_deployed_origin_and_route_plan(self):
        selection = {'route': [[-74.47, 40.54], [-74.44, 40.55]], 'buffer_m': 1000, 'zoom': 16}
        result = self.request('/api/coverage-plan', 'POST', selection, 'https://navigate.example')
        self.assertEqual(result['status'], '200 OK')
        self.assertGreater(len(json.loads(result['body'])['imagery_tiles']), 0)
        self.assertEqual(self.request('/api/coverage-plan', 'POST', selection, 'https://other.example')['status'], '403 Forbidden')

    def test_streaming_and_path_boundaries(self):
        (self.root / 'secret.txt').write_text('not public')
        for path in ('/../secret.txt', '/%2e%2e/secret.txt', '/jobs/private.json'):
            self.assertEqual(self.request(path)['status'], '404 Not Found')
        payload = b'\0asm' + b'x' * 2_000_000
        (self.web / 'module.wasm').write_bytes(payload)
        response = self.request('/module.wasm')
        self.assertEqual(response['body'], payload)
        self.assertEqual(response['headers']['Content-Type'], 'application/wasm')

    def test_secure_origin_required_for_remote_webgpu(self):
        with self.assertRaises(ValueError):
            Config(self.root, self.web, 'http://navigate.example').validate()
