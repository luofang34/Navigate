import json,sys,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serve import Handler


class BrowserServiceTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.config=SimpleNamespace(state=Path(self.folder.name),tool=Path(__file__).resolve().parents[1])
        self.server.regions={};self.server.jobs=None
        self.thread=threading.Thread(target=self.server.serve_forever,name='browser-service-test')
        self.thread.start();self.base=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive());self.folder.cleanup()

    def request(self,path,value=None):
        request=urllib.request.Request(self.base+path,data=None if value is None else json.dumps(value).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request) as response:return json.load(response)

    def test_browser_only_server_can_plan_without_a_prepared_catalog(self):
        self.assertEqual(self.request('/api/catalog'),[])
        result=self.request('/api/coverage-plan',dict(bounds=[-74.48,40.53,-74.425,40.565],zoom=16))
        self.assertEqual(len(result['imagery_tiles']),110)

    def test_disabled_native_api_cannot_silently_process_an_upload(self):
        for path in ['/api/uploads','/api/localize']:
            with self.assertRaises(urllib.error.HTTPError) as caught:self.request(path,{})
            self.assertEqual(caught.exception.code,400)
            self.assertIn('disabled',json.load(caught.exception)['error'])

if __name__=='__main__':unittest.main()
