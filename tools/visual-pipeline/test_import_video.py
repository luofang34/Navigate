import hashlib
import json
import sys
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('import_video', Path(__file__).with_name('import-video.py'))
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


class VideoImportTests(unittest.TestCase):
    def test_real_video_preserves_clock_and_source_without_inventing_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.mp4'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi',
                            '-i', 'testsrc2=size=160x90:rate=30:duration=1',
                            '-c:v', 'libx264', '-metadata', 'creation_time=2026-10-05T23:00:00Z',
                            str(source)], check=True)
            output = root / 'recording'
            subprocess.run([sys.executable, str(Path(__file__).with_name('run.py')),
                            'import-video', str(source), str(output)], check=True)
            result = json.loads((output / 'replay.json').read_text())
            self.assertEqual(len(result['frames']), 5)
            self.assertEqual(result['time_anchor']['unix_ms'], 1791241200000)
            self.assertIsNone(result['time_anchor']['uncertainty_ms'])
            self.assertIsNone(result['origin'])
            self.assertIsNone(result['agl_m'])
            self.assertIsNone(result['mosaic'])
            copied = output / result['source']['path']
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), result['source_sha256'])
            self.assertTrue(all((output / f['image']).is_file() and f['position'] is None for f in result['frames']))
            with self.assertRaises(ValueError):
                importer.import_video(source, output)


if __name__ == '__main__':
    unittest.main()
