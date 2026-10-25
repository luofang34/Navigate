"""Research cameras remain opt-in, labelled, and tied to their solver evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from photo_preview import research_inputs
from photo_reference import tile
from photo_surface import Surface
from photo_testset import digest, write_json


class ResearchPreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'run'; self.run.mkdir()
        z = 17; x, y = tile(41, -75, z)
        tiles = []
        for kind, color in [('imagery', (20, 80, 120)), ('dem', (0, 100, 128))]:
            path = self.root / (kind + '.png')
            cv2.imwrite(str(path), np.full((256, 256, 3), color, np.uint8))
            tiles.append({'kind': kind, 'xyz': [z, x, y], 'path': path.name, 'sha256': digest(path)})
        self.catalog = self.root / 'catalog.json'
        write_json(self.catalog, {'complete': True, 'configuration': {'bbox': [-75.1, 40.9, -74.9, 41.1]},
                                 'tiles': tiles})
        surface = Surface(self.catalog)
        lon = (x + .5) / 2**z * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + .5) / 2**z))))
        np.savez(self.run / 'camera-0.npz', eye=surface.local(lat, lon, 300),
                 rotation=surface.camera_rotation(lat, lon, 0, 90), size=[32, 24],
                 K=np.array([[80., 0, 15.5], [0, 80., 11.5], [0, 0, 1]]))
        cv2.imwrite(str(self.root / 'photo.png'), np.full((24, 32, 3), 120, np.uint8))
        cv2.imwrite(str(self.root / 'mask.png'), np.full((24, 32), 255, np.uint8))
        entry={'id': 'q-one', 'source': 'IMG_0001.JPG', 'image': 'photo.png',
               'image_sha256': digest(self.root / 'photo.png'), 'valid_mask': 'mask.png',
               'mask_sha256': digest(self.root / 'mask.png'), 'claims': {'capture_time': '2021-10-02T12:00:00'}}
        write_json(self.root / 'manifest.json', {'photos': [entry],
                   'flights': [{'id': 'flight-a', 'photos': ['q-one']}]})
        config={'catalog_sha256': digest(self.catalog)}
        write_json(self.run / 'run.json', config)
        write_json(self.run / 'report.json', {**config, 'cases': ['q-one'],
                   'status': 'research_proposal', 'geometry_accepted': False})
        write_json(self.run / 'q-one.json', {'id': 'q-one', 'status': 'research_proposal',
                   'geometry_accepted': False, 'camera': 'camera-0.npz',
                   'camera_sha256': digest(self.run / 'camera-0.npz'),
                   'report_sha256': digest(self.run / 'report.json')})

    def test_command_requires_opt_in_and_app_loads_labelled_flight(self):
        script = Path(__file__).with_name('photo_preview.py')
        for include in [False, True]:
            output = self.root / ('research' if include else 'default')
            result = subprocess.run([sys.executable, str(script), str(self.root), str(output),
                '--catalog', str(self.catalog), '--run', str(self.run),
                *(['--include-research'] if include else [])], check=True, capture_output=True, text=True)
            summary = json.loads(result.stdout)
            self.assertEqual(summary['geometric_candidates'], 0)
            self.assertEqual(summary['research_proposals'], int(include))
            replay = json.loads((output / 'q-one/replay.json').read_text())
            self.assertEqual('reconstruction' in replay, include)
            if include:
                self.assertIn('research', replay['name'])
                self.assertIn('requires map verification', replay['registration'])
                self.assertIsNone(replay['frames'][0]['navigation'])
                archive = json.loads((output / 'debrief.json').read_text())
                self.assertEqual(archive['units'][0]['id'], 'flight-a 2021-10-02')
        if os.environ.get('SOKOLY_APP_BINARY'):
            subprocess.run([os.environ['SOKOLY_APP_BINARY'], '--video-overlay',
                            str(self.root / 'research/debrief.json'),
                            '--export-facts', str(self.root / 'facts')], check=True, capture_output=True)
        self.assertFalse(json.loads((self.run / 'q-one.json').read_text())['geometry_accepted'])
        mask=self.root / 'mask.png';mask.write_bytes(mask.read_bytes()+b'changed')
        rejected=subprocess.run([sys.executable,str(script),str(self.root),str(self.root/'research'),
            '--catalog',str(self.catalog),'--run',str(self.run),'--include-research'],capture_output=True,text=True)
        self.assertNotEqual(rejected.returncode,0)
        self.assertIn('projection mask changed',rejected.stderr)

    def test_changed_camera_or_report_cannot_be_presented_as_solver_output(self):
        self.assertEqual(set(research_inputs([self.run])), {'q-one'})
        path = self.run / 'camera-0.npz'; original = path.read_bytes()
        path.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'camera changed'):
            research_inputs([self.run])
        path.write_bytes(original)
        (self.run / 'report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            research_inputs([self.run])


if __name__ == '__main__':
    unittest.main()
