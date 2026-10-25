"""Portable exports preserve camera pixels and reject changed download inputs."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from photo_publication import export
from photo_reproduce import contained, download, sha256


class Publication(unittest.TestCase):
    def test_export_deduplicates_basemap_without_changing_photo_or_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            picture = np.arange(24*32*3, dtype=np.uint8).reshape(24, 32, 3)
            cv2.imwrite(str(source / 'photo.png'), picture)
            cv2.imwrite(str(source / 'base.png'), picture)
            camera = {'size': [32, 24], 'focal': [25, 25]}
            replay = {'frames': [{'image': 'photo.png'}], 'reconstruction': {
                'camera': camera, 'reference_image': 'base.png',
                'basemap': {'image': 'base.png', 'corners': []}}}
            (source / 'a.json').write_text(json.dumps(replay))
            archive = {'units': [{'media': [{'id': 'one', 'replay': 'a.json'},
                                            {'id': 'two', 'replay': 'a.json'}]}]}
            (source / 'debrief.json').write_text(json.dumps(archive))
            output = root / 'output'
            result = export(source / 'debrief.json', output)
            self.assertEqual(len(result['units'][0]['media']), 2)
            first = json.loads((output / 'one.json').read_text())
            second = json.loads((output / 'two.json').read_text())
            self.assertEqual(first['reconstruction']['camera'], camera)
            self.assertEqual(first, second)
            self.assertEqual(sha256(output / first['frames'][0]['image']), sha256(source / 'photo.png'))
            self.assertEqual(len(list(output.glob('*-basemap.png'))), 1)
            self.assertEqual(len(list(output.glob('*.png'))), 3)

    def test_download_verifies_content_before_atomic_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'remote'
            source.write_bytes(b'complete data')
            item = {'path': 'nested/data', 'url': source.as_uri(), 'bytes': source.stat().st_size,
                    'sha256': sha256(source)}
            target = download(item, root / 'output')
            self.assertEqual(target.read_bytes(), source.read_bytes())
            target.write_bytes(b'edited')
            with self.assertRaisesRegex(ValueError, 'local file differs'):
                download(item, root / 'output')
            item['path'] = 'rejected'
            item['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'remote content changed'):
                download(item, root / 'output')
            self.assertFalse((root / 'output/rejected').exists())
            for name in ['../escape', '/absolute', 'nested/../escape', 'nested\\escape']:
                with self.assertRaises(ValueError):
                    contained(root / 'output', name)


if __name__ == '__main__':
    unittest.main()
