"""Check immutable package selection, byte ranges, and corrupt data rejection."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from webserver.packages import build_region, materialize


class PackageTests(unittest.TestCase):
    def test_chunks_bind_both_consumers_to_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            state = root / "state"
            for name in ["chunks", "packs", "catalog"]:
                (state / name).mkdir(parents=True)
            assets = {}
            for role, size, color in [
                ("imagery", 512, (10, 30, 50, 255)),
                ("elevation", 256, (128, 0, 0, 255)),
            ]:
                path = source / (role + ".png")
                Image.new("RGBA", (size, size), color).save(path)
                assets[role] = dict(
                    path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
            manifest = dict(
                schema_version=2,
                release_id="test",
                anchor_lat_lon=[0, 0],
                elevation_datum="fixture",
                attribution="fixture",
                tiles=[dict(xyz=[0, 0, 0], **assets)],
            )
            (source / "map.json").write_text(json.dumps(manifest))
            first = build_region(source, state, "test", "Test")
            second = build_region(source, state, "test", "Test")
            self.assertEqual(first["pack_id"], second["pack_id"])
            output = root / "decoded"
            materialize(first["manifest"], state, output)
            decoded = json.loads((output / "map.json").read_text())
            for role in assets:
                expected = (source / assets[role]["path"]).read_bytes()
                actual = (output / decoded["tiles"][0][role]["path"]).read_bytes()
                self.assertEqual(expected, actual)
            chunk = (
                state / "chunks" / (first["manifest"]["files"][0]["sha256"] + ".bin")
            )
            chunk.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Checksum failed"):
                materialize(first["manifest"], state, root / "bad")
            self.assertFalse((root / "bad/map.json").exists())

    def test_source_paths_cannot_leave_the_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            state = root / "state"
            for name in ["chunks", "packs", "catalog"]:
                (state / name).mkdir(parents=True)
            (root / "outside").write_bytes(b"outside")
            value = dict(
                schema_version=2,
                tiles=[
                    dict(
                        xyz=[0, 0, 0], imagery=dict(path="../outside", sha256="0" * 64)
                    )
                ],
            )
            (source / "map.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "escapes package root"):
                build_region(source, state, "test", "Test")


if __name__ == "__main__":
    unittest.main()
