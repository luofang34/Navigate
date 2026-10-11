"""Clock import preserves source claims and never promotes filesystem times."""
import struct
import tempfile
import unittest
from pathlib import Path
from capture_clock import anchor_from_movie_header, capture_anchor


class CaptureClockTests(unittest.TestCase):
    def test_versioned_integer_clock_and_unknown_accuracy(self):
        for version, width in [(0, 4), (1, 8)]:
            data = bytes([version, 0, 0, 0]) + (3837102974).to_bytes(width, "big") + bytes(32)
            movie = struct.pack(">I4s", len(data) + 8, b"mvhd") + data
            anchor = anchor_from_movie_header(movie, "DJI_0029.MP4")
            self.assertEqual(anchor["unix_ms"], 1754258174000)
            self.assertIsNone(anchor["uncertainty_ms"])
            self.assertEqual(anchor["resolution_ms"], 1000)

    def test_missing_and_zero_times_never_use_the_file_modification_date(self):
        movie = struct.pack(">I4s", 24, b"mvhd") + bytes(16)
        self.assertIsNone(anchor_from_movie_header(movie, "zero.mp4"))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "missing.mp4"
            path.write_bytes(struct.pack(">I4s", 12, b"ftyp") + b"isom")
            self.assertIsNone(capture_anchor(path))

    def test_truncated_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            anchor_from_movie_header(struct.pack(">I4s", 64, b"mvhd") + bytes(8), "bad.mp4")


if __name__ == "__main__":
    unittest.main()
