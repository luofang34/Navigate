"""Catalog composition follows reported clocks and retains unobserved gaps."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("compose", Path(__file__).with_name("compose-survey.py"))
compose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compose)


class SurveyTests(unittest.TestCase):
    def test_two_flights_are_sorted_on_one_clock_and_missing_dates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, time, duration in [("DJI_0029", 1754258174000, 139.472667),
                                         ("DJI_0030", 1754258480000, 146.546400)]:
                path = root / (name + ".json")
                path.write_text(json.dumps({"name": name, "duration_s": duration,
                    "time_anchor": {"source_scale": "utc", "at": 0, "unix_ms": time}}))
                paths.append(path)
            result = compose.compose(root / "debrief.json", paths[::-1], "Survey", "UAV-1")
            self.assertEqual(len(result["units"]), 1)
            self.assertEqual(result["units"][0]["media"][1]["start_s"], 306)
            self.assertEqual(result["activities"][0]["recorded"]["end"], 139473)
            self.assertEqual(result["activities"][1]["recorded"]["end"], 452546)
            self.assertEqual(result["units"][0]["media"][0]["activity"], "flight-1")
            paths[1].write_text(json.dumps({"name":"unknown","duration_s":1}))
            with self.assertRaises(ValueError):
                compose.compose(root / "debrief.json", paths, "Survey", "UAV-1")


if __name__ == "__main__":
    unittest.main()
