"""Position searches preserve prior bounds and spherical distances."""
import math
import unittest

from photo_grid import location, offsets
from photo_surface import basis


class PositionGridTests(unittest.TestCase):
    def test_uncertainty_grid_visits_nearby_locations_first(self):
        position = {'reported_horizontal_error_m': 6531.2119914346895}
        grid = offsets(position, 2000)
        self.assertEqual(len(grid), 37)
        self.assertEqual(grid[0], (0, 0))
        distances = [math.hypot(*point) for point in grid]
        self.assertEqual(distances, sorted(distances))
        self.assertLessEqual(max(distances), position['reported_horizontal_error_m'])
        self.assertEqual(offsets(position), [(0, 0)])
        self.assertEqual(offsets(position, 2000, 1000), [(0, 0)])
        with self.assertRaises(ValueError):
            offsets(position, 1)
        with self.assertRaises(ValueError):
            offsets(position, 0, 1000)
        with self.assertRaises(ValueError):
            offsets(position, float('nan'))

    def test_offsets_preserve_distance_across_the_date_line(self):
        self.assertEqual(location(41.14643888888889, -75.37546666666665, 0, 0),
                         (41.14643888888889, -75.37546666666665))
        for latitude, longitude in [(0, 179.99), (80, 179.99), (40.65, -74.88)]:
            lat, lon = location(latitude, longitude, 4000, 2000)
            dot = float(basis(latitude, longitude)[:, 2] @ basis(lat, lon)[:, 2])
            distance = math.acos(max(-1, min(1, dot))) * 6371008.8
            self.assertAlmostEqual(distance, math.hypot(4000, 2000), places=4)
            self.assertGreaterEqual(lon, -180)
            self.assertLess(lon, 180)


if __name__ == '__main__':
    unittest.main()
