"""Guard the RKNN simulator comparison against unusable tensor evidence."""

import unittest

import numpy as np
from compile_rknn import OUTPUT_SHAPES, compare_outputs


class SimulatorComparisonTests(unittest.TestCase):
    def setUp(self):
        self.expected = [
            np.zeros(shape, np.float32) for shape in OUTPUT_SHAPES.values()
        ]
        self.actual = [value.copy() for value in self.expected]

    def test_small_finite_drift_is_reported(self):
        self.actual[0].flat[0] = 0.01
        report = compare_outputs(self.actual, self.expected, 0.2)
        self.assertAlmostEqual(report["descriptors"]["max_abs_error"], 0.01)
        self.assertGreater(report["descriptors"]["mean_abs_error"], 0)

    def test_missing_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "all three"):
            compare_outputs(self.actual[:-1], self.expected, 0.2)

    def test_wrong_shape_is_rejected(self):
        self.actual[0] = self.actual[0].flatten()
        with self.assertRaisesRegex(ValueError, "expected shape"):
            compare_outputs(self.actual, self.expected, 0.2)

    def test_nonfinite_output_is_rejected(self):
        for value in (float("nan"), float("inf")):
            self.actual[1].flat[0] = value
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compare_outputs(self.actual, self.expected, 0.2)

    def test_nonfinite_reference_is_rejected(self):
        self.expected[2].flat[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_outputs(self.actual, self.expected, 0.2)

    def test_excessive_drift_is_rejected(self):
        self.actual[0].flat[0] = 0.3
        with self.assertRaisesRegex(ValueError, "exceeds"):
            compare_outputs(self.actual, self.expected, 0.2)


if __name__ == "__main__":
    unittest.main()
