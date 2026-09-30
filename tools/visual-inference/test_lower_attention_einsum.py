import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from lower_attention_einsum import lower, prepare


def fixture(equation, first, second, output):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Einsum", ["a", "b"], ["out"], equation=equation)],
        "attention", [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
                      for name, shape in [("a", first), ("b", second)]],
        [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, output)],
    )
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=9)


class AttentionLoweringTests(unittest.TestCase):
    def test_contractions_execute_for_unequal_sequence_lengths_and_batches(self):
        rng = np.random.default_rng(71)
        cases = [
            ("bhid, bhjd -> bhij", [2, 3, 7, 5], [2, 3, 11, 5], [2, 3, 7, 11]),
            ("bhij,bhjd->bhid", [2, 3, 7, 11], [2, 3, 11, 5], [2, 3, 7, 5]),
            ("bhji,bhjd->bhid", [2, 3, 11, 7], [2, 3, 11, 5], [2, 3, 7, 5]),
            ("bmd,bnd->bmn", [2, 7, 5], [2, 11, 5], [2, 7, 11]),
        ]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        for equation, a, b, output in cases:
            with self.subTest(equation=equation):
                model = fixture(equation, a, b, output)
                original = model.SerializeToString()
                lowered, equations = lower(model)
                self.assertEqual(model.SerializeToString(), original)
                self.assertEqual(len(equations), 1)
                feeds = {name: rng.standard_normal(shape).astype(np.float32)
                         for name, shape in [("a", a), ("b", b)]}
                expected = np.einsum(equation, feeds["a"], feeds["b"])
                session = ort.InferenceSession(lowered.SerializeToString(), options, providers=["CPUExecutionProvider"])
                actual = session.run(None, feeds)[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_unsupported_equation_leaves_source_unchanged(self):
        model = fixture("ij,ji->ij", [3, 4], [4, 3], [3, 4])
        original = model.SerializeToString()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            lower(model)
        self.assertEqual(original, model.SerializeToString())

    def test_preparation_binds_model_identity_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = [Path(directory)/name for name in ["source.onnx", "result.onnx"]]
            onnx.save(fixture("bmd,bnd->bmn", [1, 7, 5], [1, 11, 5], [1, 7, 11]), source)
            receipt = prepare(source, destination)
            self.assertNotEqual(receipt["source_sha256"], receipt["model_sha256"])
            before = destination.read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(source, destination)
            self.assertEqual(before, destination.read_bytes())
            with self.assertRaisesRegex(ValueError, "no supported"):
                lower(onnx.load(destination))


if __name__ == "__main__":
    unittest.main()
