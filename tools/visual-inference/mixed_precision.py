"""Use half precision for convolutions and float32 for descriptor normalization."""

import argparse
from pathlib import Path
import onnx
from onnxconverter_common import float16


def convert(source, target):
    model = onnx.load(source)
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=["ReduceL2", "Clip", "Expand", "Div"],
        node_block_list=["/Constant_1"],
    )
    # The converter inserts a half output cast for a blocked graph output.
    for output in model.graph.output:
        if output.name == "descriptors":
            output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            node = next(n for n in model.graph.node if output.name in n.output)
            if node.op_type != "Cast":
                raise ValueError("Unexpected descriptor output graph")
            next(a for a in node.attribute if a.name == "to").i = onnx.TensorProto.FLOAT
    onnx.checker.check_model(model)
    onnx.save(model, target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    convert(a.source, a.output)
