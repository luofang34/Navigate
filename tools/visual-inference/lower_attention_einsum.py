"""Prepare an embedded-weight ONNX model with batched attention matrix products.

This tool does not change weights, feature counts, or matching thresholds.
The Rust and browser runtimes do not use Python.
"""
import argparse
import hashlib
import json
from pathlib import Path

import onnx


def lower(model):
    """Replace four explicit attention contractions. Reject other equations."""
    result = onnx.ModelProto()
    result.CopyFrom(model)
    onnx.checker.check_model(result)
    names = {value.name for value in result.graph.input}
    names.update(value.name for value in result.graph.initializer)
    names.update(name for node in result.graph.node for name in node.output)
    nodes, equations = [], []
    for node in result.graph.node:
        if node.op_type != "Einsum" or node.domain not in ("", "ai.onnx"):
            nodes.append(node)
            continue
        equation = next(
            (onnx.helper.get_attribute_value(a).decode().replace(" ", "")
             for a in node.attribute if a.name == "equation"), ""
        )
        if len(node.input) != 2 or len(node.output) != 1:
            raise ValueError("attention contraction must have two inputs and one output")
        inputs = list(node.input)
        if equation == "bhij,bhjd->bhid":
            side, permutation = None, None
        elif equation == "bhid,bhjd->bhij":
            side, permutation = 1, [0, 1, 3, 2]
        elif equation == "bhji,bhjd->bhid":
            side, permutation = 0, [0, 1, 3, 2]
        elif equation == "bmd,bnd->bmn":
            side, permutation = 1, [0, 2, 1]
        else:
            raise ValueError(f"unsupported attention equation: {equation}")
        if side is not None:
            side_name = "rhs" if side == 1 else "lhs"
            name = f"{node.output[0]}_navigate_{side_name}_transposed"
            if name in names:
                raise ValueError(f"generated value name already exists: {name}")
            names.add(name)
            nodes.append(onnx.helper.make_node(
                "Transpose", [inputs[side]], [name], name=node.name + f"_{side_name}_transpose", perm=permutation
            ))
            inputs[side] = name
        nodes.append(onnx.helper.make_node(
            "MatMul", inputs, list(node.output), name=node.name + "_matmul"
        ))
        equations.append(equation)
    if not equations:
        raise ValueError("model has no supported attention contractions")
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    result = onnx.shape_inference.infer_shapes(result)
    onnx.checker.check_model(result)
    return result, equations


def prepare(source: Path, destination: Path):
    """Write a distinct model identity. Runtime placement needs separate tests."""
    receipt_path = destination.with_suffix(".lowering.json")
    if destination.exists() or receipt_path.exists():
        raise ValueError("model or receipt destination already exists")
    original = source.read_bytes()
    model = onnx.load_model_from_string(original)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError("source weights must be embedded in the ONNX file")
    lowered, equations = lower(model)
    encoded = lowered.SerializeToString()
    receipt = {
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "model_sha256": hashlib.sha256(encoded).hexdigest(),
        "equations": equations,
        "scope": "Algebraic graph rewrite. Floating-point rounding can change. "
                 "Validate assignments, image geometry, and device performance before use.",
    }
    with destination.open("xb") as output:
        output.write(encoded)
    with receipt_path.open("x") as output:
        json.dump(receipt, output, indent=2)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)


if __name__ == "__main__":
    main()
