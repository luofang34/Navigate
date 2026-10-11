use webnn_onnx_utils::protos::onnx::{self as p, NodeProto, ValueInfoProto};

pub fn value(name: &str, element: i32, dimensions: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(p::TypeProto {
            value: Some(p::type_proto::Value::TensorType(p::type_proto::Tensor {
                elem_type: element,
                shape: Some(p::TensorShapeProto {
                    dim: dimensions
                        .iter()
                        .map(|n| p::tensor_shape_proto::Dimension {
                            value: Some(p::tensor_shape_proto::dimension::Value::DimValue(*n)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}
pub fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        name: format!("navigate/{output}"),
        input: inputs.iter().map(|s| (*s).into()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}
pub fn integer(name: &str, number: i64) -> p::AttributeProto {
    p::AttributeProto {
        name: name.into(),
        i: number,
        r#type: p::attribute_proto::AttributeType::Int as i32,
        ..Default::default()
    }
}
pub fn compact(graph: &mut p::GraphProto, count: i64) {
    for (name, axis) in [("row_indices", 2), ("column_indices", 1)] {
        let mut op = node("ArgMax", &["assignment"], name);
        op.attribute = vec![
            integer("axis", axis),
            integer("keepdims", 0),
            integer("select_last_index", 0),
        ];
        graph.node.push(op);
    }
    let mut max = node("ReduceMax", &["assignment"], "row_scores");
    max.attribute = vec![
        integer("keepdims", 0),
        p::AttributeProto {
            name: "axes".into(),
            ints: vec![2],
            r#type: p::attribute_proto::AttributeType::Ints as i32,
            ..Default::default()
        },
    ];
    graph.node.push(max);
    graph.output = vec![
        value("row_indices", 7, &[1, count]),
        value("column_indices", 7, &[1, count]),
        value("row_scores", 1, &[1, count]),
    ];
}

/// Retain the topologically sorted dependencies of every declared output.
pub fn prune(graph: &mut p::GraphProto) {
    let mut needed: std::collections::BTreeSet<_> =
        graph.output.iter().map(|v| v.name.clone()).collect();
    let mut keep = std::collections::BTreeSet::new();
    for (index, node) in graph.node.iter().enumerate().rev() {
        if node.output.iter().any(|o| needed.contains(o)) {
            keep.insert(index);
            needed.extend(node.input.iter().cloned());
        }
    }
    graph.node = std::mem::take(&mut graph.node)
        .into_iter()
        .enumerate()
        .filter_map(|(i, node)| keep.contains(&i).then_some(node))
        .collect();
    graph
        .initializer
        .retain(|tensor| needed.contains(&tensor.name));
}
