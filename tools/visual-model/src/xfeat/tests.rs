use super::*;
use webnn_onnx_utils::protos::onnx::{AttributeProto, TensorProto, attribute_proto::AttributeType};

#[test]
fn dense_preparation_refuses_foreign_exports() {
    assert!(matches!(
        prepare(b"foreign model"),
        Err(Error::Digest { .. })
    ));
}

#[test]
fn patch_folding_preserves_weights_and_avoids_expanded_patch_input() {
    let mut patch = graph::node("Conv", &["patches", "weights", "bias"], OUTPUTS[1].0);
    patch.name = PATCH.into();
    patch.attribute = ["kernel_shape", "strides"]
        .map(|name| AttributeProto {
            name: name.into(),
            ints: vec![1, 1],
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        })
        .to_vec();
    let values: Vec<f32> = (0..4096).map(|n| n as f32 / 4096.0).collect();
    let mut model = GraphProto {
        input: vec![graph::value("images", 1, &[1, 3, 600, 800])],
        node: vec![
            graph::node("Identity", &["images"], NORMALIZED),
            graph::node("Expand", &[NORMALIZED], "patches"),
            patch,
            graph::node("NonZero", &[OUTPUTS[1].0], "sparse"),
        ],
        initializer: vec![TensorProto {
            name: "weights".into(),
            dims: vec![64, 64, 1, 1],
            data_type: 1,
            float_data: values.clone(),
            ..Default::default()
        }],
        ..Default::default()
    };
    transform(&mut model).unwrap();
    assert_eq!(model.initializer[0].float_data, values);
    assert_eq!(model.initializer[0].dims, [64, 1, 8, 8]);
    assert_eq!(
        model
            .node
            .iter()
            .map(|n| n.op_type.as_str())
            .collect::<Vec<_>>(),
        ["Identity", "Conv"]
    );
    let conv = &model.node[1];
    assert_eq!(conv.input[0], NORMALIZED);
    assert!(conv.attribute.iter().all(|a| a.ints == [8, 8]));
    assert_eq!(conv.output, ["keypoint_logits"]);
    assert_eq!(model.output.len(), 3);
}
