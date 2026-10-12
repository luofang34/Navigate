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

#[test]
fn grayscale_replicates_on_device_and_keeps_mean_order() {
    let mut model = GraphProto {
        input: vec![graph::value("images", 1, &[1, 3, 600, 800])],
        node: vec![graph::node("ReduceMean", &["images"], "mean")],
        ..Default::default()
    };
    grayscale(&mut model).unwrap();
    assert_eq!(model.input[0], graph::value("images", 1, &[1, 1, 600, 800]));
    assert_eq!(model.node[0].op_type, "Concat");
    assert_eq!(model.node[0].input, ["images", "images", "images"]);
    assert_eq!(model.node[0].attribute[0].i, 1);
    assert_eq!(model.node[1].op_type, "ReduceMean");
    assert_eq!(model.node[1].input, ["navigate_rgb"]);
}

#[test]
fn resolution_recipe_rejects_unsupported_channel_combinations() {
    assert!(matches!(
        prepare_scaled(b"", 2, false),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        prepare_scaled(b"", 3, true),
        Err(Error::Invalid(_))
    ));
}
