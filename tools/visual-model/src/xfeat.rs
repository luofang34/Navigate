use crate::{Error, graph};
use prost::Message;
use sha2::{Digest, Sha256};
use webnn_onnx_utils::protos::onnx::{GraphProto, ModelProto, StringStringEntryProto};

const INPUT_SHA256: &str = "41336466bbbb09b701b815e33932c7da69cff6ab75a068230418468430482912";
const PATCH: &str = "/net/keypoint_head/keypoint_head.0/layer/layer.0/Conv";
const NORMALIZED: &str = "/net/norm/InstanceNormalization_output_0";
const OUTPUTS: [(&str, &str, i64); 3] = [
    (
        "/net/block_fusion/block_fusion.2/Conv_output_0",
        "dense_descriptors",
        64,
    ),
    (
        "/net/keypoint_head/keypoint_head.3/Conv_output_0",
        "keypoint_logits",
        65,
    ),
    (
        "/net/heatmap_head/heatmap_head.3/Sigmoid_output_0",
        "reliability",
        1,
    ),
];

#[cfg(test)]
pub fn prepare(bytes: &[u8]) -> Result<Vec<u8>, Error> {
    prepare_channels(bytes, 3)
}

pub fn prepare_channels(bytes: &[u8], channels: i64) -> Result<Vec<u8>, Error> {
    prepare_scaled(bytes, channels, false)
}

pub fn prepare_scaled(bytes: &[u8], channels: i64, double: bool) -> Result<Vec<u8>, Error> {
    if ![1, 3].contains(&channels) || (double && channels != 1) {
        return Err(Error::Invalid("XFeat channel and resolution combination"));
    }
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(Error::Invalid("model size"));
    }
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != INPUT_SHA256 {
        return Err(Error::Digest {
            actual,
            expected: INPUT_SHA256,
        });
    }
    let mut model = ModelProto::decode(bytes)?;
    transform(model.graph.as_mut().ok_or(Error::Invalid("model graph"))?)?;
    if channels == 1 {
        grayscale(model.graph.as_mut().ok_or(Error::Invalid("model graph"))?)?;
    }
    if double {
        let graph = model.graph.as_mut().ok_or(Error::Invalid("model graph"))?;
        graph.input[0] = graph::value("images", 1, &[1, channels, 1200, 1600]);
        graph.output = OUTPUTS
            .iter()
            .map(|(_, name, c)| graph::value(name, 1, &[1, *c, 148, 200]))
            .collect();
    }
    let shape = format!(
        "float32/1x{channels}x{}x{}",
        if double { 1200 } else { 600 },
        if double { 1600 } else { 800 }
    );
    for (key, value) in [
        ("navigate.source_sha256", INPUT_SHA256),
        (
            "navigate.recipe",
            if double {
                "xfeat-gray-conv-2x-v1"
            } else if channels == 1 {
                "xfeat-gray-conv-v1"
            } else {
                "xfeat-dense-conv-v1"
            },
        ),
        ("navigate.input", shape.as_str()),
    ] {
        model.metadata_props.push(StringStringEntryProto {
            key: key.into(),
            value: value.into(),
        });
    }
    Ok(model.encode_to_vec())
}

fn transform(model: &mut GraphProto) -> Result<(), Error> {
    if model.input.len() != 1 || model.input[0].name != "images" {
        return Err(Error::Invalid("XFeat images input"));
    }
    model.input[0].r#type = graph::value("images", 1, &[1, 3, 600, 800]).r#type;
    fold_patches(model)?;
    for node in &mut model.node {
        for name in node.input.iter_mut().chain(&mut node.output) {
            if let Some((_, replacement, _)) = OUTPUTS.iter().find(|(source, _, _)| source == name)
            {
                *name = (*replacement).into();
            }
        }
    }
    model.output = OUTPUTS
        .iter()
        .map(|(_, name, channels)| graph::value(name, 1, &[1, *channels, 72, 100]))
        .collect();
    model.value_info.clear();
    graph::prune(model);
    Ok(())
}

fn fold_patches(model: &mut GraphProto) -> Result<(), Error> {
    let patch = model
        .node
        .iter_mut()
        .find(|node| node.name == PATCH)
        .ok_or(Error::Invalid("XFeat patch convolution"))?;
    let weight = model
        .initializer
        .iter_mut()
        .find(|t| patch.input.get(1) == Some(&t.name))
        .ok_or(Error::Invalid("XFeat patch weights"))?;
    if patch.op_type != "Conv" || weight.dims != [64, 64, 1, 1] || patch.input.len() != 3 {
        return Err(Error::Invalid("XFeat patch convolution dimensions"));
    }
    // Unfold orders an 8×8 grayscale patch by row, exactly the kernel's contiguous order.
    // Reshaping weights eliminates patch materialization without changing learned values.
    weight.dims = vec![64, 1, 8, 8];
    patch.input[0] = NORMALIZED.into();
    for attribute in &mut patch.attribute {
        if matches!(attribute.name.as_str(), "kernel_shape" | "strides") {
            attribute.ints = vec![8, 8];
        }
    }
    Ok(())
}

#[cfg(test)]
mod execution_tests;
#[cfg(test)]
mod tests;

fn grayscale(model: &mut GraphProto) -> Result<(), Error> {
    let input = model
        .input
        .first_mut()
        .ok_or(Error::Invalid("XFeat input"))?;
    input.r#type = graph::value("images", 1, &[1, 1, 600, 800]).r#type;
    for node in &mut model.node {
        for input in &mut node.input {
            if input == "images" {
                *input = "navigate_rgb".into();
            }
        }
    }
    // Keep the channel mean's floating-point order while transferring grayscale only.
    let mut replicate = graph::node("Concat", &["images", "images", "images"], "navigate_rgb");
    replicate.attribute.push(graph::integer("axis", 1));
    model.node.insert(0, replicate);
    Ok(())
}
