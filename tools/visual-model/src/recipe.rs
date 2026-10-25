use crate::{Error, graph};
use prost::Message;
use sha2::{Digest, Sha256};
use webnn_onnx_utils::protos::onnx::{GraphProto, ModelProto, StringStringEntryProto, TensorProto};

const INPUT_SHA256: &str = "170b10e74763fbfca483245cfa279972fc6747444065db2538500e22fc7ce9ca";

pub fn prepare(bytes: &[u8], count: usize, compact: bool) -> Result<Vec<u8>, Error> {
    if bytes.len() > 64 * 1024 * 1024 || !(32..=4096).contains(&count) {
        return Err(Error::Invalid("model size or feature count"));
    }
    let actual = format!("{:x}", Sha256::digest(bytes));
    // The export's named subgraph is a recipe contract, not a general ONNX rewrite rule.
    if actual != INPUT_SHA256 {
        return Err(Error::Digest {
            actual,
            expected: INPUT_SHA256,
        });
    }
    let mut model = ModelProto::decode(bytes)?;
    let graph = model.graph.as_mut().ok_or(Error::Invalid("model graph"))?;
    transform(graph, count as i64, compact)?;
    for (key, value) in [
        ("navigate.source_sha256", INPUT_SHA256.to_owned()),
        ("navigate.recipe", "lighterglue-probability-v1".into()),
        ("navigate.features", count.to_string()),
        ("navigate.compact", compact.to_string()),
    ] {
        model.metadata_props.push(StringStringEntryProto {
            key: key.into(),
            value,
        });
    }
    Ok(model.encode_to_vec())
}

fn transform(model: &mut GraphProto, count: i64, compact: bool) -> Result<(), Error> {
    for input in &mut model.input {
        let width = match input.name.as_str() {
            "keypoints0" | "keypoints1" => 2,
            "descriptors0" | "descriptors1" => 64,
            _ => return Err(Error::Invalid("unexpected matcher input")),
        };
        input.r#type = graph::value(&input.name, 1, &[1, count, width]).r#type;
    }
    for node in &mut model.node {
        match node.name.as_str() {
            "/LogSoftmax" | "/LogSoftmax_1" => node.op_type = "Softmax".into(),
            "/Add_48" | "/Add_50" | "/Add_52" => node.op_type = "Mul".into(),
            "/Sub" => {
                *node = graph::node("Sigmoid", &["/matchability/Add_output_0"], "/Sub_output_0")
            }
            "/Sub_1" => {
                *node = graph::node(
                    "Sigmoid",
                    &["/matchability_1/Add_output_0"],
                    "/Sub_1_output_0",
                )
            }
            _ => {}
        }
        if node.op_type == "Neg" {
            node.op_type = "Mul".into();
            node.input.push("navigate.neg_one".into());
        }
        for output in &mut node.output {
            if output == "log_assignment" {
                *output = "assignment".into();
            }
        }
    }
    model.initializer.push(TensorProto {
        name: "navigate.neg_one".into(),
        data_type: 1,
        float_data: vec![-1.0],
        ..Default::default()
    });
    model.output = vec![graph::value("assignment", 1, &[1, count, count])];
    model.value_info.clear();
    graph::prune(model);
    if compact {
        graph::compact(model, count);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
