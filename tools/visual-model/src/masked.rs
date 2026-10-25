//! Exclude padded features from every attention and assignment normalization.
use crate::{Error, graph, recipe};
use prost::Message;
use webnn_onnx_utils::protos::onnx::{GraphProto, ModelProto};

pub fn prepare(bytes: &[u8], count: usize, compact: bool) -> Result<Vec<u8>, Error> {
    let mut model = ModelProto::decode(recipe::prepare(bytes, count, false)?.as_slice())?;
    let graph = model.graph.as_mut().ok_or(Error::Invalid("model graph"))?;
    insert_masks(graph, count as i64)?;
    if compact {
        graph::compact(graph, count as i64);
    }
    for property in &mut model.metadata_props {
        if property.key == "navigate.recipe" {
            property.value = "lighterglue-masked-probability-v1".into();
        } else if property.key == "navigate.compact" {
            property.value = compact.to_string();
        }
    }
    Ok(model.encode_to_vec())
}

fn insert_masks(model: &mut GraphProto, count: i64) -> Result<(), Error> {
    for name in ["key_bias0", "key_bias1"] {
        model.input.push(graph::value(name, 1, &[1, 1, count]));
    }
    let mut transpose = graph::node("Transpose", &["key_bias0"], "navigate.row_bias");
    transpose
        .attribute
        .push(webnn_onnx_utils::protos::onnx::AttributeProto {
            name: "perm".into(),
            ints: vec![0, 2, 1],
            r#type: 7,
            ..Default::default()
        });
    let mut nodes = vec![transpose];
    let mut attention = 0_usize;
    let mut assignment = 0_usize;
    for mut node in std::mem::take(&mut model.node) {
        let mask = match node.name.as_str() {
            "/LogSoftmax" => Some("key_bias1"),
            "/LogSoftmax_1" => Some("navigate.row_bias"),
            name if name == "/Softmax" || name.starts_with("/Softmax_") => {
                let expected = if attention == 0 {
                    "/Softmax".into()
                } else {
                    format!("/Softmax_{attention}")
                };
                if name != expected {
                    return Err(Error::Invalid("attention order for masked matcher"));
                }
                let mask = ["key_bias0", "key_bias1", "key_bias1", "key_bias0"][attention % 4];
                attention = attention.wrapping_add(1);
                Some(mask)
            }
            _ => None,
        };
        if let Some(mask) = mask {
            if node.op_type != "Softmax" || node.input.len() != 1 {
                return Err(Error::Invalid("masked matcher normalization"));
            }
            assignment = assignment.wrapping_add(usize::from(node.name.starts_with("/LogSoftmax")));
            let output = format!("navigate.masked{}", node.name);
            nodes.push(graph::node("Add", &[&node.input[0], mask], &output));
            node.input[0] = output;
        }
        nodes.push(node);
    }
    if attention != 24 || assignment != 2 {
        return Err(Error::Invalid("masked matcher attention count"));
    }
    model.node = nodes;
    Ok(())
}
