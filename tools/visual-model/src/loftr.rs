use crate::{
    Error, graph,
    onnx_proto::{
        GraphProto, ModelProto, StringStringEntryProto, ValueInfoProto, tensor_shape_proto,
        type_proto,
    },
};
use prost::Message;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
const SOURCE: &str = "49901002e682c650dcc1cd596d4effa980f12ffcc383746b77651ac89ed88559";
const PATCHES: [&str; 2] = [
    "/net/fine_preprocess/merge_feat/Add_output_0",
    "/net/fine_preprocess/merge_feat_1/Add_output_0",
];
const QUERY: &str = "/net/coarse_matching/Mul_2_output_0";
const OFFSETS: &str = "/net/fine_matching/Mul_output_0";

pub fn prepare(bytes: &[u8], batch: Option<usize>) -> Result<Vec<u8>, Error> {
    if batch.is_some_and(|n| !(1..=1024).contains(&n)) {
        return Err(Error::Invalid("LoFTR fine batch must be 1 through 1024"));
    }
    let digest = format!("{:x}", Sha256::digest(bytes));
    if digest != SOURCE {
        return Err(Error::Digest {
            actual: digest,
            expected: SOURCE,
        });
    }
    let mut model = ModelProto::decode(bytes)?;
    let graph = model
        .graph
        .as_mut()
        .ok_or(Error::Invalid("missing model graph"))?;
    transform(graph, batch)?;
    model.metadata_props.push(StringStringEntryProto {
        key: "navigate.loftr.source_sha256".into(),
        value: SOURCE.into(),
    });
    model.metadata_props.push(StringStringEntryProto {
        key: "navigate.loftr.stage".into(),
        value: batch.map_or_else(
            || "coarse-patches-v1".into(),
            |n| format!("fine-offsets-v1/batch{n}"),
        ),
    });
    Ok(model.encode_to_vec())
}
fn dynamic_value(name: &str, dimensions: &[i64]) -> ValueInfoProto {
    let mut value = graph::value(name, 1, dimensions);
    if let Some(type_proto::Value::TensorType(t)) =
        value.r#type.as_mut().and_then(|t| t.value.as_mut())
        && let Some(dim) = t.shape.as_mut().and_then(|s| s.dim.first_mut())
    {
        dim.value = Some(tensor_shape_proto::dimension::Value::DimParam(
            "matches".into(),
        ));
    }
    value
}
fn transform(model: &mut GraphProto, batch: Option<usize>) -> Result<(), Error> {
    for name in PATCHES
        .into_iter()
        .chain([QUERY, OFFSETS, "keypoints0", "confidence"])
    {
        if !model
            .node
            .iter()
            .any(|n| n.output.iter().any(|o| o == name))
        {
            return Err(Error::Invalid("missing LoFTR stage boundary"));
        }
    }
    model.value_info.clear();
    if let Some(batch) = batch {
        model
            .node
            .retain(|n| !n.output.iter().any(|o| PATCHES.contains(&o.as_str())));
        model.input = PATCHES
            .iter()
            .map(|name| graph::value(name, 1, &[batch as i64, 25, 128]))
            .collect();
        model
            .node
            .push(graph::node("Identity", &[OFFSETS], "offsets"));
        model.output = vec![graph::value("offsets", 1, &[batch as i64, 2])];
    } else {
        for (source, name) in [
            (PATCHES[0], "patches0"),
            (PATCHES[1], "patches1"),
            (QUERY, "query_coarse"),
        ] {
            model.node.push(graph::node("Identity", &[source], name));
        }
        model.output = vec![
            dynamic_value("patches0", &[0, 25, 128]),
            dynamic_value("patches1", &[0, 25, 128]),
            dynamic_value("keypoints0", &[0, 2]),
            dynamic_value("query_coarse", &[0, 2]),
            dynamic_value("confidence", &[0]),
        ];
    }
    graph::prune(model);
    if batch.is_some() {
        for (old, new) in PATCHES.into_iter().zip(["patches0", "patches1"]) {
            for input in &mut model.input {
                if input.name == old {
                    input.name = new.into();
                }
            }
            for node in &mut model.node {
                for input in &mut node.input {
                    if input == old {
                        *input = new.into();
                    }
                }
            }
        }
    }
    let mut available: BTreeSet<_> = model
        .input
        .iter()
        .map(|v| v.name.as_str())
        .chain(model.initializer.iter().map(|v| v.name.as_str()))
        .collect();
    for node in &model.node {
        if node
            .input
            .iter()
            .any(|v| !v.is_empty() && !available.contains(v.as_str()))
        {
            return Err(Error::Invalid(
                "LoFTR stage has an external graph dependency",
            ));
        }
        available.extend(node.output.iter().map(String::as_str));
    }
    Ok(())
}
#[cfg(test)]
mod tests;
