//! Convert pinned XFeat and LighterGlue ONNX graphs to burn safetensors files.
use crate::Error;
use navigate_match::{WeightTensor, Weights};
use prost::Message;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use webnn_onnx_utils::protos::onnx::{GraphProto, ModelProto, TensorProto};

const XFEAT_SHA256: &str = "41336466bbbb09b701b815e33932c7da69cff6ab75a068230418468430482912";
const LIGHTERGLUE_SHA256: &str = "170b10e74763fbfca483245cfa279972fc6747444065db2538500e22fc7ce9ca";

const XFEAT_CONVS: &[(&str, &str)] = &[
    ("/net/block1/block1.0/layer/layer.0/Conv", "block1.0"),
    ("/net/block1/block1.1/layer/layer.0/Conv", "block1.1"),
    ("/net/block1/block1.2/layer/layer.0/Conv", "block1.2"),
    ("/net/block1/block1.3/layer/layer.0/Conv", "block1.3"),
    ("/net/skip1/skip1.1/Conv", "skip1.1"),
    ("/net/block2/block2.0/layer/layer.0/Conv", "block2.0"),
    ("/net/block2/block2.1/layer/layer.0/Conv", "block2.1"),
    ("/net/block3/block3.0/layer/layer.0/Conv", "block3.0"),
    ("/net/block3/block3.1/layer/layer.0/Conv", "block3.1"),
    ("/net/block3/block3.2/layer/layer.0/Conv", "block3.2"),
    ("/net/block4/block4.0/layer/layer.0/Conv", "block4.0"),
    ("/net/block4/block4.1/layer/layer.0/Conv", "block4.1"),
    ("/net/block4/block4.2/layer/layer.0/Conv", "block4.2"),
    ("/net/block5/block5.0/layer/layer.0/Conv", "block5.0"),
    ("/net/block5/block5.1/layer/layer.0/Conv", "block5.1"),
    ("/net/block5/block5.2/layer/layer.0/Conv", "block5.2"),
    ("/net/block5/block5.3/layer/layer.0/Conv", "block5.3"),
    (
        "/net/block_fusion/block_fusion.0/layer/layer.0/Conv",
        "block_fusion.0",
    ),
    (
        "/net/block_fusion/block_fusion.1/layer/layer.0/Conv",
        "block_fusion.1",
    ),
    ("/net/block_fusion/block_fusion.2/Conv", "block_fusion.2"),
    (
        "/net/heatmap_head/heatmap_head.0/layer/layer.0/Conv",
        "heatmap_head.0",
    ),
    (
        "/net/heatmap_head/heatmap_head.1/layer/layer.0/Conv",
        "heatmap_head.1",
    ),
    ("/net/heatmap_head/heatmap_head.2/Conv", "heatmap_head.2"),
    (
        "/net/keypoint_head/keypoint_head.0/layer/layer.0/Conv",
        "keypoint_head.0",
    ),
    (
        "/net/keypoint_head/keypoint_head.1/layer/layer.0/Conv",
        "keypoint_head.1",
    ),
    (
        "/net/keypoint_head/keypoint_head.2/layer/layer.0/Conv",
        "keypoint_head.2",
    ),
    ("/net/keypoint_head/keypoint_head.3/Conv", "keypoint_head.3"),
];

/// Convert `xfeat.onnx` to a safetensors weight buffer for [`navigate_match::XFeat`].
pub(crate) fn prepare_xfeat(bytes: &[u8], half: bool) -> Result<Vec<u8>, Error> {
    check_digest(bytes, XFEAT_SHA256)?;
    let model = ModelProto::decode(bytes)?;
    let graph = model.graph.as_ref().ok_or(Error::Invalid("model graph"))?;
    let inits = initializer_map(graph)?;
    let mut weights = Weights::default();
    for &(node_name, target) in XFEAT_CONVS {
        let node = graph
            .node
            .iter()
            .find(|n| n.name == node_name && n.op_type == "Conv")
            .ok_or(Error::Invalid("missing XFeat Conv node"))?;
        let w_name = node
            .input
            .get(1)
            .ok_or(Error::Invalid("XFeat Conv weight input"))?;
        let b_name = node
            .input
            .get(2)
            .ok_or(Error::Invalid("XFeat Conv bias input"))?;
        let w = inits
            .get(w_name.as_str())
            .ok_or(Error::Invalid("missing XFeat weight initializer"))?
            .clone();
        let b = inits
            .get(b_name.as_str())
            .ok_or(Error::Invalid("missing XFeat bias initializer"))?
            .clone();
        weights.insert(format!("{target}.weight"), w);
        weights.insert(format!("{target}.bias"), b);
    }
    tracing::info!(
        tensors = weights.len(),
        parameters = weights.parameter_count(),
        half,
        "Converted XFeat ONNX graph to burn safetensors"
    );
    Ok(weights.to_safetensors(half)?)
}

/// Convert `lighterglue.onnx` to a safetensors weight buffer for [`navigate_match::LighterGlue`].
pub(crate) fn prepare_lighterglue(bytes: &[u8], half: bool) -> Result<Vec<u8>, Error> {
    check_digest(bytes, LIGHTERGLUE_SHA256)?;
    let model = ModelProto::decode(bytes)?;
    let graph = model.graph.as_ref().ok_or(Error::Invalid("model graph"))?;
    let mut inits = initializer_map(graph)?;
    let mut weights = Weights::default();
    let named: Vec<String> = inits
        .keys()
        .filter(|k| k.starts_with("net."))
        .cloned()
        .collect();
    for key in named {
        if let Some(tensor) = inits.remove(&key) {
            weights.insert(key, tensor);
        }
    }
    let mut matmuls = vec![
        (
            "/input_proj/MatMul".to_owned(),
            "net.input_proj.weight".to_owned(),
        ),
        ("/Wr/MatMul".to_owned(), "net.posenc.Wr.weight".to_owned()),
    ];
    for layer in 0..6 {
        let s = format!("net.transformers.{layer}.self_attn");
        let c = format!("net.transformers.{layer}.cross_attn");
        let suffix2 = if layer == 0 {
            String::new()
        } else {
            format!("_{}", 2 * layer)
        };
        let suffix4 = if layer == 0 {
            String::new()
        } else {
            format!("_{}", 4 * layer)
        };
        let cross4 = format!("_{}", 4 * layer + 2);
        matmuls.extend([
            (format!("/Wqkv{suffix2}/MatMul"), format!("{s}.Wqkv.weight")),
            (
                format!("/out_proj{suffix2}/MatMul"),
                format!("{s}.out_proj.weight"),
            ),
            (
                format!("/ffn/ffn.0{suffix4}/MatMul"),
                format!("{s}.ffn.0.weight"),
            ),
            (
                format!("/ffn/ffn.3{suffix4}/MatMul"),
                format!("{s}.ffn.3.weight"),
            ),
            (
                format!("/to_qk{suffix2}/MatMul"),
                format!("{c}.to_qk.weight"),
            ),
            (format!("/to_v{suffix2}/MatMul"), format!("{c}.to_v.weight")),
            (
                format!("/to_out{suffix2}/MatMul"),
                format!("{c}.to_out.weight"),
            ),
            (
                format!("/ffn/ffn.0{cross4}/MatMul"),
                format!("{c}.ffn.0.weight"),
            ),
            (
                format!("/ffn/ffn.3{cross4}/MatMul"),
                format!("{c}.ffn.3.weight"),
            ),
        ]);
    }
    matmuls.push((
        "/final_proj/MatMul".to_owned(),
        "net.log_assignment.5.final_proj.weight".to_owned(),
    ));
    matmuls.push((
        "/matchability/MatMul".to_owned(),
        "net.log_assignment.5.matchability.weight".to_owned(),
    ));
    for (node_name, target) in matmuls {
        let node = graph
            .node
            .iter()
            .find(|n| n.name == node_name && n.op_type == "MatMul")
            .ok_or(Error::Invalid("missing LighterGlue MatMul node"))?;
        let init_name = node
            .input
            .get(1)
            .ok_or(Error::Invalid("LighterGlue MatMul weight input"))?;
        let transposed = inits
            .remove(init_name.as_str())
            .ok_or(Error::Invalid("missing LighterGlue MatMul initializer"))?;
        weights.insert(target, transpose_2d(&transposed)?);
    }
    tracing::info!(
        tensors = weights.len(),
        parameters = weights.parameter_count(),
        half,
        "Converted LighterGlue ONNX graph to burn safetensors"
    );
    Ok(weights.to_safetensors(half)?)
}

fn check_digest(bytes: &[u8], expected: &'static str) -> Result<(), Error> {
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(Error::Invalid("model size"));
    }
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(Error::Digest { actual, expected });
    }
    Ok(())
}

fn initializer_map(graph: &GraphProto) -> Result<BTreeMap<String, WeightTensor>, Error> {
    let mut map = BTreeMap::new();
    for init in &graph.initializer {
        if init.data_type != 1 {
            continue;
        }
        let shape: Vec<usize> = init
            .dims
            .iter()
            .map(|&d| usize::try_from(d).map_err(|_| Error::Invalid("negative initializer dim")))
            .collect::<Result<_, _>>()?;
        let values = decode_floats(init, &shape)?;
        map.insert(init.name.clone(), WeightTensor { shape, values });
    }
    Ok(map)
}

fn decode_floats(tensor: &TensorProto, shape: &[usize]) -> Result<Vec<f32>, Error> {
    let expected: usize = shape.iter().product();
    let values: Vec<f32> = if !tensor.float_data.is_empty() {
        tensor.float_data.clone()
    } else {
        if tensor.raw_data.len() != expected * 4 {
            return Err(Error::Invalid("initializer raw_data byte length"));
        }
        tensor
            .raw_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    };
    if values.len() != expected || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid(
            "initializer element count or non-finite value",
        ));
    }
    Ok(values)
}

fn transpose_2d(tensor: &WeightTensor) -> Result<WeightTensor, Error> {
    let [rows, cols] = tensor.shape.as_slice() else {
        return Err(Error::Invalid("MatMul initializer is not rank 2"));
    };
    let mut out = vec![0.0; rows * cols];
    for r in 0..*rows {
        for c in 0..*cols {
            out[c * rows + r] = tensor.values[r * cols + c];
        }
    }
    Ok(WeightTensor {
        shape: vec![*cols, *rows],
        values: out,
    })
}

#[cfg(test)]
mod tests;
