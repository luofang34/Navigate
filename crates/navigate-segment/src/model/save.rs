//! HuggingFace SafeTensors weight writer for [`super::Segformer`].

use super::Segformer;
use crate::{
    SegmentError,
    config::{PreprocessorConfig, SegformerConfig},
};
use burn::{
    nn::{LayerNorm, Linear, conv::Conv2d},
    tensor::Tensor,
};
use safetensors::{Dtype, tensor::TensorView};
use std::{collections::BTreeMap, fs, path::Path};

struct OwnedEntry {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn tensor_vec<const D: usize>(t: Tensor<D>) -> Result<(Vec<usize>, Vec<f32>), SegmentError> {
    let shape = t.dims().to_vec();
    let values = t
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| SegmentError::Safetensors(format!("{e:?}")))?;
    Ok((shape, values))
}

fn put_f32(map: &mut BTreeMap<String, OwnedEntry>, key: String, shape: Vec<usize>, vals: &[f32]) {
    map.insert(
        key,
        OwnedEntry {
            dtype: Dtype::F32,
            shape,
            bytes: f32_bytes(vals),
        },
    );
}

fn put_linear(
    map: &mut BTreeMap<String, OwnedEntry>,
    prefix: &str,
    linear: &Linear,
) -> Result<(), SegmentError> {
    let (shape, row_vals) = tensor_vec(linear.weight.val())?;
    let (d_in, d_out) = (shape[0], shape[1]);
    let mut col_vals = vec![0.0_f32; d_out * d_in];
    for i in 0..d_in {
        for o in 0..d_out {
            col_vals[o * d_in + i] = row_vals[i * d_out + o];
        }
    }
    put_f32(
        map,
        format!("{prefix}.weight"),
        vec![d_out, d_in],
        &col_vals,
    );
    if let Some(bias) = &linear.bias {
        let (bshape, bvals) = tensor_vec(bias.val())?;
        put_f32(map, format!("{prefix}.bias"), bshape, &bvals);
    }
    Ok(())
}

fn put_ln(
    map: &mut BTreeMap<String, OwnedEntry>,
    prefix: &str,
    ln: &LayerNorm,
) -> Result<(), SegmentError> {
    let (wshape, wvals) = tensor_vec(ln.gamma.val())?;
    put_f32(map, format!("{prefix}.weight"), wshape, &wvals);
    if let Some(beta) = &ln.beta {
        let (bshape, bvals) = tensor_vec(beta.val())?;
        put_f32(map, format!("{prefix}.bias"), bshape, &bvals);
    }
    Ok(())
}

fn put_conv(
    map: &mut BTreeMap<String, OwnedEntry>,
    prefix: &str,
    conv: &Conv2d,
) -> Result<(), SegmentError> {
    let (wshape, wvals) = tensor_vec(conv.weight.val())?;
    put_f32(map, format!("{prefix}.weight"), wshape, &wvals);
    if let Some(bias) = &conv.bias {
        let (bshape, bvals) = tensor_vec(bias.val())?;
        put_f32(map, format!("{prefix}.bias"), bshape, &bvals);
    }
    Ok(())
}

/// Save `model`, `config`, and `preprocessor` in HuggingFace `from_pretrained` layout at `dir`.
pub(super) fn save_dir(
    model: &Segformer,
    config: &SegformerConfig,
    preprocessor: &PreprocessorConfig,
    dir: &Path,
) -> Result<(), SegmentError> {
    fs::create_dir_all(dir).map_err(|source| SegmentError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let mut map = BTreeMap::new();
    for (s, stage) in model.encoder.stages.iter().enumerate() {
        let pe = format!("segformer.encoder.patch_embeddings.{s}");
        put_conv(&mut map, &format!("{pe}.proj"), &stage.patch_embed.proj)?;
        put_ln(
            &mut map,
            &format!("{pe}.layer_norm"),
            &stage.patch_embed.layer_norm,
        )?;
        for (b, block) in stage.blocks.iter().enumerate() {
            let bp = format!("segformer.encoder.block.{s}.{b}");
            put_ln(&mut map, &format!("{bp}.layer_norm_1"), &block.layer_norm_1)?;
            put_linear(
                &mut map,
                &format!("{bp}.attention.self.query"),
                &block.attention.query,
            )?;
            put_linear(
                &mut map,
                &format!("{bp}.attention.self.key"),
                &block.attention.key,
            )?;
            put_linear(
                &mut map,
                &format!("{bp}.attention.self.value"),
                &block.attention.value,
            )?;
            put_linear(
                &mut map,
                &format!("{bp}.attention.output.dense"),
                &block.attention.dense,
            )?;
            if let Some(red) = &block.attention.reduction {
                put_conv(&mut map, &format!("{bp}.attention.self.sr"), &red.sr)?;
                put_ln(
                    &mut map,
                    &format!("{bp}.attention.self.layer_norm"),
                    &red.layer_norm,
                )?;
            }
            put_ln(&mut map, &format!("{bp}.layer_norm_2"), &block.layer_norm_2)?;
            put_linear(&mut map, &format!("{bp}.mlp.dense1"), &block.mlp.dense1)?;
            put_conv(
                &mut map,
                &format!("{bp}.mlp.dwconv.dwconv"),
                &block.mlp.dwconv,
            )?;
            put_linear(&mut map, &format!("{bp}.mlp.dense2"), &block.mlp.dense2)?;
        }
        put_ln(
            &mut map,
            &format!("segformer.encoder.layer_norm.{s}"),
            &stage.layer_norm,
        )?;
    }

    for (s, proj) in model.decode_head.linear_c.iter().enumerate() {
        put_linear(&mut map, &format!("decode_head.linear_c.{s}.proj"), proj)?;
    }
    put_conv(
        &mut map,
        "decode_head.linear_fuse",
        &model.decode_head.linear_fuse,
    )?;
    let (bn_ws, bn_wv) = tensor_vec(model.decode_head.batch_norm.gamma.val())?;
    let (bn_bs, bn_bv) = tensor_vec(model.decode_head.batch_norm.beta.val())?;
    let (bn_ms, bn_mv) = tensor_vec(model.decode_head.batch_norm.running_mean.value())?;
    let (bn_vs, bn_vv) = tensor_vec(model.decode_head.batch_norm.running_var.value())?;
    put_f32(
        &mut map,
        "decode_head.batch_norm.weight".into(),
        bn_ws,
        &bn_wv,
    );
    put_f32(
        &mut map,
        "decode_head.batch_norm.bias".into(),
        bn_bs,
        &bn_bv,
    );
    put_f32(
        &mut map,
        "decode_head.batch_norm.running_mean".into(),
        bn_ms,
        &bn_mv,
    );
    put_f32(
        &mut map,
        "decode_head.batch_norm.running_var".into(),
        bn_vs,
        &bn_vv,
    );
    map.insert(
        "decode_head.batch_norm.num_batches_tracked".into(),
        OwnedEntry {
            dtype: Dtype::I64,
            shape: Vec::new(),
            bytes: 0_i64.to_le_bytes().to_vec(),
        },
    );
    put_conv(
        &mut map,
        "decode_head.classifier",
        &model.decode_head.classifier,
    )?;

    let mut views = Vec::with_capacity(map.len());
    for (name, entry) in &map {
        let view = TensorView::new(entry.dtype, entry.shape.clone(), &entry.bytes)
            .map_err(|e| SegmentError::Safetensors(e.to_string()))?;
        views.push((name.as_str(), view));
    }
    let bytes = safetensors::tensor::serialize(views, None)
        .map_err(|e| SegmentError::Safetensors(e.to_string()))?;
    let weight_path = dir.join("model.safetensors");
    fs::write(&weight_path, bytes).map_err(|source| SegmentError::Io {
        path: weight_path,
        source,
    })?;

    write_pretty_json(&dir.join("config.json"), config)?;
    write_pretty_json(&dir.join("preprocessor_config.json"), preprocessor)?;
    Ok(())
}

fn write_pretty_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), SegmentError> {
    let mut text = serde_json::to_string_pretty(value).map_err(|source| SegmentError::Json {
        context: path.display().to_string(),
        source,
    })?;
    text.push('\n');
    fs::write(path, text).map_err(|source| SegmentError::Io {
        path: path.to_path_buf(),
        source,
    })
}
