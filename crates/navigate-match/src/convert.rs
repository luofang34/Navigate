//! Conversion of the LoFTR training checkpoint to a Navigate weight file.
//!
//! The checkpoint keeps each batch normalization apart from its convolution. Inference
//! uses fixed statistics, so the converter folds each normalization into the
//! convolution before it. The folded convolution gets a bias. Other tensors keep
//! their checkpoint names.
use crate::{MatchError, Weights, weights::WeightTensor};

/// PyTorch `BatchNorm2d` default epsilon, which the checkpoint used in training.
const BATCH_NORM_EPSILON: f32 = 1e-5;

/// Convert the `state_dict` of `loftr_outdoor.ckpt` to the weights of [`crate::Loftr`].
///
/// The caller leaves out the integer `num_batches_tracked` counters.
///
/// # Errors
///
/// Returns an error when a convolution or a normalization tensor is missing or has
/// an unexpected shape.
pub fn loftr_from_checkpoint(mut checkpoint: Weights) -> Result<Weights, MatchError> {
    let mut out = Weights::default();
    for (convolution, normalization) in folded_pairs() {
        let weight = take(&mut checkpoint, &format!("{convolution}.weight"))?;
        let channels = weight.shape.first().copied().unwrap_or(0);
        let gamma = take_vector(&mut checkpoint, &normalization, "weight", channels)?;
        let beta = take_vector(&mut checkpoint, &normalization, "bias", channels)?;
        let mean = take_vector(&mut checkpoint, &normalization, "running_mean", channels)?;
        let var = take_vector(&mut checkpoint, &normalization, "running_var", channels)?;
        let (weight, bias) = fold(weight, &gamma, &beta, &mean, &var);
        out.insert(format!("{convolution}.weight"), weight);
        out.insert(format!("{convolution}.bias"), bias);
    }
    for (name, tensor) in checkpoint.drain() {
        out.insert(name, tensor);
    }
    Ok(out)
}

/// Convolution and normalization module names, in backbone order.
fn folded_pairs() -> Vec<(String, String)> {
    let mut pairs = vec![("backbone.conv1".to_owned(), "backbone.bn1".to_owned())];
    for layer in 1..=3 {
        for block in 0..2 {
            let prefix = format!("backbone.layer{layer}.{block}");
            pairs.push((format!("{prefix}.conv1"), format!("{prefix}.bn1")));
            pairs.push((format!("{prefix}.conv2"), format!("{prefix}.bn2")));
            if layer > 1 && block == 0 {
                pairs.push((
                    format!("{prefix}.downsample.0"),
                    format!("{prefix}.downsample.1"),
                ));
            }
        }
    }
    for level in [2, 1] {
        let prefix = format!("backbone.layer{level}_outconv2");
        pairs.push((format!("{prefix}.0"), format!("{prefix}.1")));
    }
    pairs
}

fn take(weights: &mut Weights, name: &str) -> Result<WeightTensor, MatchError> {
    weights
        .remove(name)
        .ok_or_else(|| MatchError::MissingWeight { name: name.into() })
}

fn take_vector(
    weights: &mut Weights,
    module: &str,
    field: &str,
    channels: usize,
) -> Result<Vec<f32>, MatchError> {
    let name = format!("{module}.{field}");
    let tensor = take(weights, &name)?;
    if tensor.shape != [channels] {
        return Err(MatchError::WeightShape {
            name,
            actual: tensor.shape,
            expected: vec![channels],
        });
    }
    Ok(tensor.values)
}

/// Scale each output channel of `weight` and make the bias of the normalization.
fn fold(
    mut weight: WeightTensor,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
) -> (WeightTensor, WeightTensor) {
    let per_channel = weight.values.len() / gamma.len().max(1);
    let mut bias = Vec::with_capacity(gamma.len());
    for (channel, values) in weight.values.chunks_mut(per_channel.max(1)).enumerate() {
        let scale = gamma[channel] / (var[channel] + BATCH_NORM_EPSILON).sqrt();
        values.iter_mut().for_each(|v| *v *= scale);
        bias.push(beta[channel] - mean[channel] * scale);
    }
    let channels = gamma.len();
    (
        weight,
        WeightTensor {
            shape: vec![channels],
            values: bias,
        },
    )
}

#[cfg(test)]
mod tests;
