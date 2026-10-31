//! LoFTR training checkpoint to the weight file of `navigate-match`.
use crate::error::Error;
use navigate_match::{WeightTensor, Weights, loftr_from_checkpoint};
use pytorch_reader::{DType, PytorchReader};
use sha2::{Digest, Sha256};
use std::path::Path;

/// SHA-256 of `loftr_outdoor.ckpt`, as `tools/visual-model/model-assets/loftr.json` records.
const CHECKPOINT_SHA256: &str = "21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec";

/// Fold the batch normalizations and write safetensors, in F16 when `half` is set.
pub(crate) fn prepare(path: &Path, bytes: &[u8], half: bool) -> Result<Vec<u8>, Error> {
    let actual = hex(&Sha256::digest(bytes));
    if actual != CHECKPOINT_SHA256 {
        return Err(Error::Digest {
            actual,
            expected: CHECKPOINT_SHA256,
        });
    }
    // The zip reader needs the file; the digest above covers the same bytes.
    let reader =
        PytorchReader::with_top_level_key(path, "state_dict").map_err(Error::Checkpoint)?;
    let mut checkpoint = Weights::default();
    for (name, tensor) in reader.tensors() {
        if name.ends_with("num_batches_tracked") {
            continue;
        }
        if tensor.dtype() != DType::F32 {
            return Err(Error::Invalid("checkpoint tensors must be F32"));
        }
        let raw = tensor.read().map_err(|source| Error::CheckpointTensor {
            name: name.clone(),
            source,
        })?;
        let values = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        checkpoint.insert(
            name.clone(),
            WeightTensor {
                shape: tensor.shape().to_vec(),
                values,
            },
        );
    }
    let weights = loftr_from_checkpoint(checkpoint)?;
    tracing::info!(
        tensors = weights.len(),
        parameters = weights.parameter_count(),
        half,
        "Converted LoFTR checkpoint"
    );
    Ok(weights.to_safetensors(half)?)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests;
