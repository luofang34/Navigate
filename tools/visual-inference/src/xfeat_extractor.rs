//! [`SparseFeatureExtractor`] adapter backed by [`OrtModelSession`].

use crate::{ExecutionConfig, InferenceError, OrtModelSession, Provider};
use navigate_match::{ModelSessionExtractor, SparseFeatureExtractor, Weights, XFeat};
use std::path::Path;

/// Sparse XFeat extractor backed by an ONNX Runtime (`CoreML`, `TensorRT`, `CUDA`, or `CPU`) session.
pub type OrtXFeatExtractor = ModelSessionExtractor<OrtModelSession>;

/// Return a stable runtime label for `provider`.
#[must_use]
pub fn provider_runtime_name(provider: Provider) -> &'static str {
    match provider {
        Provider::Cpu => "onnx-cpu",
        Provider::CoreMlGpu => "coreml-gpu",
        Provider::CoreMlAne => "coreml",
        Provider::Cuda => "cuda",
        Provider::TensorRt => "tensorrt",
    }
}

/// Load a dense XFeat `.onnx` export and bind the canonical `encoder_contract` from `safetensors_path`.
///
/// The returned [`OrtXFeatExtractor`] implements [`SparseFeatureExtractor`] and produces
/// descriptors and stamps compatible with Burn [`XFeat`] banks built from `safetensors_path`.
///
/// # Errors
///
/// Returns [`InferenceError`] when reading `safetensors_path` or creating the ONNX session fails.
pub fn load_xfeat_extractor_blocking(
    onnx_path: &Path,
    safetensors_path: &Path,
    config: &ExecutionConfig,
) -> Result<OrtXFeatExtractor, InferenceError> {
    let bytes = std::fs::read(safetensors_path).map_err(|source| InferenceError::Io {
        path: safetensors_path.to_path_buf(),
        source,
    })?;
    let weights = Weights::from_safetensors(&bytes)
        .map_err(|e| InferenceError::Invalid(format!("load XFeat safetensors contract: {e}")))?;
    let contract = XFeat::encoder_contract_of(&weights);
    let session = OrtModelSession::load_blocking(onnx_path, config)?;
    let input_name = session.primary_input_name()?.to_owned();
    let extractor = ModelSessionExtractor::new(
        session,
        input_name,
        contract,
        provider_runtime_name(config.provider),
    );
    let _ = extractor.encoder_contract();
    Ok(extractor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_names_match_cli_runtime_identifiers() {
        assert_eq!(provider_runtime_name(Provider::CoreMlAne), "coreml");
        assert_eq!(provider_runtime_name(Provider::TensorRt), "tensorrt");
        assert_eq!(provider_runtime_name(Provider::Cpu), "onnx-cpu");
    }
}
