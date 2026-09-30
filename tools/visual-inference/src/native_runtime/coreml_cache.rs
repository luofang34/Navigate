use super::{ExecutionConfig, InferenceError, Provider};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub(super) struct CachedModel {
    pub bytes: Vec<u8>,
    pub directory: PathBuf,
}

pub(super) fn prepare_blocking(
    path: &Path,
    config: &ExecutionConfig,
    dimensions: &[(String, i64)],
) -> Result<Option<CachedModel>, InferenceError> {
    let Some(root) = &config.coreml_cache_directory else {
        return Ok(None);
    };
    if !matches!(config.provider, Provider::CoreMlGpu | Provider::CoreMlAne) {
        return Ok(None);
    }
    let bytes = std::fs::read(path).map_err(|source| InferenceError::Io {
        path: path.to_owned(),
        source,
    })?;
    let directory = root.join(cache_key(&bytes, config, dimensions, ort::info()));
    std::fs::create_dir_all(&directory).map_err(|source| InferenceError::Cache {
        path: directory.clone(),
        source,
    })?;
    // Load this snapshot in memory: hashing a path and reopening it permits weight changes.
    // ORT cannot resolve external weight files without a model path.
    Ok(Some(CachedModel { bytes, directory }))
}

fn cache_key(
    bytes: &[u8],
    config: &ExecutionConfig,
    dimensions: &[(String, i64)],
    runtime: &str,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"navigate-coreml-mlprogram-static-cache-v1");
    hash.update(Sha256::digest(bytes));
    hash.update(format!(
        "{:?}/{}/{}/{dimensions:?}/{runtime}",
        config.provider, config.threads, config.cpu_spinning,
    ));
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests;
