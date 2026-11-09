//! Lowercase hexadecimal SHA-256 digests of files and byte buffers.

use crate::PipelineError;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

/// Compute the lowercase hex SHA-256 digest of `path`.
///
/// # Errors
///
/// Returns [`PipelineError::Io`] when `path` cannot be read.
pub fn file_sha256(path: &Path) -> Result<String, PipelineError> {
    let bytes = fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(bytes_sha256(&bytes))
}

/// Compute the lowercase hex SHA-256 digest of `bytes`.
pub fn bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
