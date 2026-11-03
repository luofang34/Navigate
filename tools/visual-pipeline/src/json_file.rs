//! Deterministic JSON file reads and writes.

use crate::PipelineError;
use serde::{Serialize, de::DeserializeOwned};
use std::{fs, path::Path};

/// Read and deserialize a JSON file from `path`.
///
/// # Errors
///
/// Returns [`PipelineError`] when reading or JSON parsing fails.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, PipelineError> {
    let text = fs::read_to_string(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| PipelineError::Json {
        context: path.display().to_string(),
        source,
    })
}

/// Serialize `value` as 2-space pretty JSON with a trailing newline to `path`.
///
/// # Errors
///
/// Returns [`PipelineError`] when serialization or file write fails.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), PipelineError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut text = serde_json::to_string_pretty(value).map_err(|source| PipelineError::Json {
        context: path.display().to_string(),
        source,
    })?;
    text.push('\n');
    fs::write(path, text).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
}
