//! Verify the same immutable bytes consumed by the browser preview.
use crate::{BenchError, package::digest, read_blocking};
use navigate_data::{DataStore, DataUri};
use navigate_data_fs::FileStore;
use navigate_imagery::{Asset, Package};
use std::{collections::BTreeMap, path::Path};
fn invalid(reason: impl Into<String>) -> BenchError {
    BenchError::Package {
        reason: reason.into(),
    }
}
pub(crate) async fn verify_blocking(path: &Path, root: &Path) -> Result<(), BenchError> {
    let manifest: Package =
        serde_json::from_slice(&read_blocking(path)?).map_err(|source| BenchError::Json {
            path: path.into(),
            source,
        })?;
    manifest
        .validate_for_reading()
        .map_err(|source| invalid(source.to_string()))?;
    if manifest.files.len() > 128 {
        return Err(invalid("invalid offline package dimensions"));
    }
    let store = FileStore::new(root).await?;
    let mut chunks = BTreeMap::new();
    for chunk in &manifest.files {
        if chunks.contains_key(&chunk.sha256) {
            return Err(invalid("invalid or duplicate offline chunk"));
        }
        let uri = DataUri::parse(format!("pilotage://chunks/{}.bin", chunk.sha256))?;
        let file = store.open(&uri).await?;
        if usize::try_from(file.len()).ok() != Some(chunk.size) {
            return Err(invalid(format!("chunk size mismatch: {}", uri.as_str())));
        }
        let bytes = file.read_at(0, chunk.size).await?;
        if digest(&bytes) != chunk.sha256 {
            return Err(invalid(format!(
                "chunk checksum mismatch: {}",
                uri.as_str()
            )));
        }
        chunks.insert(chunk.sha256.clone(), bytes);
    }
    for tile in &manifest.tiles {
        for asset in [&tile.imagery, &tile.elevation].into_iter().flatten() {
            verify_asset(asset, &chunks)?;
        }
    }
    tracing::info!(
        chunks = manifest.files.len(),
        tiles = manifest.tiles.len(),
        "offline package ranges verified"
    );
    Ok(())
}
fn verify_asset(asset: &Asset, chunks: &BTreeMap<String, Vec<u8>>) -> Result<(), BenchError> {
    let bytes = chunks
        .get(&asset.chunk)
        .ok_or_else(|| invalid("asset references an absent chunk"))?;
    navigate_imagery::verify_asset(asset, bytes).map_err(|source| invalid(source.to_string()))
}
#[cfg(test)]
mod tests;
