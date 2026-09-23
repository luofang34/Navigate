//! Verify the same immutable bytes consumed by the browser preview.
use crate::{BenchError, package::digest, read_blocking};
use navigate_data::{DataStore, DataUri};
use navigate_data_fs::FileStore;
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};
#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    files: Vec<Chunk>,
    tiles: Vec<Tile>,
}
#[derive(Deserialize)]
struct Chunk {
    sha256: String,
    size: u64,
}
#[derive(Deserialize)]
struct Tile {
    imagery: Option<Asset>,
    elevation: Option<Asset>,
}
#[derive(Deserialize)]
struct Asset {
    chunk: String,
    offset: u64,
    length: usize,
    sha256: String,
}
fn invalid(reason: impl Into<String>) -> BenchError {
    BenchError::Package {
        reason: reason.into(),
    }
}
fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
pub(crate) async fn verify_blocking(path: &Path, root: &Path) -> Result<(), BenchError> {
    let manifest: Manifest =
        serde_json::from_slice(&read_blocking(path)?).map_err(|source| BenchError::Json {
            path: path.into(),
            source,
        })?;
    if manifest.schema_version != 1
        || manifest.files.is_empty()
        || manifest.files.len() > 128
        || manifest.tiles.is_empty()
        || manifest.tiles.len() > 1024
    {
        return Err(invalid("invalid offline package dimensions"));
    }
    let store = FileStore::new(root).await?;
    let mut chunks = BTreeMap::new();
    for chunk in &manifest.files {
        if !hash(&chunk.sha256)
            || chunk.size > 8 * 1024 * 1024
            || chunks.contains_key(&chunk.sha256)
        {
            return Err(invalid("invalid or duplicate offline chunk"));
        }
        let uri = DataUri::parse(format!("pilotage://chunks/{}.bin", chunk.sha256))?;
        let file = store.open(&uri).await?;
        if file.len() != chunk.size {
            return Err(invalid(format!("chunk size mismatch: {}", uri.as_str())));
        }
        let bytes = file.read_at(0, chunk.size as usize).await?;
        if digest(&bytes) != chunk.sha256 {
            return Err(invalid(format!(
                "chunk checksum mismatch: {}",
                uri.as_str()
            )));
        }
        chunks.insert(chunk.sha256.clone(), bytes);
    }
    for tile in &manifest.tiles {
        if tile.imagery.is_none() && tile.elevation.is_none() {
            return Err(invalid("tile has no data"));
        }
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
    let offset =
        usize::try_from(asset.offset).map_err(|_| invalid("asset offset exceeds address space"))?;
    let end = offset
        .checked_add(asset.length)
        .ok_or_else(|| invalid("asset range overflows"))?;
    let body = bytes
        .get(offset..end)
        .ok_or_else(|| invalid("asset range exceeds chunk"))?;
    if !hash(&asset.sha256) || digest(body) != asset.sha256 {
        return Err(invalid("asset checksum mismatch"));
    }
    Ok(())
}
#[cfg(test)]
mod tests;
