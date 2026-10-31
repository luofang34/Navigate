//! Hilbert curve sharding and disk serialization for [`super::CorridorTileStore`].

#[cfg(not(target_arch = "wasm32"))]
use super::{CORRIDOR_SHARD_MAGIC, CORRIDOR_STORE_INDEX_MAGIC};
#[cfg(not(target_arch = "wasm32"))]
use crate::MatchError;
use crate::corridor::CorridorTileKey;
#[cfg(not(target_arch = "wasm32"))]
use crate::corridor::{
    CORRIDOR_VERSION, CorridorTileEntry, MAX_TILES,
    codec::{decode_tile_shard, encode_tile_shard, ensure},
};
#[cfg(not(target_arch = "wasm32"))]
use sha2::{Digest, Sha256};
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeSet;

#[cfg(not(target_arch = "wasm32"))]
const INDEX_HEADER_BYTES: usize = 92;
#[cfg(not(target_arch = "wasm32"))]
const CHECKSUM_BYTES: usize = 32;

/// Compute a 32-bit 2D Hilbert curve index `(tile_x, tile_y) -> u64` for locality sharding.
#[must_use]
pub fn hilbert_tile_code(key: CorridorTileKey) -> u64 {
    let mut x = (i64::from(key.tile_x) + 32_768).clamp(0, 65_535) as u64;
    let mut y = (i64::from(key.tile_y) + 32_768).clamp(0, 65_535) as u64;
    let mut d = 0_u64;
    let mut s = 32_768_u64;
    while s > 0 {
        let rx = u64::from((x & s) > 0);
        let ry = u64::from((y & s) > 0);
        d += s * s * ((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 {
                x = s.saturating_mul(2).saturating_sub(1).saturating_sub(x);
                y = s.saturating_mul(2).saturating_sub(1).saturating_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

#[cfg(not(target_arch = "wasm32"))]
fn shard_path(root: &std::path::Path, key: CorridorTileKey) -> std::path::PathBuf {
    let bucket = ((hilbert_tile_code(key) >> 4) & 0xff) as u8;
    root.join("shards")
        .join(format!("h_{bucket:02x}"))
        .join(format!("tile_{}_{}.nvkt", key.tile_x, key.tile_y))
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn read_shard_file(
    root: &std::path::Path,
    key: CorridorTileKey,
) -> Result<Option<CorridorTileEntry>, MatchError> {
    let path = shard_path(root, key);
    if path.is_file() {
        let bytes = std::fs::read(&path).map_err(|e| io_err("shard read", e))?;
        Ok(Some(decode_tile_shard(CORRIDOR_SHARD_MAGIC, &bytes)?))
    } else {
        Ok(None)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn write_shard_file(
    root: &std::path::Path,
    key: CorridorTileKey,
    tile: &CorridorTileEntry,
) -> Result<(), MatchError> {
    let path = shard_path(root, key);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io_err("shard mkdir", e))?;
    }
    let bytes = encode_tile_shard(CORRIDOR_SHARD_MAGIC, tile);
    std::fs::write(&path, bytes).map_err(|e| io_err("shard write", e))
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn write_index_file(
    path: &std::path::Path,
    catalog_sha256: [u8; 32],
    encoder_contract: [u8; 32],
    seed: u64,
    tile_size_m: f32,
    keys: &BTreeSet<CorridorTileKey>,
) -> Result<(), MatchError> {
    let mut sorted: Vec<(u64, CorridorTileKey)> =
        keys.iter().map(|&k| (hilbert_tile_code(k), k)).collect();
    sorted.sort_by_key(|&(h, k)| (h, k));
    let count = sorted.len().min(MAX_TILES);
    let mut out = Vec::with_capacity(INDEX_HEADER_BYTES + count * 8 + CHECKSUM_BYTES);
    out.extend_from_slice(CORRIDOR_STORE_INDEX_MAGIC);
    out.extend_from_slice(&CORRIDOR_VERSION.to_le_bytes());
    out.extend_from_slice(&catalog_sha256);
    out.extend_from_slice(&encoder_contract);
    out.extend_from_slice(&seed.to_le_bytes());
    out.extend_from_slice(&tile_size_m.to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());
    for (_, key) in sorted.into_iter().take(count) {
        out.extend_from_slice(&key.tile_x.to_le_bytes());
        out.extend_from_slice(&key.tile_y.to_le_bytes());
    }
    let digest: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&digest);
    std::fs::write(path, out).map_err(|e| io_err("corridor store index write", e))
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) type DecodedStoreIndex = ([u8; 32], [u8; 32], u64, f32, BTreeSet<CorridorTileKey>);

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn decode_index_bytes(bytes: &[u8]) -> Result<DecodedStoreIndex, MatchError> {
    ensure(
        bytes.len() >= INDEX_HEADER_BYTES + CHECKSUM_BYTES,
        "truncated corridor store index",
    )?;
    let (payload, trailer) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
    let digest: [u8; 32] = Sha256::digest(payload).into();
    ensure(
        digest.as_slice() == trailer,
        "corridor store index checksum",
    )?;
    ensure(
        &payload[..8] == CORRIDOR_STORE_INDEX_MAGIC,
        "corridor store index magic",
    )?;
    let version = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
    ensure(version == CORRIDOR_VERSION, "corridor store index version")?;
    let mut catalog = [0_u8; 32];
    catalog.copy_from_slice(&payload[12..44]);
    let mut contract = [0_u8; 32];
    contract.copy_from_slice(&payload[44..76]);
    let seed = u64::from_le_bytes([
        payload[76],
        payload[77],
        payload[78],
        payload[79],
        payload[80],
        payload[81],
        payload[82],
        payload[83],
    ]);
    let tile_size_m = f32::from_le_bytes([payload[84], payload[85], payload[86], payload[87]]);
    let count = u32::from_le_bytes([payload[88], payload[89], payload[90], payload[91]]) as usize;
    ensure(
        tile_size_m.is_finite()
            && tile_size_m >= 10.0
            && count <= MAX_TILES
            && payload.len() == INDEX_HEADER_BYTES + count * 8,
        "corridor store index length",
    )?;
    let mut keys = BTreeSet::new();
    for i in 0..count {
        let off = INDEX_HEADER_BYTES + i * 8;
        let tx = i32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]);
        let ty = i32::from_le_bytes([
            payload[off + 4],
            payload[off + 5],
            payload[off + 6],
            payload[off + 7],
        ]);
        let _ = keys.insert(CorridorTileKey {
            tile_x: tx,
            tile_y: ty,
        });
    }
    Ok((catalog, contract, seed, tile_size_m, keys))
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn io_err(stage: &'static str, err: std::io::Error) -> MatchError {
    MatchError::Device {
        stage,
        message: err.to_string(),
    }
}
