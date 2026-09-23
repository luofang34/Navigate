//! Checks that every reader applies before it trusts a package or an asset.

use crate::{Asset, ImageryError, Package, Tile, digest, error::invalid};

const MAX_TILES: usize = 1024;
const MAX_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const MAX_ZOOM: u32 = 22;

/// Whether a value is a SHA-256 digest in lowercase hexadecimal.
pub fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

impl Package {
    /// Check the manifest bounds and identities before a reader opens any chunk.
    ///
    /// This check does not read chunk bytes. Use [`verify_asset`] for each
    /// asset after its chunk digest is verified.
    ///
    /// # Errors
    /// Rejects an unknown schema, invalid identities or anchor, empty or
    /// oversized contents, duplicate tile addresses, and empty tiles.
    pub fn validate_for_reading(&self) -> Result<(), ImageryError> {
        let [lat, lon] = self.anchor_lat_lon;
        let anchor = lat.is_finite() && lon.is_finite() && lat.abs() <= 85.0 && lon.abs() <= 180.0;
        let identities = is_digest(&self.pack_id)
            && !self.release_id.is_empty()
            && self.supersedes.as_deref().is_none_or(is_digest);
        let files = !self.files.is_empty()
            && self
                .files
                .iter()
                .all(|c| c.size <= MAX_CHUNK_BYTES && is_digest(&c.sha256));
        let mut seen = std::collections::BTreeSet::new();
        let tiles = !self.tiles.is_empty()
            && self.tiles.len() <= MAX_TILES
            && self.tiles.iter().all(|t| {
                let Tile(z, x, y) = t.xyz;
                z <= MAX_ZOOM
                    && x < 1 << z
                    && y < 1 << z
                    && seen.insert(t.xyz)
                    && (t.imagery.is_some() || t.elevation.is_some())
            });
        if self.schema_version != 1 || !anchor || !identities || !files || !tiles {
            return Err(invalid("invalid or oversized offline manifest"));
        }
        Ok(())
    }
}

/// Check one asset range against the verified bytes of its chunk.
///
/// # Errors
/// Rejects a range outside the chunk and a digest mismatch.
pub fn verify_asset(asset: &Asset, chunk: &[u8]) -> Result<(), ImageryError> {
    let body = asset
        .offset
        .checked_add(asset.length)
        .and_then(|end| chunk.get(asset.offset..end))
        .ok_or_else(|| invalid("asset range exceeds chunk"))?;
    if !is_digest(&asset.sha256) || digest(body) != asset.sha256 {
        return Err(invalid("asset checksum mismatch"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
