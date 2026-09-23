use crate::{ImageryError, Tile, error::invalid};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CHUNK_LIMIT: usize = 4 * 1024 * 1024;

/// SHA-256 in lowercase hexadecimal.
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// One verified range within an immutable chunk.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Asset {
    /// Chunk content digest.
    pub chunk: String,
    /// Byte offset in the chunk.
    pub offset: usize,
    /// Byte length of the encoded tile.
    pub length: usize,
    /// Encoded tile content digest.
    pub sha256: String,
}

/// Reference assets at one map tile address.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TileRecord {
    /// Web Mercator address.
    pub xyz: Tile,
    /// Imagery bytes, including their validity mask.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub imagery: Option<Asset>,
    /// Terrarium elevation bytes, including their validity mask.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elevation: Option<Asset>,
}

/// A content-addressed download object.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Chunk {
    /// Content digest.
    pub sha256: String,
    /// Encoded byte length.
    pub size: usize,
    /// Service-relative object URL.
    pub url: String,
}

/// Offline reference data consumed by native and WASM readers.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Package {
    /// Storage manifest version.
    pub schema_version: u32,
    /// Stable catalogue identity.
    pub region_id: String,
    /// Data release identity.
    pub release_id: String,
    /// Local tangent-plane anchor in latitude, longitude order.
    pub anchor_lat_lon: [f64; 2],
    /// Source elevation datum, including an explicit unknown value.
    pub elevation_datum: String,
    /// Source attribution.
    pub attribution: String,
    /// Required immutable files.
    pub files: Vec<Chunk>,
    /// Tile asset ranges.
    pub tiles: Vec<TileRecord>,
    /// Source identities and processing information. Unknown uncertainty stays unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<serde_json::Value>,
    /// Manifest digest, excluding this field.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub pack_id: String,
}

/// Incremental chunk construction with an application-supplied storage writer.
pub struct PackageBuilder<F> {
    manifest: Package,
    body: Vec<u8>,
    pending: Vec<(usize, bool)>,
    write: F,
}

impl<F: FnMut(&str, &[u8]) -> Result<(), ImageryError>> PackageBuilder<F> {
    /// Start an empty package with caller-supplied source metadata.
    ///
    /// # Errors
    /// Rejects a populated manifest or an invalid anchor.
    pub fn new(manifest: Package, write: F) -> Result<Self, ImageryError> {
        if manifest.schema_version != 1
            || !manifest.tiles.is_empty()
            || !manifest.files.is_empty()
            || !manifest.pack_id.is_empty()
            || manifest.anchor_lat_lon.iter().any(|v| !v.is_finite())
            || manifest.anchor_lat_lon[0].abs() > 85.0
            || manifest.anchor_lat_lon[1].abs() > 180.0
        {
            return Err(invalid(
                "package needs an empty schema-1 manifest and a valid anchor",
            ));
        }
        Ok(Self {
            manifest,
            write,
            body: Vec::new(),
            pending: Vec::new(),
        })
    }

    /// Add imagery or elevation bytes at one address.
    ///
    /// # Errors
    /// Rejects invalid tile addresses, duplicate roles, oversized assets, or checksum errors.
    /// Propagates storage failures. Callers must publish only after `finish` succeeds.
    pub fn add(
        &mut self,
        xyz: Tile,
        elevation: bool,
        bytes: &[u8],
        sha: &str,
    ) -> Result<(), ImageryError> {
        if xyz.0 > 24
            || xyz.1 >= (1 << xyz.0)
            || xyz.2 >= (1 << xyz.0)
            || bytes.is_empty()
            || bytes.len() > CHUNK_LIMIT
            || digest(bytes) != sha
        {
            return Err(invalid(format!(
                "invalid tile bytes or checksum at {xyz:?}"
            )));
        }
        if self.body.len() + bytes.len() > CHUNK_LIMIT {
            self.flush()?;
        }
        let index = match self.manifest.tiles.iter().position(|t| t.xyz == xyz) {
            Some(index) => index,
            None => {
                self.manifest.tiles.push(TileRecord {
                    xyz,
                    imagery: None,
                    elevation: None,
                });
                self.manifest.tiles.len() - 1
            }
        };
        let record = &mut self.manifest.tiles[index];
        let slot = if elevation {
            &mut record.elevation
        } else {
            &mut record.imagery
        };
        if slot.is_some() {
            return Err(invalid(format!("duplicate tile role at {xyz:?}")));
        }
        *slot = Some(Asset {
            chunk: String::new(),
            offset: self.body.len(),
            length: bytes.len(),
            sha256: sha.into(),
        });
        self.pending.push((index, elevation));
        self.body.extend_from_slice(bytes);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ImageryError> {
        if self.body.is_empty() {
            return Ok(());
        }
        let sha = digest(&self.body);
        (self.write)(&sha, &self.body)?;
        if !self.manifest.files.iter().any(|f| f.sha256 == sha) {
            self.manifest.files.push(Chunk {
                sha256: sha.clone(),
                size: self.body.len(),
                url: format!("/chunks/{sha}.bin"),
            });
        }
        for (i, elevation) in self.pending.drain(..) {
            let tile = &mut self.manifest.tiles[i];
            if let Some(a) = if elevation {
                &mut tile.elevation
            } else {
                &mut tile.imagery
            } {
                a.chunk = sha.clone();
            }
        }
        self.body.clear();
        Ok(())
    }

    /// Finish all chunks and calculate the versioned manifest identity.
    ///
    /// # Errors
    /// Rejects packages without imagery or terrain. Propagates storage errors.
    pub fn finish(mut self) -> Result<Package, ImageryError> {
        self.flush()?;
        if !self.manifest.tiles.iter().any(|t| t.imagery.is_some())
            || !self.manifest.tiles.iter().any(|t| t.elevation.is_some())
        {
            return Err(invalid(
                "a localization package requires imagery and terrain",
            ));
        }
        self.manifest.tiles.sort_by_key(|t| t.xyz);
        self.manifest.pack_id =
            digest(&serde_json::to_vec(&serde_json::to_value(&self.manifest)?)?);
        Ok(self.manifest)
    }
}

#[cfg(test)]
mod tests;
