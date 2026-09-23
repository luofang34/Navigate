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
    /// Release that produced this asset, when it is not the package release.
    ///
    /// A derived package keeps the assets of its parent. Each kept asset names
    /// the earlier release, so a reader can see which assets new evidence changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produced_by: Option<String>,
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
    /// `pack_id` of the package that this package replaces in the same region.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<String>,
    /// Manifest digest, excluding this field.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub pack_id: String,
}

impl Package {
    /// Digest of the manifest with an empty `pack_id`.
    ///
    /// # Errors
    /// Propagates serialization errors.
    pub fn compute_pack_id(&self) -> Result<String, ImageryError> {
        let mut unsigned = self.clone();
        unsigned.pack_id.clear();
        Ok(digest(&serde_json::to_vec(&serde_json::to_value(
            &unsigned,
        )?)?))
    }
}

/// Incremental chunk construction with an application-supplied storage writer.
pub struct PackageBuilder<F> {
    manifest: Package,
    body: Vec<u8>,
    pending: Vec<(usize, bool)>,
    inherited: std::collections::BTreeSet<(Tile, bool)>,
    write: F,
}

/// The writer's error type `E` carries storage failures. Package rule
/// violations reach the caller as `E::from(ImageryError)`.
impl<F, E> PackageBuilder<F>
where
    F: FnMut(&str, &[u8]) -> Result<(), E>,
    E: From<ImageryError>,
{
    /// Start an empty package with caller-supplied source metadata.
    ///
    /// # Errors
    /// Rejects a populated manifest or an invalid anchor.
    pub fn new(manifest: Package, write: F) -> Result<Self, E> {
        if manifest.schema_version != 1
            || !manifest.tiles.is_empty()
            || !manifest.files.is_empty()
            || !manifest.pack_id.is_empty()
            || manifest.anchor_lat_lon.iter().any(|v| !v.is_finite())
            || manifest.anchor_lat_lon[0].abs() > 85.0
            || manifest.anchor_lat_lon[1].abs() > 180.0
        {
            return Err(invalid_as(
                "package needs an empty schema-1 manifest and a valid anchor",
            ));
        }
        Ok(Self {
            manifest,
            write,
            body: Vec::new(),
            pending: Vec::new(),
            inherited: std::collections::BTreeSet::new(),
        })
    }

    /// Start a package that keeps the tiles of `parent` and replaces some of them.
    ///
    /// The new package has the region of the parent, the new `release_id`,
    /// and `supersedes` set to the parent `pack_id`. The parent chunks are
    /// content-addressed, so kept tiles are not written again. [`Self::add`]
    /// may replace each kept tile role once, for example with imagery or
    /// elevation that new flights refined.
    ///
    /// # Errors
    /// Rejects a parent whose `pack_id` does not match its manifest, and an
    /// empty or unchanged release identity.
    pub fn from_parent(parent: &Package, release_id: String, write: F) -> Result<Self, E> {
        if parent.pack_id.is_empty() || parent.compute_pack_id()? != parent.pack_id {
            return Err(invalid_as("parent pack_id does not match its manifest"));
        }
        if release_id.is_empty() || release_id == parent.release_id {
            return Err(invalid_as("a derived package needs a new release identity"));
        }
        let mut manifest = parent.clone();
        manifest.supersedes = Some(parent.pack_id.clone());
        manifest.pack_id.clear();
        let mut inherited = std::collections::BTreeSet::new();
        for tile in &mut manifest.tiles {
            for (elevation, asset) in [(false, &mut tile.imagery), (true, &mut tile.elevation)] {
                if let Some(asset) = asset {
                    asset
                        .produced_by
                        .get_or_insert_with(|| parent.release_id.clone());
                    inherited.insert((tile.xyz, elevation));
                }
            }
        }
        manifest.release_id = release_id;
        Ok(Self {
            manifest,
            write,
            body: Vec::new(),
            pending: Vec::new(),
            inherited,
        })
    }

    /// Add imagery or elevation bytes at one address.
    ///
    /// # Errors
    /// Rejects invalid tile addresses, duplicate roles, oversized assets, or checksum errors.
    /// Propagates storage failures. Callers must publish only after `finish` succeeds.
    pub fn add(&mut self, xyz: Tile, elevation: bool, bytes: &[u8], sha: &str) -> Result<(), E> {
        if xyz.0 > 24
            || xyz.1 >= (1 << xyz.0)
            || xyz.2 >= (1 << xyz.0)
            || bytes.is_empty()
            || bytes.len() > CHUNK_LIMIT
            || digest(bytes) != sha
        {
            return Err(invalid_as(format!(
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
        if slot.is_some() && !self.inherited.remove(&(xyz, elevation)) {
            return Err(invalid_as(format!("duplicate tile role at {xyz:?}")));
        }

        *slot = Some(Asset {
            chunk: String::new(),
            offset: self.body.len(),
            length: bytes.len(),
            sha256: sha.into(),
            produced_by: None,
        });
        self.pending.push((index, elevation));
        self.body.extend_from_slice(bytes);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), E> {
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
    pub fn finish(mut self) -> Result<Package, E> {
        self.flush()?;
        if !self.manifest.tiles.iter().any(|t| t.imagery.is_some())
            || !self.manifest.tiles.iter().any(|t| t.elevation.is_some())
        {
            return Err(invalid_as(
                "a localization package requires imagery and terrain",
            ));
        }
        self.manifest.tiles.sort_by_key(|t| t.xyz);
        let used: std::collections::BTreeSet<&str> = self
            .manifest
            .tiles
            .iter()
            .flat_map(|t| [&t.imagery, &t.elevation])
            .flatten()
            .map(|a| a.chunk.as_str())
            .collect();
        let files = self
            .manifest
            .files
            .iter()
            .filter(|f| used.contains(f.sha256.as_str()))
            .cloned()
            .collect();
        self.manifest.files = files;
        self.manifest.pack_id = self.manifest.compute_pack_id()?;
        Ok(self.manifest)
    }
}

fn invalid_as<E: From<ImageryError>>(reason: impl Into<String>) -> E {
    E::from(invalid(reason))
}

#[cfg(test)]
mod tests;
