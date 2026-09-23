//! Source folders that name imagery and elevation files by relative path and digest.
//!
//! A source folder holds `map.json` and encoded tile files. It is an import
//! format. [`SourceManifest::build`] turns it into a [`Package`], whose
//! `pack_id` is the one map identity that every reader reports.

use crate::{ImageryError, Package, PackageBuilder, Tile, error::invalid, is_digest};
use serde::{Deserialize, Serialize};

const MAX_TILES: usize = 1024;
const MAX_ZOOM: u32 = 22;

/// One encoded file in a source folder.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceAsset {
    /// Path below the folder, with `/` separators and no `.` or `..` parts.
    pub path: String,
    /// SHA-256 of the exact encoded file bytes.
    pub sha256: String,
}

/// Source files at one Web Mercator tile address.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceTile {
    /// Web Mercator address.
    pub xyz: Tile,
    /// Imagery file: 512 × 512 RGBA, alpha marks missing imagery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imagery: Option<SourceAsset>,
    /// Elevation file: 256 × 256 opaque Terrarium RGB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elevation: Option<SourceAsset>,
}

/// The `map.json` manifest of a source folder.
///
/// Version 1 pairs imagery and elevation at each address. Version 2 also
/// permits a tile with only one of them.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceManifest {
    /// Source schema version, 1 or 2.
    pub schema_version: u32,
    /// Data release identity.
    pub release_id: String,
    /// Local frame anchor in latitude, longitude order.
    pub anchor_lat_lon: [f64; 2],
    /// Vertical datum of the elevation, including an explicit unknown value.
    pub elevation_datum: String,
    /// Source attribution.
    pub attribution: String,
    /// Source files by tile address.
    pub tiles: Vec<SourceTile>,
    /// Source identities and processing information.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<serde_json::Value>,
}

impl SourceManifest {
    /// Check metadata, tile addresses, roles and relative paths.
    ///
    /// # Errors
    /// Rejects an unknown schema, empty metadata, an invalid anchor, invalid
    /// or repeated addresses, missing roles, and unsafe paths.
    pub fn validate(&self) -> Result<(), ImageryError> {
        let [lat, lon] = self.anchor_lat_lon;
        if ![1, 2].contains(&self.schema_version)
            || self.release_id.is_empty()
            || self.elevation_datum.is_empty()
            || self.attribution.is_empty()
            || !(lat.is_finite() && lon.is_finite() && lat.abs() <= 85.0 && lon.abs() <= 180.0)
            || self.tiles.is_empty()
            || self.tiles.len() > MAX_TILES
        {
            return Err(invalid("invalid source manifest metadata"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for tile in &self.tiles {
            let Tile(z, x, y) = tile.xyz;
            let paired = tile.imagery.is_some() && tile.elevation.is_some();
            let any = tile.imagery.is_some() || tile.elevation.is_some();
            if z > MAX_ZOOM || x >= 1 << z || y >= 1 << z || !seen.insert(tile.xyz) {
                return Err(invalid(format!("invalid or repeated tile {:?}", tile.xyz)));
            }
            if !any || (self.schema_version == 1 && !paired) {
                return Err(invalid(format!(
                    "tile {:?} lacks required sources for schema {}",
                    tile.xyz, self.schema_version
                )));
            }
            for asset in [&tile.imagery, &tile.elevation].into_iter().flatten() {
                if !safe_path(&asset.path) || !is_digest(&asset.sha256) {
                    return Err(invalid(format!("invalid source asset {:?}", asset.path)));
                }
            }
        }
        Ok(())
    }

    /// Build the package for `region_id`, reading each file through `read`.
    ///
    /// `read` receives the tile address, whether the file is elevation, and
    /// the asset. [`PackageBuilder::add`] verifies each digest.
    ///
    /// # Errors
    /// Returns validation failures, `read` and `write` failures, and package rule failures.
    pub fn build<R, W, E>(&self, region_id: &str, mut read: R, write: W) -> Result<Package, E>
    where
        R: FnMut(Tile, bool, &SourceAsset) -> Result<Vec<u8>, E>,
        W: FnMut(&str, &[u8]) -> Result<(), E>,
        E: From<ImageryError>,
    {
        self.validate()?;
        let manifest = Package {
            schema_version: 1,
            region_id: region_id.into(),
            release_id: self.release_id.clone(),
            anchor_lat_lon: self.anchor_lat_lon,
            elevation_datum: self.elevation_datum.clone(),
            attribution: self.attribution.clone(),
            files: vec![],
            tiles: vec![],
            provenance: self.provenance.clone(),
            supersedes: None,
            pack_id: String::new(),
        };
        let mut builder = PackageBuilder::new(manifest, write)?;
        let mut tiles: Vec<&SourceTile> = self.tiles.iter().collect();
        tiles.sort_by_key(|t| t.xyz);
        for tile in tiles {
            for (elevation, asset) in [(false, &tile.imagery), (true, &tile.elevation)] {
                if let Some(asset) = asset {
                    let bytes = read(tile.xyz, elevation, asset)?;
                    builder.add(tile.xyz, elevation, &bytes, &asset.sha256)?;
                }
            }
        }
        builder.finish()
    }
}

fn safe_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

#[cfg(test)]
mod tests;
