//! Bind one immutable set of files to display and localization.
//!
//! `map.json` is a [`Manifest`]. Asset paths are relative to its directory.
//! Every asset must match its SHA-256. Each tile contains opaque 512 × 512
//! imagery and opaque 256 × 256 Terrarium elevation. Terrarium height in meters
//! is `R*256 + G + B/256 - 32768`. The prior uses the declared vertical datum.
//!
//! The tool loads one selected package into memory. Both display rendering and
//! localization references use those same source files. There is no downloader
//! or second localization database. A host can replace this development reader
//! with decoded tiles from its pinned archive selection. Keep source selection
//! and derived feature caches separate; derived caches must bind the manifest
//! hash, camera calibration, renderer settings, and matcher/model identity.

use crate::{BenchError, read_blocking};
use image::RgbaImage;
use navigate_visual::MapRevision;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Artifact {
    /// Path inside the package root. Parent paths and escaping symlinks reject.
    pub path: PathBuf,
    /// SHA-256 of the exact encoded file bytes.
    pub sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Tile {
    /// XYZ tile coordinate in `[zoom, x, y]` order.
    pub xyz: [u32; 3],
    pub imagery: Artifact,
    pub elevation: Artifact,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    /// Manifest contract version. Only version 1 is supported.
    pub schema_version: u32,
    /// Immutable selection label shared by display and localization.
    pub release_id: String,
    /// Local Mercator frame anchor as latitude and longitude in degrees.
    pub anchor_lat_lon: [f64; 2],
    /// Vertical reference for terrain and camera altitude. No conversion is inferred.
    pub elevation_datum: String,
    pub attribution: String,
    pub tiles: Vec<Tile>,
}

pub(crate) struct DecodedTile {
    pub xyz: [u32; 3],
    pub imagery: RgbaImage,
    pub elevation: RgbaImage,
}

pub(crate) struct MapPackage {
    pub revision: MapRevision,
    pub manifest: Manifest,
    pub tiles: Vec<DecodedTile>,
}

impl MapPackage {
    pub fn open_blocking(root: &Path) -> Result<Self, BenchError> {
        let path = root.join("map.json");
        let bytes = read_blocking(&path)?;
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|source| BenchError::Json {
                path: path.clone(),
                source,
            })?;
        if manifest.schema_version != 1
            || manifest.tiles.is_empty()
            || manifest.release_id.is_empty()
            || manifest.elevation_datum.is_empty()
            || manifest.attribution.is_empty()
            || !manifest.anchor_lat_lon.iter().all(|v| v.is_finite())
            || manifest.anchor_lat_lon[0].abs() > 85.0
            || manifest.anchor_lat_lon[1].abs() > 180.0
        {
            return Err(BenchError::Package {
                reason: "invalid manifest metadata".into(),
            });
        }
        let mut coordinates = std::collections::BTreeSet::new();
        let mut tiles = Vec::new();
        for tile in &manifest.tiles {
            let [z, x, y] = tile.xyz;
            if z > 22 || x >= (1 << z) || y >= (1 << z) || !coordinates.insert(tile.xyz) {
                return Err(BenchError::Package {
                    reason: format!("invalid or repeated tile {:?}", tile.xyz),
                });
            }
            let decoded = DecodedTile {
                xyz: tile.xyz,
                imagery: load_image_blocking(root, &tile.imagery)?,
                elevation: load_image_blocking(root, &tile.elevation)?,
            };
            if decoded.imagery.dimensions() != (512, 512)
                || decoded.elevation.dimensions() != (256, 256)
                || decoded
                    .imagery
                    .pixels()
                    .chain(decoded.elevation.pixels())
                    .any(|p| p[3] != 255)
            {
                return Err(BenchError::Package {
                    reason: format!(
                        "tile {:?} needs opaque 512px imagery and 256px Terrarium DEM",
                        tile.xyz
                    ),
                });
            }
            tiles.push(decoded);
        }
        Ok(Self {
            revision: MapRevision {
                release_id: manifest.release_id.clone(),
                manifest_sha256: digest(&bytes),
            },
            manifest,
            tiles,
        })
    }
}

pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn load_image_blocking(root: &Path, artifact: &Artifact) -> Result<RgbaImage, BenchError> {
    if artifact.path.is_absolute()
        || artifact
            .path
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err(BenchError::Package {
            reason: format!("invalid asset path {:?}", artifact.path),
        });
    }
    let path = root.join(&artifact.path);
    let canonical_root = root.canonicalize().map_err(|source| BenchError::Io {
        path: root.to_owned(),
        source,
    })?;
    let canonical_path = path.canonicalize().map_err(|source| BenchError::Io {
        path: path.clone(),
        source,
    })?;
    if !canonical_path.starts_with(canonical_root) {
        return Err(BenchError::Package {
            reason: format!("asset escapes package: {}", path.display()),
        });
    }
    let bytes = read_blocking(&path)?;
    if digest(&bytes) != artifact.sha256 {
        return Err(BenchError::Package {
            reason: format!("digest mismatch at {}", path.display()),
        });
    }
    image::load_from_memory(&bytes)
        .map(|image| image.to_rgba8())
        .map_err(|source| BenchError::Image { path, source })
}

#[cfg(test)]
mod tests;
