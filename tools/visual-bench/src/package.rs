//! Bind one immutable map package to display and localization.
//!
//! The tool opens either a published package or a source folder:
//!
//! - A published package is a [`Package`] manifest file. Its chunks are in
//!   `chunks/<sha256>.bin` beside the manifest, or beside its `packs/` folder
//!   in a provider state folder.
//! - A source folder holds `map.json`, a [`navigate_imagery::SourceManifest`],
//!   and the encoded tile files it names. The tool builds its package in memory.
//!
//! Both report the package `pack_id` as the map identity, so the bench and the
//! browser preview name the same package the same way. Imagery uses 512 × 512
//! RGBA pixels, and alpha marks missing imagery. Elevation uses opaque
//! 256 × 256 Terrarium pixels. Terrarium height in metres is
//! `R*256 + G + B/256 - 32768`. The prior uses the declared vertical datum.
//!
//! Derived feature caches must bind the `pack_id`, camera calibration,
//! renderer revision, and matcher/model identity.

use crate::{BenchError, read_blocking};
use image::RgbaImage;
use navigate_imagery::{Asset, ImageryError, Package, SourceManifest, Tile};
use navigate_visual::{LocalFrame, MapRevision};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub(crate) use navigate_imagery::digest;

/// Region identity of a package that the tool builds from a local source folder.
const LOCAL_SOURCE_REGION: &str = "local-source";

pub(crate) struct DecodedTile {
    pub xyz: [u32; 3],
    pub imagery: Option<RgbaImage>,
    pub elevation: Option<RgbaImage>,
}

pub(crate) struct MapPackage {
    pub revision: MapRevision,
    pub frame: LocalFrame,
    pub manifest: Package,
    pub tiles: Vec<DecodedTile>,
}

type Chunks = BTreeMap<String, Vec<u8>>;

impl MapPackage {
    /// Open a published package manifest, or a source folder with `map.json`.
    pub fn open_blocking(path: &Path) -> Result<Self, BenchError> {
        let (manifest, chunks) = if path.join("map.json").is_file() {
            build_source_blocking(path)?
        } else {
            open_published_blocking(path)?
        };
        manifest.validate_for_reading().map_err(package_error)?;
        let tiles = manifest
            .tiles
            .iter()
            .map(|tile| decode_tile(tile.xyz, &tile.imagery, &tile.elevation, &chunks))
            .collect::<Result<Vec<_>, _>>()?;
        let [lat, lon] = manifest.anchor_lat_lon;
        Ok(Self {
            revision: MapRevision {
                release_id: manifest.release_id.clone(),
                manifest_sha256: manifest.pack_id.clone(),
            },
            frame: LocalFrame::anchor_mercator(lat, lon)?,
            manifest,
            tiles,
        })
    }
}

fn package_error(source: ImageryError) -> BenchError {
    BenchError::Package {
        reason: source.to_string(),
    }
}

fn build_source_blocking(root: &Path) -> Result<(Package, Chunks), BenchError> {
    let path = root.join("map.json");
    let source: SourceManifest =
        serde_json::from_slice(&read_blocking(&path)?).map_err(|source| BenchError::Json {
            path: path.clone(),
            source,
        })?;
    let canonical_root = root.canonicalize().map_err(|source| BenchError::Io {
        path: root.to_owned(),
        source,
    })?;
    let mut chunks = Chunks::new();
    let package = source.build(
        LOCAL_SOURCE_REGION,
        |_, _, asset| read_source_asset_blocking(&canonical_root, &asset.path),
        |sha, bytes| {
            chunks.insert(sha.to_owned(), bytes.to_vec());
            Ok::<(), BenchError>(())
        },
    )?;
    Ok((package, chunks))
}

fn read_source_asset_blocking(root: &Path, relative: &str) -> Result<Vec<u8>, BenchError> {
    let path = root.join(relative);
    let canonical = path.canonicalize().map_err(|source| BenchError::Io {
        path: path.clone(),
        source,
    })?;
    if !canonical.starts_with(root) {
        return Err(BenchError::Package {
            reason: format!("asset escapes package: {}", path.display()),
        });
    }
    read_blocking(&canonical)
}

fn open_published_blocking(path: &Path) -> Result<(Package, Chunks), BenchError> {
    let manifest: Package =
        serde_json::from_slice(&read_blocking(path)?).map_err(|source| BenchError::Json {
            path: path.to_owned(),
            source,
        })?;
    if manifest.compute_pack_id().map_err(package_error)? != manifest.pack_id {
        return Err(BenchError::Package {
            reason: "pack_id does not match the manifest".into(),
        });
    }
    let root = chunk_root(path);
    let mut chunks = Chunks::new();
    for chunk in &manifest.files {
        let file = root.join("chunks").join(format!("{}.bin", chunk.sha256));
        let bytes = read_blocking(&file)?;
        if bytes.len() != chunk.size || digest(&bytes) != chunk.sha256 {
            return Err(BenchError::Package {
                reason: format!("digest mismatch at {}", file.display()),
            });
        }
        chunks.insert(chunk.sha256.clone(), bytes);
    }
    Ok((manifest, chunks))
}

/// A provider state folder keeps manifests in `packs/` beside `chunks/`.
fn chunk_root(manifest: &Path) -> PathBuf {
    let parent = manifest.parent().unwrap_or(Path::new("."));
    match parent.file_name() {
        Some(name) if name == "packs" => parent.parent().unwrap_or(parent).to_owned(),
        _ => parent.to_owned(),
    }
}

fn decode_tile(
    xyz: Tile,
    imagery: &Option<Asset>,
    elevation: &Option<Asset>,
    chunks: &Chunks,
) -> Result<DecodedTile, BenchError> {
    let Tile(z, x, y) = xyz;
    let decode = |asset: &Asset| -> Result<RgbaImage, BenchError> {
        let chunk = chunks
            .get(&asset.chunk)
            .ok_or_else(|| BenchError::Package {
                reason: format!("tile {xyz:?} references an absent chunk"),
            })?;
        navigate_imagery::verify_asset(asset, chunk).map_err(package_error)?;
        let bytes = &chunk[asset.offset..asset.offset + asset.length];
        image::load_from_memory(bytes)
            .map(|image| image.to_rgba8())
            .map_err(|source| BenchError::Image {
                path: PathBuf::from(format!("{z}/{x}/{y}")),
                source,
            })
    };
    let imagery = imagery.as_ref().map(decode).transpose()?;
    let elevation = elevation.as_ref().map(decode).transpose()?;
    if imagery
        .as_ref()
        .is_some_and(|image| image.dimensions() != (512, 512))
        || elevation.as_ref().is_some_and(|image| {
            image.dimensions() != (256, 256) || image.pixels().any(|p| p[3] != 255)
        })
    {
        return Err(BenchError::Package {
            reason: format!("tile {xyz:?} needs 512px imagery or opaque 256px Terrarium DEM"),
        });
    }
    Ok(DecodedTile {
        xyz: [z, x, y],
        imagery,
        elevation,
    })
}

#[cfg(test)]
mod tests;
