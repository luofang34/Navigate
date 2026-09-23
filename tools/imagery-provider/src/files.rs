use crate::{ProviderError, error::invalid};
use image::{DynamicImage, Rgba, RgbaImage, imageops};
use navigate_imagery::{Package, PackageBuilder, Tile, digest, tile_envelope};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};

/// A persisted catalogue entry and its verified download manifest.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Region {
    /// Catalogue identity.
    pub id: String,
    /// Display label.
    pub label: String,
    /// Immutable package identity.
    pub pack_id: String,
    /// Local plane origin in latitude, longitude order.
    pub anchor_lat_lon: [f64; 2],
    /// Imagery envelope in west, south, east, north order.
    pub bounds: [f64; 4],
    /// Total encoded chunk size.
    pub bytes: usize,
    /// Service-relative overview image URL.
    pub thumbnail: String,
    /// Verified package manifest.
    pub manifest: Package,
}

pub(super) fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> ProviderError + '_ {
    |source| ProviderError::Io {
        path: path.to_owned(),
        source,
    }
}

pub(super) fn read_blocking(path: &Path) -> Result<Vec<u8>, ProviderError> {
    std::fs::read(path).map_err(io_error(path))
}

pub(super) fn write_blocking(path: &Path, bytes: &[u8]) -> Result<(), ProviderError> {
    let parent = path
        .parent()
        .ok_or_else(|| invalid("output path has no parent"))?;
    std::fs::create_dir_all(parent).map_err(io_error(parent))?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(io_error(parent))?;
    file.write_all(bytes).map_err(io_error(path))?;
    file.as_file().sync_all().map_err(io_error(path))?;
    file.persist(path)
        .map_err(|error| io_error(path)(error.error))?;
    Ok(())
}

pub(super) fn chunk_blocking(state: &Path, sha: &str, bytes: &[u8]) -> Result<(), ProviderError> {
    let path = state.join("chunks").join(format!("{sha}.bin"));
    if path.exists() {
        if digest(&read_blocking(&path)?) != sha {
            return Err(invalid(format!("corrupt stored chunk {sha}")));
        }
        return Ok(());
    }
    write_blocking(&path, bytes)
}

pub(super) fn png(image: RgbaImage) -> Result<Vec<u8>, ProviderError> {
    let mut data = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut data, image::ImageFormat::Png)?;
    Ok(data.into_inner())
}

pub(super) fn publish_blocking(
    state: &Path,
    manifest: Package,
    label: String,
    overview: Overview,
) -> Result<Region, ProviderError> {
    let id = &manifest.region_id;
    if id.is_empty()
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(invalid("invalid region identity"));
    }
    let bounds = overview.bounds()?;
    let mut data = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(overview.image)
        .thumbnail(1000, 600)
        .to_rgb8()
        .write_to(&mut data, image::ImageFormat::Jpeg)?;
    write_blocking(
        &state.join("catalog").join(format!("{id}.jpg")),
        data.get_ref(),
    )?;
    write_blocking(
        &state
            .join("packs")
            .join(format!("{}.json", manifest.pack_id)),
        &serde_json::to_vec(&manifest)?,
    )?;
    let region = Region {
        id: id.clone(),
        label,
        pack_id: manifest.pack_id.clone(),
        anchor_lat_lon: manifest.anchor_lat_lon,
        bounds,
        bytes: manifest.files.iter().map(|f| f.size).sum(),
        thumbnail: format!("/catalog/{id}.jpg"),
        manifest,
    };
    write_blocking(
        &state.join("catalog").join(format!("{}.json", region.id)),
        &serde_json::to_vec(&region)?,
    )?;
    Ok(region)
}

pub(super) struct Overview {
    image: RgbaImage,
    tiles: Vec<Tile>,
    origin: [u32; 2],
}
impl Overview {
    pub(super) fn new(tiles: Vec<Tile>) -> Result<Self, ProviderError> {
        let z = tiles
            .iter()
            .map(|t| t.0)
            .max()
            .ok_or_else(|| invalid("no imagery for overview"))?;
        let tiles: Vec<_> = tiles.into_iter().filter(|t| t.0 == z).collect();
        let x0 = tiles
            .iter()
            .map(|t| t.1)
            .min()
            .ok_or_else(|| invalid("no imagery"))?;
        let y0 = tiles
            .iter()
            .map(|t| t.2)
            .min()
            .ok_or_else(|| invalid("no imagery"))?;
        let x1 = tiles
            .iter()
            .map(|t| t.1)
            .max()
            .ok_or_else(|| invalid("no imagery"))?
            + 1;
        let y1 = tiles
            .iter()
            .map(|t| t.2)
            .max()
            .ok_or_else(|| invalid("no imagery"))?
            + 1;
        if u64::from(x1 - x0) * u64::from(y1 - y0) > 4096 {
            return Err(invalid("overview envelope is too large"));
        }
        Ok(Self {
            image: RgbaImage::from_pixel((x1 - x0) * 32, (y1 - y0) * 32, Rgba([14, 24, 31, 255])),
            tiles,
            origin: [x0, y0],
        })
    }
    pub(super) fn add(&mut self, tile: Tile, image: &RgbaImage) {
        if !self.tiles.contains(&tile) {
            return;
        }
        let thumb = imageops::resize(image, 32, 32, imageops::FilterType::Triangle);
        imageops::overlay(
            &mut self.image,
            &thumb,
            i64::from(tile.1 - self.origin[0]) * 32,
            i64::from(tile.2 - self.origin[1]) * 32,
        );
    }
    fn bounds(&self) -> Result<[f64; 4], ProviderError> {
        Ok(tile_envelope(&self.tiles)?)
    }
}

#[derive(Deserialize)]
struct SourceAsset {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
struct SourceTile {
    xyz: Tile,
    imagery: Option<SourceAsset>,
    elevation: Option<SourceAsset>,
}

/// Convert an existing schema-2 reference folder to verified offline chunks.
///
/// # Errors
/// Rejects unsupported schemas, corrupt assets, path escapes, or file failures.
pub fn import_region_blocking(
    source: &Path,
    state: &Path,
    id: &str,
    label: &str,
) -> Result<Region, ProviderError> {
    let source = source.canonicalize().map_err(io_error(source))?;
    let mut value: serde_json::Value =
        serde_json::from_slice(&read_blocking(&source.join("map.json"))?)?;
    if value["schema_version"] != 2 {
        return Err(invalid("source map requires schema 2"));
    }
    let mut tiles: Vec<SourceTile> = serde_json::from_value(value["tiles"].take())?;
    tiles.sort_by_key(|t| t.xyz);
    value["schema_version"] = 1.into();
    value["region_id"] = id.into();
    value["tiles"] = serde_json::json!([]);
    value["files"] = serde_json::json!([]);
    let manifest: Package = serde_json::from_value(value)?;
    let mut overview = Overview::new(
        tiles
            .iter()
            .filter(|t| t.imagery.is_some())
            .map(|t| t.xyz)
            .collect(),
    )?;
    let mut builder =
        PackageBuilder::new(manifest, |sha, bytes| chunk_blocking(state, sha, bytes))?;
    for tile in tiles {
        for (elevation, asset) in [(false, tile.imagery), (true, tile.elevation)] {
            let Some(asset) = asset else {
                continue;
            };
            let path = source
                .join(asset.path)
                .canonicalize()
                .map_err(io_error(&source))?;
            if !path.starts_with(&source) {
                return Err(invalid("source asset escapes package directory"));
            }
            let bytes = read_blocking(&path)?;
            builder.add(tile.xyz, elevation, &bytes, &asset.sha256)?;
            if !elevation {
                overview.add(tile.xyz, &image::load_from_memory(&bytes)?.to_rgba8());
            }
        }
    }
    publish_blocking(state, builder.finish()?, label.into(), overview)
}

/// Load saved catalogues and optionally import local schema-2 packages.
///
/// # Errors
/// Returns invalid catalogue, data, or filesystem errors.
pub fn load_catalog_blocking(
    state: &Path,
    catalog: Option<&Path>,
) -> Result<BTreeMap<String, Region>, ProviderError> {
    let mut result = BTreeMap::new();
    for name in ["chunks", "packs", "catalog"] {
        let p = state.join(name);
        std::fs::create_dir_all(&p).map_err(io_error(&p))?;
    }
    if let Some(path) = catalog {
        #[derive(Deserialize)]
        struct Entry {
            id: String,
            label: String,
            path: PathBuf,
        }
        let entries: Vec<Entry> = serde_json::from_slice(&read_blocking(path)?)?;
        for e in entries {
            let r = import_region_blocking(&e.path, state, &e.id, &e.label)?;
            result.insert(r.id.clone(), r);
        }
    }
    let dir = state.join("catalog");
    for entry in std::fs::read_dir(&dir).map_err(io_error(&dir))? {
        let path = entry.map_err(io_error(&dir))?.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let r: Region = serde_json::from_slice(&read_blocking(&path)?)?;
            result.insert(r.id.clone(), r);
        }
    }
    Ok(result)
}
