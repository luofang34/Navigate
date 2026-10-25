//! Render scenes from imagery and Terrarium elevation tiles of a [`TileSource`].
use crate::{LocateError, area::TileGrid, area::tile_xy};
use futures_util::future::join_all;
use image::{DynamicImage, GrayImage, ImageReader, RgbImage, imageops::FilterType};
use navigate_visual::{LocalFrame, MapRevision};
use navigate_visual_render::{AtlasLayout, ImageAtlas, RenderScene, TerrainAtlas};
use sha2::{Digest, Sha256};
use std::{future::Future, io::Cursor};

/// The two tile layers of a scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileKind {
    /// Map imagery, any 256 × 256 image format that the `image` crate reads.
    Imagery,
    /// Terrarium elevation PNG: `height = R·256 + G + B/256 − 32768` metres.
    Terrain,
}

impl TileKind {
    fn name(self) -> &'static str {
        match self {
            Self::Imagery => "imagery",
            Self::Terrain => "terrain",
        }
    }
}

/// Port to tile bytes. The browser host fetches over HTTP; native hosts can read a
/// catalog on disk. `Ok(None)` marks a tile that the source does not have.
pub trait TileSource {
    /// Get the encoded tile `z/x/y` of `kind`.
    fn fetch(
        &self,
        kind: TileKind,
        z: u32,
        x: u32,
        y: u32,
    ) -> impl Future<Output = Result<Option<Vec<u8>>, String>>;
    /// Identity of the source for result records, for example the URL templates.
    fn identity(&self) -> String;
}

/// The map area of a localization.
#[derive(Clone, Copy, Debug)]
pub struct SceneRequest {
    /// Latitude and longitude of the local anchor.
    pub anchor: [f64; 2],
    /// Half side of the square area, in metres.
    pub radius_m: f64,
    /// Imagery tile zoom.
    pub imagery_zoom: u32,
    /// Terrain tile zoom.
    pub terrain_zoom: u32,
    /// Samples per imagery tile side, a power of two up to 256.
    pub imagery_tile_pixels: u32,
    /// Most bytes of both atlases.
    pub max_bytes: usize,
    /// Tiles fetched at the same time.
    pub concurrency: usize,
}

/// Fetch, decode and assemble the scene. `progress` receives tiles done and total.
///
/// # Errors
///
/// Returns an error for a failed fetch, an invalid tile, or an area over the limit.
pub async fn build<S: TileSource>(
    source: &S,
    request: &SceneRequest,
    mut progress: impl FnMut(usize, usize),
) -> Result<RenderScene, LocateError> {
    if !(1..=256).contains(&request.imagery_tile_pixels)
        || !request.imagery_tile_pixels.is_power_of_two()
    {
        return Err(LocateError::Request("imagery tile pixels"));
    }
    let imagery = TileGrid::around(request.anchor, request.radius_m, request.imagery_zoom);
    let terrain = TileGrid::around(request.anchor, request.radius_m, request.terrain_zoom);
    let mut image_atlas = ImageAtlas {
        layout: layout(&imagery, request.imagery_tile_pixels),
        pixels: Vec::new(),
        coverage: Vec::new(),
    };
    let mut terrain_atlas = TerrainAtlas {
        layout: layout(&terrain, 256),
        heights_m: Vec::new(),
        coverage: Vec::new(),
    };
    let image_count = pixels(&image_atlas.layout);
    let terrain_count = pixels(&terrain_atlas.layout);
    let needed = image_count * 2 + terrain_count * 5;
    if needed > request.max_bytes {
        return Err(LocateError::Memory {
            needed,
            limit: request.max_bytes,
        });
    }
    image_atlas.pixels = vec![0; image_count];
    image_atlas.coverage = vec![0; image_count];
    terrain_atlas.heights_m = vec![0.0; terrain_count];
    terrain_atlas.coverage = vec![0; terrain_count];
    let mut jobs: Vec<(TileKind, u32, u32, u32)> = terrain
        .tiles()
        .map(|(x, y)| (TileKind::Terrain, terrain.zoom, x, y))
        .collect();
    jobs.extend(
        imagery
            .tiles()
            .map(|(x, y)| (TileKind::Imagery, imagery.zoom, x, y)),
    );
    let total = jobs.len();
    let mut done = 0;
    // The map identity: every tile address and its bytes, in fetch order.
    let mut digest = Sha256::new();
    for chunk in jobs.chunks(request.concurrency.max(1)) {
        let fetched = join_all(chunk.iter().map(|&(kind, z, x, y)| async move {
            (kind, z, x, y, source.fetch(kind, z, x, y).await)
        }))
        .await;
        for (kind, z, x, y, bytes) in fetched {
            let bytes = bytes.map_err(|message| LocateError::Fetch {
                kind: kind.name(),
                z,
                x,
                y,
                message,
            })?;
            digest.update(format!("{}/{z}/{x}/{y}:", kind.name()).as_bytes());
            if let Some(bytes) = bytes {
                digest.update(&bytes);
                let image = decode(&bytes).ok_or(LocateError::Tile {
                    kind: kind.name(),
                    z,
                    x,
                    y,
                })?;
                match kind {
                    TileKind::Imagery => put_imagery(&mut image_atlas, [x, y], image),
                    TileKind::Terrain => put_terrain(&mut terrain_atlas, [x, y], &image),
                }
            }
            done += 1;
        }
        progress(done, total);
    }
    let [lat, lon] = request.anchor;
    Ok(RenderScene {
        map: MapRevision {
            release_id: source.identity(),
            manifest_sha256: hex::encode(digest.finalize()),
        },
        frame: LocalFrame::anchor_mercator(lat, lon)?,
        imagery: image_atlas,
        terrain: terrain_atlas,
    })
}

fn layout(grid: &TileGrid, tile_pixels: u32) -> AtlasLayout {
    let [w, h] = grid.dimensions();
    AtlasLayout {
        zoom: grid.zoom,
        origin_tile: [grid.x[0], grid.y[0]],
        tile_pixels,
        size: [w * tile_pixels, h * tile_pixels],
    }
}

fn pixels(layout: &AtlasLayout) -> usize {
    layout.size[0] as usize * layout.size[1] as usize
}

/// Decode one 256 × 256 tile with bounded memory.
fn decode(bytes: &[u8]) -> Option<RgbImage> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(256);
    limits.max_image_height = Some(256);
    limits.max_alloc = Some(8 * 1024 * 1024);
    reader.limits(limits);
    let image = reader.decode().ok()?.to_rgb8();
    (image.dimensions() == (256, 256)).then_some(image)
}

fn put_imagery(atlas: &mut ImageAtlas, [x, y]: [u32; 2], image: RgbImage) {
    let pixels = atlas.layout.tile_pixels;
    let gray: GrayImage = DynamicImage::ImageRgb8(image).to_luma8();
    let scaled = image::imageops::resize(&gray, pixels, pixels, FilterType::Triangle);
    let left = (x - atlas.layout.origin_tile[0]) * pixels;
    let top = (y - atlas.layout.origin_tile[1]) * pixels;
    for row in 0..pixels {
        let dest = ((top + row) * atlas.layout.size[0] + left) as usize;
        let source = (row * pixels) as usize;
        atlas.pixels[dest..dest + pixels as usize]
            .copy_from_slice(&scaled.as_raw()[source..source + pixels as usize]);
        atlas.coverage[dest..dest + pixels as usize].fill(255);
    }
}

fn put_terrain(atlas: &mut TerrainAtlas, [x, y]: [u32; 2], image: &RgbImage) {
    let left = (x - atlas.layout.origin_tile[0]) * 256;
    let top = (y - atlas.layout.origin_tile[1]) * 256;
    for (column, row, rgb) in image.enumerate_pixels() {
        let dest = ((top + row) * atlas.layout.size[0] + left + column) as usize;
        atlas.heights_m[dest] =
            f32::from(rgb[0]) * 256.0 + f32::from(rgb[1]) + f32::from(rgb[2]) / 256.0 - 32768.0;
        atlas.coverage[dest] = 255;
    }
}

/// Terrain height at `[lat, lon]`, from the nearest covered sample.
pub fn height_at(atlas: &TerrainAtlas, point: [f64; 2]) -> Option<f64> {
    let [tx, ty] = tile_xy(point, atlas.layout.zoom);
    let scale = f64::from(atlas.layout.tile_pixels);
    let column = ((tx - f64::from(atlas.layout.origin_tile[0])) * scale).floor();
    let row = ((ty - f64::from(atlas.layout.origin_tile[1])) * scale).floor();
    let [w, h] = atlas.layout.size;
    if column < 0.0 || row < 0.0 || column >= f64::from(w) || row >= f64::from(h) {
        return None;
    }
    let index = row as usize * w as usize + column as usize;
    (atlas.coverage.get(index) == Some(&255)).then(|| f64::from(atlas.heights_m[index]))
}

#[cfg(test)]
mod tests;
