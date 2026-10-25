use super::{Error, assets, catalog::Tile};
use image::{ImageReader, RgbImage};
use navigate_visual_render::{ImageAtlas, TerrainAtlas};
use std::{
    io::Cursor,
    path::{Component, Path},
};

pub(super) fn decode_blocking(root: &Path, tile: &Tile) -> Result<RgbImage, Error> {
    if tile
        .path
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
        || tile.bytes > 4 * 1024 * 1024
    {
        return Err(Error::Input(format!(
            "invalid tile path or size: {}",
            tile.path.display()
        )));
    }
    let path = root.join(&tile.path);
    let bytes = assets::read_blocking(&path, 4 * 1024 * 1024)?;
    if bytes.len() != tile.bytes {
        return Err(Error::Input(format!(
            "tile size mismatch: {}",
            path.display()
        )));
    }
    assets::verify(&bytes, &tile.sha256)?;
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|source| Error::Io {
            operation: "detect tile format",
            path: path.clone(),
            source,
        })?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(256);
    limits.max_image_height = Some(256);
    limits.max_alloc = Some(8 * 1024 * 1024);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|source| Error::Image {
            path: path.clone(),
            source,
        })?
        .to_rgb8();
    if image.dimensions() != (256, 256) {
        return Err(Error::Input(format!("tile dimensions: {}", path.display())));
    }
    Ok(image)
}

pub(super) fn imagery(atlas: &mut ImageAtlas, tile: &Tile, image: RgbImage) {
    let pixels = atlas.layout.tile_pixels;
    let gray = image::DynamicImage::ImageRgb8(image).to_luma8();
    let scaled =
        image::imageops::resize(&gray, pixels, pixels, image::imageops::FilterType::Triangle);
    let x = (tile.xyz[1] - atlas.layout.origin_tile[0]) * pixels;
    let y = (tile.xyz[2] - atlas.layout.origin_tile[1]) * pixels;
    for row in 0..pixels {
        let dest = ((y + row) * atlas.layout.size[0] + x) as usize;
        let source = (row * pixels) as usize;
        atlas.pixels[dest..dest + pixels as usize]
            .copy_from_slice(&scaled.as_raw()[source..source + pixels as usize]);
        atlas.coverage[dest..dest + pixels as usize].fill(255);
    }
}

pub(super) fn terrain(atlas: &mut TerrainAtlas, tile: &Tile, image: RgbImage) {
    let x = (tile.xyz[1] - atlas.layout.origin_tile[0]) * 256;
    let y = (tile.xyz[2] - atlas.layout.origin_tile[1]) * 256;
    for (column, row, rgb) in image.enumerate_pixels() {
        let dest = ((y + row) * atlas.layout.size[0] + x + column) as usize;
        atlas.heights_m[dest] =
            f32::from(rgb[0]) * 256.0 + f32::from(rgb[1]) + f32::from(rgb[2]) / 256.0 - 32768.0;
        atlas.coverage[dest] = 255;
    }
}
