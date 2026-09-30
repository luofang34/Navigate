//! Reference raster overviews with conservative coverage.
use crate::{ImageryError, Tile, error::invalid};
use image::{Rgba, RgbaImage};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

/// Derive missing coarser raster tiles from verified 512 by 512 imagery.
///
/// Supplied tiles take precedence. Missing quadrants remain transparent. A
/// parent pixel is opaque only if every contributing child pixel is opaque.
/// These images are derived cache data, not an independent map release.
///
/// # Errors
/// Rejects invalid XYZ coordinates, zoom limits, duplicate tiles, and sizes.
pub fn raster_overviews<'a>(
    tiles: impl IntoIterator<Item = (Tile, &'a RgbaImage)>,
    minimum_zoom: u32,
) -> Result<Vec<(Tile, RgbaImage)>, ImageryError> {
    if minimum_zoom > 30 {
        return Err(invalid("overview minimum zoom exceeds 30"));
    }
    let mut levels = BTreeMap::new();
    for (tile, image) in tiles {
        if tile.0 > 30
            || tile.1 >= (1_u32 << tile.0)
            || tile.2 >= (1_u32 << tile.0)
            || image.dimensions() != (512, 512)
        {
            return Err(invalid(format!(
                "unsupported overview tile {tile:?}: {:?}",
                image.dimensions()
            )));
        }
        if levels.insert(tile, Cow::Borrowed(image)).is_some() {
            return Err(invalid(format!("duplicate overview tile {tile:?}")));
        }
    }
    let supplied: BTreeSet<_> = levels.keys().copied().collect();
    let maximum = levels.keys().map(|t| t.0).max().unwrap_or(0);
    for zoom in (minimum_zoom.saturating_add(1)..=maximum).rev() {
        let keys: Vec<_> = levels.keys().filter(|t| t.0 == zoom).copied().collect();
        for child in keys {
            let parent = Tile(child.0 - 1, child.1 / 2, child.2 / 2);
            if supplied.contains(&parent) {
                continue;
            }
            let Some(image) = levels.get(&child) else {
                continue;
            };
            let reduced = reduce(image);
            let parent_image = levels
                .entry(parent)
                .or_insert_with(|| Cow::Owned(RgbaImage::new(512, 512)))
                .to_mut();
            for (x, y, pixel) in reduced.enumerate_pixels() {
                parent_image.put_pixel(x + (child.1 % 2) * 256, y + (child.2 % 2) * 256, *pixel);
            }
        }
    }
    Ok(levels
        .into_iter()
        .filter(|(tile, _)| !supplied.contains(tile))
        .map(|(tile, image)| (tile, image.into_owned()))
        .collect())
}
fn reduce(image: &RgbaImage) -> RgbaImage {
    RgbaImage::from_fn(256, 256, |x, y| {
        let mut sum = [0_u16; 3];
        let mut alpha = 255;
        for dy in 0..2 {
            for dx in 0..2 {
                let p = image.get_pixel(x * 2 + dx, y * 2 + dy);
                for c in 0..3 {
                    sum[c] += u16::from(p[c]);
                }
                alpha = alpha.min(p[3]);
            }
        }
        Rgba([
            (sum[0] / 4) as u8,
            (sum[1] / 4) as u8,
            (sum[2] / 4) as u8,
            alpha,
        ])
    })
}
#[cfg(test)]
mod tests;
