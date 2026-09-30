//! Complete terrain overviews from Terrarium elevation tiles.
use crate::{ImageryError, Tile, error::invalid};
use image::{Rgba, RgbaImage};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

/// Derive complete coarser Terrarium tiles from verified 256 by 256 tiles.
///
/// A parent needs all four children. Missing terrain is never filled with zero
/// elevation. Heights are averaged in physical units, including channel carries.
/// These tiles describe a coarser rendered surface, not new measured geometry.
///
/// # Errors
/// Rejects invalid coordinates, sizes, transparency, duplicates, and zoom limits.
pub fn terrain_overviews<'a>(
    tiles: impl IntoIterator<Item = (Tile, &'a RgbaImage)>,
    minimum_zoom: u32,
) -> Result<Vec<(Tile, RgbaImage)>, ImageryError> {
    if minimum_zoom > 30 {
        return Err(invalid("terrain overview minimum zoom exceeds 30"));
    }
    let mut levels = BTreeMap::new();
    for (tile, image) in tiles {
        if tile.0 > 30
            || tile.1 >= (1_u32 << tile.0)
            || tile.2 >= (1_u32 << tile.0)
            || image.dimensions() != (256, 256)
            || image.pixels().any(|p| p[3] != 255)
        {
            return Err(invalid(format!("invalid terrain overview tile {tile:?}")));
        }
        if levels.insert(tile, Cow::Borrowed(image)).is_some() {
            return Err(invalid(format!("duplicate terrain overview tile {tile:?}")));
        }
    }
    let supplied: BTreeSet<_> = levels.keys().copied().collect();
    let maximum = levels.keys().map(|t| t.0).max().unwrap_or(0);
    for zoom in (minimum_zoom.saturating_add(1)..=maximum).rev() {
        let parents: BTreeSet<_> = levels
            .keys()
            .filter(|t| t.0 == zoom)
            .map(|t| Tile(t.0 - 1, t.1 / 2, t.2 / 2))
            .collect();
        for parent in parents {
            if supplied.contains(&parent) {
                continue;
            }
            let children: Option<Vec<_>> = (0..4)
                .map(|i| levels.get(&Tile(zoom, parent.1 * 2 + i % 2, parent.2 * 2 + i / 2)))
                .collect();
            let Some(children) = children else {
                continue;
            };
            let image = RgbaImage::from_fn(256, 256, |x, y| {
                let child = &children[(y / 128 * 2 + x / 128) as usize];
                mean_height(child, x % 128 * 2, y % 128 * 2)
            });
            levels.insert(parent, Cow::Owned(image));
        }
    }
    Ok(levels
        .into_iter()
        .filter(|(tile, _)| !supplied.contains(tile))
        .map(|(tile, image)| (tile, image.into_owned()))
        .collect())
}
fn mean_height(image: &RgbaImage, x: u32, y: u32) -> Rgba<u8> {
    let mut sum = 0_u32;
    for dy in 0..2 {
        for dx in 0..2 {
            let p = image.get_pixel(x + dx, y + dy);
            sum += u32::from(p[0]) * 65536 + u32::from(p[1]) * 256 + u32::from(p[2]);
        }
    }
    let mean = sum / 4;
    Rgba([(mean >> 16) as u8, (mean >> 8) as u8, mean as u8, 255])
}
#[cfg(test)]
mod tests;
