use image::{
    RgbaImage,
    imageops::{FilterType, overlay, resize},
};
use maplibre::{coords::WorldTileCoords, raster::AvailableRasterLayerData};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn parents(tiles: &[AvailableRasterLayerData]) -> Vec<AvailableRasterLayerData> {
    let mut levels: BTreeMap<(u8, i32, i32), RgbaImage> = tiles
        .iter()
        .map(|t| ((t.coords.z.into(), t.coords.x, t.coords.y), t.image.clone()))
        .collect();
    let supplied: BTreeSet<_> = levels.keys().copied().collect();
    let Some(maximum) = levels.keys().map(|t| t.0).max() else {
        return Vec::new();
    };
    for zoom in (1..=maximum).rev() {
        let children: Vec<_> = levels
            .iter()
            .filter(|(key, _)| key.0 == zoom)
            .map(|(&(z, x, y), image)| ((z, x, y), resize(image, 256, 256, FilterType::Triangle)))
            .collect();
        for ((z, x, y), image) in children {
            let key = (z - 1, x / 2, y / 2);
            if supplied.contains(&key) {
                continue;
            }
            let parent = levels
                .entry(key)
                .or_insert_with(|| RgbaImage::new(512, 512));
            overlay(
                parent,
                &image,
                i64::from(x % 2) * 256,
                i64::from(y % 2) * 256,
            );
        }
    }
    levels
        .into_iter()
        .filter(|(key, _)| !supplied.contains(key))
        .map(|((z, x, y), image)| AvailableRasterLayerData {
            coords: WorldTileCoords::from((x, y, z.into())),
            source_layer: "imagery".into(),
            image,
        })
        .collect()
}

#[cfg(test)]
#[path = "display_tiles/tests.rs"]
mod tests;
