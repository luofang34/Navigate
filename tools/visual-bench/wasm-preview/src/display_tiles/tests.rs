#![allow(clippy::expect_used)]
use super::*;
use image::Rgba;
fn tile(z: u8, x: i32, color: [u8; 4]) -> AvailableRasterLayerData {
    AvailableRasterLayerData {
        coords: WorldTileCoords::from((x, 0, z.into())),
        source_layer: "imagery".into(),
        image: RgbaImage::from_pixel(512, 512, Rgba(color)),
    }
}
#[test]
fn display_overviews_preserve_supplied_levels_and_missing_coverage() {
    let tiles = vec![
        tile(2, 0, [200, 0, 0, 255]),
        tile(2, 1, [0, 200, 0, 255]),
        tile(1, 1, [0, 0, 200, 255]),
    ];
    let derived = parents(&tiles);
    assert!(!derived.iter().any(|t| t.coords == tiles[2].coords));
    let parent = derived
        .iter()
        .find(|t| t.coords == WorldTileCoords::from((0, 0, 1_u8.into())))
        .expect("parent");
    assert_eq!(parent.image.get_pixel(30, 30), &Rgba([200, 0, 0, 255]));
    assert_eq!(parent.image.get_pixel(300, 30), &Rgba([0, 200, 0, 255]));
    assert_eq!(parent.image.get_pixel(30, 300)[3], 0);
    let root = derived
        .iter()
        .find(|t| t.coords.z == 0_u8.into())
        .expect("root");
    assert_eq!(root.image.get_pixel(400, 50), &Rgba([0, 0, 200, 255]));
}
