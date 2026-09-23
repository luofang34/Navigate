#![allow(clippy::expect_used)]
use super::*;
use image::Rgba;
fn tile(z: u8, x: i32, y: i32, color: [u8; 4]) -> AvailableRasterLayerData {
    AvailableRasterLayerData {
        coords: WorldTileCoords::from((x, y, z.into())),
        source_layer: "imagery".into(),
        image: RgbaImage::from_pixel(512, 512, Rgba(color)),
    }
}
#[test]
fn incomplete_overviews_leave_global_context_visible() {
    let tiles = vec![
        tile(2, 0, 0, [200, 0, 0, 255]),
        tile(2, 1, 0, [0, 200, 0, 255]),
        tile(1, 1, 0, [0, 0, 200, 255]),
    ];
    assert!(parents(&tiles).is_empty());
}
#[test]
fn complete_overviews_preserve_pixels_without_replacing_supplied_tiles_or_root() {
    let mut tiles = vec![
        tile(2, 0, 0, [200, 0, 0, 255]),
        tile(2, 1, 0, [0, 200, 0, 255]),
        tile(2, 0, 1, [0, 0, 200, 255]),
        tile(2, 1, 1, [200, 200, 0, 255]),
    ];
    let derived = parents(&tiles);
    assert_eq!(derived.len(), 1);
    let parent = &derived[0];
    assert_eq!(parent.coords, WorldTileCoords::from((0, 0, 1_u8.into())));
    assert_eq!(parent.image.get_pixel(30, 30), &Rgba([200, 0, 0, 255]));
    assert_eq!(parent.image.get_pixel(300, 30), &Rgba([0, 200, 0, 255]));
    assert_eq!(parent.image.get_pixel(30, 300), &Rgba([0, 0, 200, 255]));
    assert_eq!(parent.image.get_pixel(300, 300), &Rgba([200, 200, 0, 255]));
    tiles.push(tile(1, 0, 0, [50, 50, 50, 255]));
    assert!(parents(&tiles).is_empty());
}
