use super::*;
#[test]
fn absent_and_transparent_terrain_remain_unsupported() {
    let frame = navigate_visual::LocalFrame::anchor_mercator(0.0, 0.0).expect("anchor");
    let mut terrain = Terrain {
        frame,
        tiles: vec![],
    };
    assert_eq!(terrain.elevation(0.0, 0.0), None);
    let tile = image::RgbaImage::from_pixel(256, 256, image::Rgba([128, 20, 128, 255]));
    terrain.tiles.push(([1, 1, 1], tile));
    assert_eq!(terrain.elevation(0.0, 0.0), Some(20.5));
    terrain.tiles[0].1.get_pixel_mut(0, 0).0[3] = 0;
    assert_eq!(terrain.elevation(0.0, 0.0), None);
}
