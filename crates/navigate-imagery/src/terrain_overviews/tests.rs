use super::*;
#[test]
fn incomplete_children_cannot_create_a_terrain_surface() {
    let image = RgbaImage::from_pixel(256, 256, Rgba([128, 17, 0, 255]));
    let incomplete = (0..3).map(|i| (Tile(2, i % 2, i / 2), &image));
    assert!(
        terrain_overviews(incomplete, 0)
            .expect("valid children")
            .is_empty()
    );
    let complete = (0..4).map(|i| (Tile(2, i % 2, i / 2), &image));
    let output = terrain_overviews(complete, 0).expect("complete children");
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].0, Tile(1, 0, 0));
    assert!(output[0].1.pixels().all(|p| *p == Rgba([128, 17, 0, 255])));
}
#[test]
fn height_averaging_handles_channel_carries_and_negative_elevation() {
    let image = RgbaImage::from_fn(256, 256, |x, _| {
        if x % 2 == 0 {
            Rgba([127, 255, 0, 255])
        } else {
            Rgba([128, 1, 0, 255])
        }
    });
    assert_eq!(mean_height(&image, 0, 0), Rgba([128, 0, 0, 255]));
    let mut invalid_image = image;
    invalid_image.put_pixel(0, 0, Rgba([0, 0, 0, 0]));
    assert!(terrain_overviews([(Tile(2, 0, 0), &invalid_image)], 0).is_err());
}
