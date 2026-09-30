use super::*;
#[test]
fn missing_quadrants_and_transparent_pixels_never_become_valid() {
    let mut child = RgbaImage::from_pixel(512, 512, Rgba([80, 120, 160, 255]));
    child.put_pixel(2, 2, Rgba([0, 0, 0, 0]));
    let parents = raster_overviews([(Tile(3, 3, 2), &child)], 1).expect("valid child");
    let p = &parents
        .iter()
        .find(|(t, _)| *t == Tile(2, 1, 1))
        .expect("parent")
        .1;
    assert_eq!(*p.get_pixel(270, 12), Rgba([80, 120, 160, 255]));
    assert_eq!(p.get_pixel(257, 1)[3], 0);
    assert_eq!(p.get_pixel(10, 10)[3], 0);
    assert_eq!(p.get_pixel(400, 300)[3], 0);
    let grandparent = &parents
        .iter()
        .find(|(t, _)| t.0 == 1)
        .expect("grandparent")
        .1;
    assert!(grandparent.pixels().any(|p| p[3] == 255));
    assert!(grandparent.pixels().any(|p| p[3] == 0));
}
#[test]
fn supplied_parents_win_and_bound_the_generated_levels() {
    let child = RgbaImage::from_pixel(512, 512, Rgba([200, 0, 0, 255]));
    let parent = RgbaImage::from_pixel(512, 512, Rgba([0, 90, 0, 255]));
    let result = raster_overviews([(Tile(3, 0, 0), &child), (Tile(2, 0, 0), &parent)], 1)
        .expect("valid tiles");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, Tile(1, 0, 0));
    assert_eq!(*result[0].1.get_pixel(20, 20), Rgba([0, 90, 0, 255]));
}
#[test]
fn invalid_coordinates_sizes_and_duplicate_inputs_are_rejected() {
    let image = RgbaImage::new(512, 512);
    assert!(raster_overviews([(Tile(31, 0, 0), &image)], 0).is_err());
    assert!(raster_overviews([(Tile(3, 8, 0), &image)], 0).is_err());
    assert!(raster_overviews([(Tile(3, 1, 0), &image); 2], 0).is_err());
    assert!(raster_overviews([(Tile(0, 0, 0), &RgbaImage::new(256, 256))], 0).is_err());
}
