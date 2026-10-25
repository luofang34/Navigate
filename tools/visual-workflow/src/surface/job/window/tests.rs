use super::*;
use navigate_visual::{MapRevision, RasterGeometry};

fn source() -> OrthographicReference {
    OrthographicReference {
        map: MapRevision {
            release_id: "window-test".into(),
            manifest_sha256: "a".repeat(64),
        },
        image: image::GrayImage::from_fn(80, 64, |x, y| image::Luma([(x + y) as u8])),
        heights_m: (0..80 * 64).map(|i| (i % 113) as f32).collect(),
        geometry: RasterGeometry {
            origin: [100.0, 200.0].into(),
            column: [0.2, 0.03].into(),
            row: [0.04, -0.2].into(),
        },
    }
}
#[test]
fn cropped_pixels_retain_metric_points_on_a_skewed_grid() {
    let original = source();
    let crop = crop(source(), [13, 17, 48, 32], 40_000).unwrap();
    assert_eq!(crop.map, original.map);
    assert_eq!(crop.image.dimensions(), (48, 32));
    for y in 0..31 {
        for x in 0..47 {
            assert_eq!(
                crop.image.get_pixel(x, y),
                original.image.get_pixel(x + 13, y + 17)
            );
            let p = nalgebra::Vector2::new(f64::from(x) + 0.2, f64::from(y) + 0.7);
            let expected = original
                .world_at(p + nalgebra::Vector2::new(13.0, 17.0))
                .unwrap();
            assert!((crop.world_at(p).unwrap() - expected).norm() < 1e-10);
        }
    }
}
#[test]
fn invalid_windows_and_overlapping_allocations_are_rejected() {
    for window in [
        [0, 0, 31, 32],
        [49, 0, 32, 32],
        [0, 33, 32, 32],
        [u32::MAX, 0, 32, 32],
    ] {
        assert!(crop(source(), window, usize::MAX).is_err());
    }
    let bound = (80 * 64 + 32 * 32) * 5;
    assert!(crop(source(), [0, 0, 32, 32], bound - 1).is_err());
    assert!(crop(source(), [0, 0, 32, 32], bound).is_ok());
    assert!(crop(source(), [0, 0, 80, 64], 80 * 64 * 5).is_ok());
}

#[test]
fn borrowed_crop_has_identical_pixels_heights_and_geometry() {
    let source = source();
    let borrowed = crop_borrowed(&source, [13, 17, 48, 32], 40_000).unwrap();
    let owned = crop(source, [13, 17, 48, 32], 40_000).unwrap();
    assert_eq!(borrowed.image, owned.image);
    assert_eq!(borrowed.heights_m, owned.heights_m);
    assert_eq!(borrowed.geometry.origin, owned.geometry.origin);
    assert_eq!(borrowed.map, owned.map);
}
