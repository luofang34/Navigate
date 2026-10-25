//! Height sources must be sampled by metric location, not array offset.
use super::*;
#[test]
fn remap_respects_grid_origin_spacing_and_missing_coverage() {
    let source = OrthographicReference {
        map: MapRevision {
            release_id: "source".into(),
            manifest_sha256: "a".repeat(64),
        },
        image: image::GrayImage::new(64, 64),
        heights_m: (0..64)
            .flat_map(|y| (0..64).map(move |x| (x + 2 * y) as f32))
            .collect(),
        geometry: navigate_visual::RasterGeometry {
            origin: [10.0, 20.0].into(),
            column: [0.5, 0.0].into(),
            row: [0.0, 0.5].into(),
        },
    };
    let target = OrthographicReference {
        map: MapRevision {
            release_id: "target".into(),
            manifest_sha256: "b".repeat(64),
        },
        image: image::GrayImage::new(32, 32),
        heights_m: vec![0.0; 1024],
        geometry: navigate_visual::RasterGeometry {
            origin: [12.0, 23.0].into(),
            column: [1.0, 0.0].into(),
            row: [0.0, 1.0].into(),
        },
    };
    let result = remap(&target, &source).unwrap();
    assert_eq!(result.heights_m[0], 16.0);
    assert_eq!(result.heights_m[32 + 1], 22.0);
    assert!(result.heights_m[31].is_nan());
    assert!(result.heights_m[31 * 32].is_nan());
    assert_eq!(result.image, target.image);
    assert_ne!(result.map.manifest_sha256, target.map.manifest_sha256);
}
