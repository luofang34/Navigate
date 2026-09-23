#![allow(clippy::expect_used)]

use super::*;
use image::Luma;
use nalgebra::{UnitQuaternion, Vector3};
use navigate_visual::{CameraPose, MapRevision};

fn fixture() -> (VerifiedMatches, ReferenceView) {
    let image = GrayImage::from_pixel(8, 8, Luma([10]));
    let depth_m = vec![100.0; 64];
    let matches = VerifiedMatches {
        backend_identity: "verified-test-matcher".into(),
        reference_image_sha256: digest(image.as_raw()),
        query_image_sha256: digest(image.as_raw()),
        reference_depth_sha256: depth_digest(&depth_m),
        matches: vec![Pair {
            reference: [2.0, 3.0],
            query: [4.0, 5.0],
        }],
    };
    let reference = ReferenceView {
        frame: navigate_visual::LocalFrame::anchor_mercator(47.0, 11.0).expect("valid anchor"),
        map: MapRevision {
            release_id: "test".into(),
            manifest_sha256: "a".repeat(64),
        },
        pose: CameraPose {
            position: Vector3::new(0.0, 0.0, 100.0),
            orientation: UnitQuaternion::identity(),
        },
        image,
        depth_m,
    };
    (matches, reference)
}

#[test]
fn matched_pixels_require_the_exact_query_and_reference() {
    let (mut matches, reference) = fixture();
    assert_eq!(
        matches
            .match_images_blocking(&reference.image, &reference.image)
            .expect("bound images")
            .len(),
        1
    );
    let mut changed = reference.image.clone();
    changed.put_pixel(2, 3, Luma([11]));
    assert!(
        matches
            .match_images_blocking(&reference.image, &changed)
            .is_err()
    );
    assert!(
        matches
            .match_images_blocking(&changed, &reference.image)
            .is_err()
    );
}

#[test]
fn changed_terrain_depth_invalidates_correspondences() {
    let (matches, mut reference) = fixture();
    matches.validate_depth(&reference).expect("bound depth");
    reference.depth_m[0] = 101.0;
    assert!(matches.validate_depth(&reference).is_err());
}
