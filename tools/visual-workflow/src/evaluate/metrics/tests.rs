use super::*;
use nalgebra::{UnitQuaternion, Vector2, Vector3};
use navigate_visual::{LocalFrame, MapRevision};

#[test]
fn spatial_membership_depends_only_on_fixed_reference_coordinates() {
    for x in 0..640 {
        for y in (0..480).step_by(17) {
            let pair = PixelMatch {
                reference: Vector2::new(x as f64, y as f64),
                query: Vector2::new(20.0, 30.0),
            };
            let moved = PixelMatch {
                query: Vector2::new(580.0, 400.0),
                ..pair
            };
            let held = (0..5)
                .filter(|fold| partition(&pair, *fold) == Some(true))
                .count();
            assert!(held <= 1);
            for fold in 0..5 {
                assert_eq!(partition(&pair, fold), partition(&moved, fold));
            }
        }
    }
    let edge = PixelMatch {
        reference: Vector2::new(64.0, 100.0),
        query: Vector2::zeros(),
    };
    assert_eq!(partition(&edge, 0), None);
}

#[test]
fn focal_perturbation_keeps_map_geometry_fixed() {
    let camera = camera();
    let pair = PixelMatch {
        reference: Vector2::new(80.0, 70.0),
        query: Vector2::new(120.0, 100.0),
    };
    let changed = calibrated(pair, &camera, 1.1);
    assert_eq!(changed.reference, pair.reference);
    assert!((changed.query.x - (64.0 + 56.0 / 1.1)).abs() < 1e-10);
}

#[test]
fn heldout_errors_include_bad_matches_and_wrong_poses() {
    let camera = camera();
    let pose = CameraPose {
        position: Vector3::zeros(),
        orientation: UnitQuaternion::identity(),
    };
    let reference = ReferenceView {
        map: MapRevision {
            release_id: "test".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(35.0, -115.0).unwrap(),
        pose,
        image: image::GrayImage::new(128, 128),
        depth_m: vec![100.0; 128 * 128],
    };
    let pairs: Vec<_> = [20.0, 40.0, 70.0, 90.0]
        .into_iter()
        .map(|x| PixelMatch {
            reference: Vector2::new(x, 60.0),
            query: Vector2::new(x, 60.0),
        })
        .collect();
    let correct = score(&pairs, &camera, &reference, &pose, 1.0);
    assert_eq!(correct["within_2_5_px"], 4);
    let wrong = CameraPose {
        position: Vector3::new(10.0, 0.0, 0.0),
        ..pose
    };
    let result = score(&pairs, &camera, &reference, &wrong, 1.0);
    assert_eq!(result["within_2_5_px"], 0);
    assert!((result["median_px"].as_f64().unwrap() - 10.0).abs() < 1e-10);
    let mut contaminated = pairs;
    contaminated[0].query.x += 50.0;
    let result = score(&contaminated, &camera, &reference, &pose, 1.0);
    assert_eq!(result["valid_depth"], 4);
    assert_eq!(result["within_2_5_px"], 3);
    assert_eq!(result["p90_px"], 50.0);
}

fn camera() -> CameraModel {
    CameraModel {
        width: 128,
        height: 128,
        fx: 100.0,
        fy: 100.0,
        cx: 64.0,
        cy: 64.0,
    }
}
