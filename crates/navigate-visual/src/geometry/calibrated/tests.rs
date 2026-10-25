use super::*;
use crate::{CameraPose, FrameStamp, LocalFrame, LocalizerConfig, MapRevision};
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector3};

#[test]
fn different_focal_lengths_and_image_sizes_keep_metric_geometry() {
    let (frame, reference, camera, prior, pairs, truth) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    assert!(
        verifier
            .verify(&frame, &reference, &prior, &pairs, "test")
            .is_err()
    );
    let result = verifier
        .verify_calibrated_reference(&frame, &reference, &camera, &prior, &pairs, "test")
        .unwrap();
    assert!((result.pose.position - truth.position).norm() < 1e-5);
    assert!(result.pose.orientation.angle_to(&truth.orientation) < 1e-6);
    assert!(result.quality.spatial_support >= 20);
    assert_eq!(result.map, reference.map);
    assert_eq!(result.observation_sha256, frame.evidence_sha256());
}
#[test]
fn different_calibrations_do_not_relax_support_or_prior_limits() {
    let (frame, mut reference, camera, mut prior, pairs, _) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let sparse = verifier.verify_calibrated_reference(
        &frame,
        &reference,
        &camera,
        &prior,
        &pairs[..10],
        "test",
    );
    assert!(matches!(
        sparse,
        Err(VisualError::InsufficientMatches { .. })
    ));
    prior.position_radius_m = 0.1;
    assert!(matches!(
        verifier.verify_calibrated_reference(&frame, &reference, &camera, &prior, &pairs, "test"),
        Err(VisualError::OutsidePrior { .. })
    ));
    reference.depth_m.fill(0.0);
    assert!(
        verifier
            .verify_calibrated_reference(&frame, &reference, &camera, &prior, &pairs, "test")
            .is_err()
    );
}
#[test]
fn nonfinite_and_duplicate_matches_cannot_supply_support() {
    let (frame, reference, camera, prior, pairs, _) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let mut repeated = vec![pairs[0]; 100];
    repeated.push(PixelMatch {
        reference: Vector2::new(f64::NAN, 10.0),
        query: pairs[1].query,
    });
    assert!(
        verifier
            .verify_calibrated_reference(&frame, &reference, &camera, &prior, &repeated, "test")
            .is_err()
    );
}
#[cfg(test)]
fn scene() -> (
    Frame,
    ReferenceView,
    CameraModel,
    PosePrior,
    Vec<PixelMatch>,
    CameraPose,
) {
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: 600.0,
        fy: 610.0,
        cx: 319.5,
        cy: 239.5,
    };
    let query = CameraModel {
        width: 320,
        height: 240,
        fx: 260.0,
        fy: 270.0,
        cx: 159.5,
        cy: 119.5,
    };
    let pose = CameraPose {
        position: Vector3::zeros(),
        orientation: UnitQuaternion::identity(),
    };
    let truth = CameraPose {
        position: Vector3::new(3.0, -2.0, 1.0),
        orientation: UnitQuaternion::from_euler_angles(0.01, -0.02, 0.015),
    };
    let depth: Vec<f32> = (0..480)
        .flat_map(|y| (0..640).map(move |x| 60.0 + 20.0 * ((x + y) as f32 * 0.01).sin()))
        .collect();
    let mut pairs = Vec::new();
    for y in (40..440).step_by(40) {
        for x in (40..600).step_by(40) {
            let p = Vector2::new(x as f64, y as f64);
            let world = camera.unproject(&pose, p, f64::from(depth[y * 640 + x]));
            if let Some(q) = query.project(&truth, world)
                && inside(q, &query)
            {
                pairs.push(PixelMatch {
                    reference: p,
                    query: q,
                });
            }
        }
    }
    let frame = Frame {
        camera: query,
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
        image: GrayImage::new(320, 240),
    };
    let reference = ReferenceView {
        map: MapRevision {
            release_id: "test".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(41.0, -75.0).unwrap(),
        pose,
        image: GrayImage::new(640, 480),
        depth_m: depth,
    };
    let prior = PosePrior {
        pose,
        position_radius_m: 20.0,
        attitude_radius_rad: 0.2,
    };
    (frame, reference, camera, prior, pairs, truth)
}

#[test]
fn a_shared_camera_keeps_existing_geometry_results_exact() {
    let (mut frame, reference, camera, prior, mut pairs, truth) = scene();
    frame.camera = camera;
    frame.image = GrayImage::new(camera.width, camera.height);
    for pair in &mut pairs {
        let x = pair.reference.x.round() as usize;
        let y = pair.reference.y.round() as usize;
        let world = camera.unproject(
            &reference.pose,
            pair.reference,
            f64::from(reference.depth_m[y * camera.width as usize + x]),
        );
        pair.query = camera.project(&truth, world).unwrap();
    }
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let old = verifier
        .verify(&frame, &reference, &prior, &pairs, "test")
        .unwrap();
    let calibrated = verifier
        .verify_calibrated_reference(&frame, &reference, &camera, &prior, &pairs, "test")
        .unwrap();
    assert_eq!(old.pose.position, calibrated.pose.position);
    assert_eq!(old.pose.orientation, calibrated.pose.orientation);
    assert_eq!(old.geometry_covariance, calibrated.geometry_covariance);
    assert_eq!(
        old.quality.spatial_support,
        calibrated.quality.spatial_support
    );
}
