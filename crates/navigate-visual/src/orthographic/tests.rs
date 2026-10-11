use super::*;
use crate::{CameraModel, Frame, FrameStamp, LocalizerConfig, PixelMatch, PosePrior, PoseVerifier};
use nalgebra::UnitQuaternion;

#[cfg(test)]
fn scene() -> (
    Frame,
    OrthographicReference,
    PosePrior,
    Vec<PixelMatch>,
    CameraPose,
) {
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: 450.0,
        fy: 460.0,
        cx: 319.5,
        cy: 239.5,
    };
    let geometry = RasterGeometry {
        origin: Vector2::new(-80.0, 60.0),
        column: Vector2::new(0.25, 0.0),
        row: Vector2::new(0.0, -0.25),
    };
    let heights = (0..480)
        .flat_map(|y| (0..640).map(move |x| 5.0 * ((x + y) as f32 * 0.03).sin()))
        .collect();
    let reference = OrthographicReference {
        map: MapRevision {
            release_id: "test".into(),
            manifest_sha256: "b".repeat(64),
        },
        image: GrayImage::new(640, 480),
        heights_m: heights,
        geometry,
    };
    let truth = CameraPose {
        position: Vector3::new(2.0, -3.0, 130.0),
        orientation: UnitQuaternion::from_euler_angles(0.03, -0.04, 0.2),
    };
    let prior = PosePrior {
        pose: CameraPose {
            position: Vector3::new(0.0, 0.0, 125.0),
            orientation: UnitQuaternion::identity(),
        },
        position_radius_m: 50.0,
        attitude_radius_rad: 0.5,
    };
    let mut pairs = Vec::new();
    for y in (50..430).step_by(40) {
        for x in (50..590).step_by(40) {
            let p = Vector2::new(f64::from(x), f64::from(y));
            let q = camera
                .project(&truth, reference.world_at(p).unwrap())
                .unwrap();
            if q.x > 0.0 && q.x < 639.0 && q.y > 0.0 && q.y < 479.0 {
                pairs.push(PixelMatch {
                    reference: p,
                    query: q,
                });
            }
        }
    }
    (
        Frame {
            camera,
            stamp: FrameStamp {
                sequence: 0,
                capture_time_ns: 1,
            },
            image: GrayImage::new(640, 480),
        },
        reference,
        prior,
        pairs,
        truth,
    )
}
#[test]
fn surface_pose_recovers_metric_translation_and_rotation_without_reference_camera() {
    let (frame, reference, prior, pairs, truth) = scene();
    let result = PoseVerifier::new(LocalizerConfig::default())
        .unwrap()
        .verify_orthographic(&frame, &reference, &prior, &pairs, "test")
        .unwrap();
    assert!((result.pose.position - truth.position).norm() < 1e-5);
    assert!(result.pose.orientation.angle_to(&truth.orientation) < 1e-6);
    assert!(result.quality.spatial_support >= 20);
    assert_eq!(result.map, reference.map);
    assert_eq!(result.observation_sha256, frame.evidence_sha256());
}
#[test]
fn missing_heights_duplicates_sparse_support_and_prior_violations_stay_rejected() {
    let (frame, mut reference, mut prior, pairs, _) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    for selection in [&pairs[..10], vec![pairs[0]; 100].as_slice()] {
        assert!(
            verifier
                .verify_orthographic(&frame, &reference, &prior, selection, "test")
                .is_err()
        );
    }
    prior.position_radius_m = 0.1;
    assert!(matches!(
        verifier.verify_orthographic(&frame, &reference, &prior, &pairs, "test"),
        Err(VisualError::OutsidePrior { .. })
    ));
    prior.position_radius_m = 50.0;
    reference.heights_m.fill(f32::NAN);
    assert!(
        verifier
            .verify_orthographic(&frame, &reference, &prior, &pairs, "test")
            .is_err()
    );
}
#[test]
fn raster_sampling_supports_negative_heights_and_rejects_invalid_geometry() {
    let (_, mut reference, _, _, _) = scene();
    reference.heights_m.fill(-10.0);
    let pixel = Vector2::new(12.25, 19.5);
    let point = reference.world_at(pixel).unwrap();
    assert_eq!(point.z, -10.0);
    assert!((reference.geometry.project(point).unwrap() - pixel).norm() < 1e-10);
    assert!(reference.world_at(Vector2::new(f64::NAN, 0.0)).is_none());
    assert!(reference.world_at(Vector2::new(640.0, 0.0)).is_none());
    reference.geometry.row = reference.geometry.column;
    assert!(reference.validate().is_err());
}
