use super::*;
use crate::{CameraModel, LocalizerConfig};

fn map(camera: CameraModel, scale: f64) -> RectificationMap {
    let source = CameraModel {
        width: (f64::from(camera.width) * scale) as u32 + 1,
        height: (f64::from(camera.height) * scale) as u32 + 1,
        fx: camera.fx * scale,
        fy: camera.fy * scale,
        cx: camera.cx * scale,
        cy: camera.cy * scale,
    };
    RectificationMap::new(source, camera, [0.0; 5], 8 * 640 * 480).unwrap()
}
#[test]
fn common_support_preserves_a_real_fitted_pose_and_its_identity() {
    let (frame, reference, prior, pairs, truth) = crate::orthographic::tests::scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let mut estimate = verifier
        .verify_orthographic(&frame, &reference, &prior, &pairs, "test")
        .unwrap();
    let pose = estimate.pose;
    let source = CameraModel {
        width: 1600,
        height: 1280,
        fx: 900.0,
        fy: 920.0,
        cx: 800.0,
        cy: 640.0,
    };
    let distorted = RectificationMap::new(
        source,
        frame.camera,
        [0.04, 0.01, 0.001, -0.001, 0.001],
        640 * 480 * 8,
    )
    .unwrap();
    let raw = verifier
        .check_rectified_orthographic(
            &frame,
            &reference,
            &prior,
            &pairs,
            &distorted,
            &mut estimate,
        )
        .unwrap();
    assert_eq!(estimate.pose.position, pose.position);
    assert_eq!(estimate.pose.orientation, pose.orientation);
    assert!((estimate.pose.position - truth.position).norm() < 1e-5);
    assert_eq!(raw.spatial_support, estimate.quality.spatial_support);
    assert!(raw.query_cells >= 4 && raw.reference_cells >= 4);
    assert!(raw.reprojection_rms_px < 1e-5);
    assert_eq!(estimate.observation_sha256, frame.evidence_sha256());
}
#[test]
fn a_pinhole_pass_can_fail_raw_residuals_without_mutating_the_estimate() {
    let (frame, reference, prior, mut pairs, _) = crate::orthographic::tests::scene();
    for (i, p) in pairs.iter_mut().enumerate() {
        p.query.x += if i % 2 == 0 { 1.4 } else { -1.4 };
    }
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let mut estimate = verifier
        .verify_orthographic(&frame, &reference, &prior, &pairs, "test")
        .unwrap();
    let covariance = estimate.geometry_covariance;
    let inliers = estimate.quality.inliers;
    let error = verifier
        .check_rectified_orthographic(
            &frame,
            &reference,
            &prior,
            &pairs,
            &map(frame.camera, 4.0),
            &mut estimate,
        )
        .unwrap_err();
    assert!(matches!(error, VisualError::InsufficientMatches { .. }));
    assert_eq!(estimate.geometry_covariance, covariance);
    assert_eq!(estimate.quality.inliers, inliers);
}
#[test]
fn raw_separation_is_required_even_with_exact_projections() {
    let (frame, reference, prior, pairs, _) = crate::orthographic::tests::scene();
    let verifier = PoseVerifier::new(LocalizerConfig {
        min_inliers: 80,
        ..LocalizerConfig::default()
    })
    .unwrap();
    let mut estimate = verifier
        .verify_orthographic(&frame, &reference, &prior, &pairs, "test")
        .unwrap();
    let error = verifier
        .check_rectified_orthographic(
            &frame,
            &reference,
            &prior,
            &pairs,
            &map(frame.camera, 0.25),
            &mut estimate,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        VisualError::InsufficientSpatialSupport { .. }
    ));
}
#[test]
fn wrong_camera_source_identity_and_missing_heights_cannot_pass() {
    let (frame, mut reference, prior, pairs, _) = crate::orthographic::tests::scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let mut estimate = verifier
        .verify_orthographic(&frame, &reference, &prior, &pairs, "test")
        .unwrap();
    let mut other_camera = frame.camera;
    other_camera.fx += 1.0;
    assert!(
        verifier
            .check_rectified_orthographic(
                &frame,
                &reference,
                &prior,
                &pairs,
                &map(other_camera, 2.0),
                &mut estimate
            )
            .is_err()
    );
    let map = map(frame.camera, 2.0);
    estimate.observation_sha256 = "f".repeat(64);
    assert!(
        verifier
            .check_rectified_orthographic(&frame, &reference, &prior, &pairs, &map, &mut estimate)
            .is_err()
    );
    estimate.observation_sha256 = frame.evidence_sha256();
    reference.heights_m.fill(f32::NAN);
    assert!(
        verifier
            .check_rectified_orthographic(&frame, &reference, &prior, &pairs, &map, &mut estimate)
            .is_err()
    );
}

#[test]
fn raw_coverage_uses_sensor_dimensions_instead_of_the_crop() {
    let (frame, reference, prior, pairs, _) = crate::orthographic::tests::scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let mut estimate = verifier
        .verify_orthographic(&frame, &reference, &prior, &pairs, "test")
        .unwrap();
    let source = CameraModel {
        width: 3200,
        height: 2400,
        cx: frame.camera.cx + 1280.0,
        cy: frame.camera.cy + 960.0,
        ..frame.camera
    };
    let map = RectificationMap::new(source, frame.camera, [0.0; 5], 640 * 480 * 8).unwrap();
    assert!(matches!(
        verifier.check_rectified_orthographic(
            &frame,
            &reference,
            &prior,
            &pairs,
            &map,
            &mut estimate
        ),
        Err(VisualError::DegenerateGeometry)
    ));
}
