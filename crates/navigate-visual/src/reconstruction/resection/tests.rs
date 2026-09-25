//! Pose and evidence controls for scene-camera fitting.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use nalgebra::UnitQuaternion;

fn fixture() -> (
    CameraModel,
    LocalScenePose,
    LocalScenePose,
    Vec<ScenePointMatch>,
) {
    let camera = CameraModel {
        width: 960,
        height: 544,
        fx: 700.0,
        fy: 690.0,
        cx: 479.5,
        cy: 271.5,
    };
    let expected = LocalScenePose {
        position: Vector3::new(17.0, -8.0, 6.0),
        orientation: UnitQuaternion::from_euler_angles(1.1, -0.4, 0.7),
    };
    let initial = LocalScenePose {
        position: expected.position + Vector3::new(0.1, -0.08, 0.03),
        orientation: expected.orientation * UnitQuaternion::from_euler_angles(0.01, -0.02, 0.01),
    };
    let pose = Pose::from_scene(expected);
    let points = (0..150)
        .map(|i| {
            let eye = Vector3::new(
                (i % 15) as f64 / 9.0 - 0.7,
                (i / 15) as f64 / 8.0 - 0.5,
                -3.0 - (i % 7) as f64 * 0.2,
            );
            let position = expected.position + expected.orientation * eye;
            let pixel = pose
                .project(&camera, position)
                .expect("visible fixture point")
                + if i % 5 == 0 {
                    Vector2::new(25.0, -19.0)
                } else {
                    Vector2::new(0.05, -0.08)
                };
            ScenePointMatch {
                feature_id: i,
                position,
                pixel,
            }
        })
        .collect();
    (camera, expected, initial, points)
}

#[test]
fn fits_arbitrary_orientation_and_retains_source_evidence() {
    let (camera, expected, initial, points) = fixture();
    let fit = refit_scene_camera(
        &camera,
        &"AB".repeat(32),
        &"CD".repeat(32),
        initial,
        &points,
    )
    .expect("valid request")
    .expect("supported fit");
    assert_eq!(fit.scene_sha256, "ab".repeat(32));
    assert_eq!(fit.observation_sha256, "cd".repeat(32));
    assert_eq!(fit.inlier_feature_ids.len(), 120);
    assert!(fit.inlier_feature_ids.iter().all(|id| id % 5 != 0));
    assert!((fit.pose.position - expected.position).norm() < 0.005);
    assert!(fit.pose.orientation.angle_to(&expected.orientation) < 0.001);
    assert!(fit.reprojection_rms_px < 0.01);
    let repeated = refit_scene_camera(
        &camera,
        &fit.scene_sha256,
        &fit.observation_sha256,
        initial,
        &points,
    )
    .expect("same request")
    .expect("same fit");
    assert_eq!(repeated.inlier_feature_ids, fit.inlier_feature_ids);
    assert_eq!(repeated.pose.position, fit.pose.position);
    assert_eq!(repeated.reprojection_rms_px, fit.reprojection_rms_px);
}

#[test]
fn rejects_repeated_evidence_and_invalid_points() {
    let (camera, _, initial, mut points) = fixture();
    let call = |p: &[ScenePointMatch]| {
        refit_scene_camera(&camera, &"ab".repeat(32), &"cd".repeat(32), initial, p)
    };
    let original = points[1].clone();
    points[1].feature_id = points[0].feature_id;
    assert!(matches!(
        call(&points),
        Err(SceneResectionError::Match {
            reason: "repeated feature identity",
            ..
        })
    ));
    points[1] = original.clone();
    points[1].pixel = points[0].pixel;
    assert!(matches!(
        call(&points),
        Err(SceneResectionError::Match {
            reason: "repeated image pixel",
            ..
        })
    ));
    points[1] = original.clone();
    points[1].position.x = f64::NAN;
    assert!(matches!(
        call(&points),
        Err(SceneResectionError::Match { .. })
    ));
    points[1] = original;
    points[1].pixel.x = 960.0;
    assert!(matches!(
        call(&points),
        Err(SceneResectionError::Match { .. })
    ));
}

#[test]
fn insufficient_support_stays_unresolved() {
    let (camera, _, initial, points) = fixture();
    assert!(
        refit_scene_camera(
            &camera,
            &"ab".repeat(32),
            &"cd".repeat(32),
            initial,
            &points[..19]
        )
        .expect("valid insufficient links")
        .is_none()
    );
    assert!(matches!(
        refit_scene_camera(&camera, "unknown", &"cd".repeat(32), initial, &points),
        Err(SceneResectionError::Identity { kind: "scene" })
    ));
}
