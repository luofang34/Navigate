#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use nalgebra::{UnitQuaternion, Vector2, Vector3};
fn camera() -> CameraModel {
    CameraModel {
        width: 960,
        height: 544,
        fx: 700.0,
        fy: 690.0,
        cx: 479.5,
        cy: 271.5,
    }
}
fn fixture() -> LocalScene {
    let cameras = (0..6)
        .map(|i| {
            let r = UnitQuaternion::from_euler_angles(
                i as f64 * 0.035,
                i as f64 * 0.025,
                -i as f64 * 0.03,
            )
            .to_rotation_matrix()
            .into_inner();
            LocalSceneCamera {
                observation_sha256: format!("{i:064x}"),
                pose: pose::Pose {
                    r,
                    t: Vector3::new(i as f64 * 0.25, 0.05 * i as f64, 0.0),
                }
                .to_scene(),
                fixed: i < 2,
            }
        })
        .collect::<Vec<_>>();
    let points = (0..80)
        .map(|i| {
            let position = Vector3::new(
                (i % 10) as f64 * 0.18 - 0.9,
                (i / 10) as f64 * 0.15 - 0.5,
                4.0 + (i % 7) as f64 * 0.2,
            );
            let observations = cameras
                .iter()
                .enumerate()
                .filter_map(|(camera_index, c)| {
                    pose::Pose::from_scene(c.pose)
                        .project(&camera(), position)
                        .map(|pixel| ScenePointObservation {
                            camera_index,
                            pixel,
                        })
                })
                .collect();
            LocalScenePoint {
                feature_id: i,
                position,
                observations,
            }
        })
        .collect();
    LocalScene { cameras, points }
}
fn perturb(scene: &mut LocalScene) {
    for c in scene.cameras.iter_mut().filter(|c| !c.fixed) {
        c.pose.position += Vector3::new(0.025, -0.018, 0.01);
        c.pose.orientation *= UnitQuaternion::from_euler_angles(0.004, -0.006, 0.009);
    }
    for p in &mut scene.points {
        p.position += Vector3::new(0.012, 0.008, -0.02)
    }
}
#[test]
fn refines_free_poses_and_points_without_rewriting_evidence() {
    let expected = fixture();
    let mut scene = expected.clone();
    perturb(&mut scene);
    let result = refine_local_scene(&camera(), &mut scene, 40).expect("conditional refinement");
    assert!(result.final_cost < 1e-5);
    assert!(result.final_cost < result.initial_cost);
    for (a, b) in scene.cameras.iter().zip(&expected.cameras) {
        assert_eq!(a.observation_sha256, b.observation_sha256);
        assert_eq!(a.fixed, b.fixed);
        assert!((a.pose.position - b.pose.position).norm() < 1e-4);
        assert!(a.pose.orientation.angle_to(&b.pose.orientation) < 1e-5);
        if a.fixed {
            assert_eq!(a.pose.orientation.coords, b.pose.orientation.coords);
            assert_eq!(a.pose.position, b.pose.position)
        }
    }
    for (a, b) in scene.points.iter().zip(&expected.points) {
        assert_eq!(a.feature_id, b.feature_id);
        assert!((a.position - b.position).norm() < 1e-4);
        for (x, y) in a.observations.iter().zip(&b.observations) {
            assert_eq!(x.camera_index, y.camera_index);
            assert_eq!(x.pixel, y.pixel)
        }
    }
}
#[test]
fn rejects_repeated_capture_identity_before_mutation() {
    let mut scene = fixture();
    scene.cameras[2].observation_sha256 = scene.cameras[1].observation_sha256.clone();
    let original = scene.cameras[3].pose.position;
    assert!(matches!(
        refine_local_scene(&camera(), &mut scene, 10),
        Err(LocalSceneError::Observation { index: 2, .. })
    ));
    assert_eq!(scene.cameras[3].pose.position, original);
}
#[test]
fn rejects_repeated_point_observation_before_mutation() {
    let mut scene = fixture();
    let repeated = scene.points[0].observations[0].clone();
    scene.points[0].observations.push(repeated);
    let original = scene.points[0].position;
    assert!(matches!(
        refine_local_scene(&camera(), &mut scene, 10),
        Err(LocalSceneError::Point { feature_id: 0, .. })
    ));
    assert_eq!(scene.points[0].position, original);
}
#[test]
fn missing_boundary_observations_do_not_constrain_the_scene() {
    let mut scene = fixture();
    for point in &mut scene.points {
        point.observations.retain(|o| o.camera_index >= 2)
    }
    assert!(matches!(
        refine_local_scene(&camera(), &mut scene, 10),
        Err(LocalSceneError::Unconstrained { .. })
    ));
}
#[test]
fn zero_baseline_and_disconnected_components_stay_unconstrained() {
    let mut scene = fixture();
    scene.cameras[1].pose.position = scene.cameras[0].pose.position;
    assert!(matches!(
        refine_local_scene(&camera(), &mut scene, 10),
        Err(LocalSceneError::Unconstrained { .. })
    ));
    let mut scene = fixture();
    for (i, point) in scene.points.iter_mut().enumerate() {
        point.observations.retain(|o| {
            if i % 2 == 0 {
                o.camera_index < 3
            } else {
                o.camera_index >= 3
            }
        });
    }
    assert!(matches!(
        refine_local_scene(&camera(), &mut scene, 10),
        Err(LocalSceneError::Unconstrained { .. })
    ));
    scene.cameras[3].fixed = true;
    scene.cameras[4].fixed = true;
    assert!(refine_local_scene(&camera(), &mut scene, 10).is_ok());
}
#[test]
fn robust_fit_limits_the_effect_of_bad_correspondences() {
    let expected = fixture();
    let mut scene = expected.clone();
    perturb(&mut scene);
    for (i, point) in scene.points.iter_mut().enumerate() {
        if i % 7 == 0 {
            point.observations[3].pixel += Vector2::new(40.0, -30.0)
        }
    }
    let result = refine_local_scene(&camera(), &mut scene, 40).expect("robust refinement");
    assert!(result.final_cost < result.initial_cost);
    for (a, b) in scene.cameras.iter().zip(&expected.cameras) {
        assert!((a.pose.position - b.pose.position).norm() < 0.04);
        assert!(a.pose.orientation.angle_to(&b.pose.orientation) < 0.01)
    }
}
#[test]
fn invalid_calibration_and_pixels_are_rejected() {
    let mut scene = fixture();
    let mut invalid = camera();
    invalid.fx = f64::NAN;
    assert!(matches!(
        refine_local_scene(&invalid, &mut scene, 10),
        Err(LocalSceneError::Camera { .. })
    ));
    scene.points[0].observations[0].pixel.x = -1.0;
    assert!(matches!(
        refine_local_scene(&camera(), &mut scene, 10),
        Err(LocalSceneError::Point { feature_id: 0, .. })
    ));
}
