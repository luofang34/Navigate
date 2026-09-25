//! Behavior checks for conditional group coordinate alignment.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use crate::LocalSceneCamera;
fn source() -> LocalScene {
    LocalScene {
        cameras: (0..10)
            .map(|i| LocalSceneCamera {
                observation_sha256: format!("{i:064x}"),
                pose: LocalScenePose {
                    position: Vector3::new(
                        i as f64 * 0.2,
                        (i as f64 * 0.12).sin(),
                        i as f64 * 0.05,
                    ),
                    orientation: UnitQuaternion::from_euler_angles(
                        0.05 * i as f64,
                        0.03 * i as f64,
                        -0.08 * i as f64,
                    ),
                },
                fixed: i == 0,
            })
            .collect(),
        points: Vec::new(),
    }
}
fn transform() -> SceneTransform {
    SceneTransform {
        scale: 7.0,
        rotation: UnitQuaternion::from_euler_angles(0.3, -0.4, 1.1),
        translation: Vector3::new(20.0, -10.0, 3.0),
    }
}
fn mapped(source: &LocalScene, t: SceneTransform) -> LocalScene {
    LocalScene {
        cameras: source
            .cameras
            .iter()
            .map(|c| LocalSceneCamera {
                pose: t.pose(c.pose),
                ..c.clone()
            })
            .collect(),
        points: Vec::new(),
    }
}
#[test]
fn shared_poses_recover_scale_rotation_and_translation_without_mutation() {
    let source = source();
    let expected = transform();
    let mut target = mapped(&source, expected);
    target.cameras.reverse();
    let result = align_scenes(&source, &target, AlignmentConfig::default())
        .expect("consistent shared estimates");
    assert_eq!(result.observation_sha256.len(), 10);
    assert!(result.excluded_observation_sha256.is_empty());
    assert!((result.transform.scale - expected.scale).abs() < 1e-10);
    assert!(result.transform.rotation.angle_to(&expected.rotation) < 1e-10);
    assert!((result.transform.translation - expected.translation).norm() < 1e-10);
    assert!(result.position_rms_scene_units < 1e-10);
    assert!(result.rotation_rms_rad < 1e-10);
    assert_eq!(source.cameras[0].pose.position, Vector3::zeros());
    assert_eq!(target.cameras[0].observation_sha256, format!("{:064x}", 9));
    let point = Vector3::new(0.2, -0.7, 1.3);
    assert!((result.transform.point(point) - expected.point(point)).norm() < 1e-10);
}
#[test]
fn shared_observations_with_inconsistent_poses_are_reported() {
    let source = source();
    let mut target = mapped(&source, transform());
    target.cameras[0].pose.orientation *= UnitQuaternion::from_euler_angles(0.7, 0.0, 0.0);
    target.cameras[9].pose.position += Vector3::new(3.0, -2.0, 4.0);
    let result = align_scenes(&source, &target, AlignmentConfig::default())
        .expect("eight consistent observations");
    assert_eq!(result.observation_sha256.len(), 8);
    assert_eq!(
        result.excluded_observation_sha256,
        vec![format!("{:064x}", 0), format!("{:064x}", 9)]
    );
    assert!((result.transform.scale - 7.0).abs() < 1e-10);
}
#[test]
fn one_shared_camera_or_stationary_overlap_cannot_invent_scale() {
    let mut source = source();
    let mut target = mapped(&source, transform());
    target.cameras.truncate(1);
    assert!(matches!(
        align_scenes(&source, &target, AlignmentConfig::default()),
        Err(SceneAlignmentError::Support { retained: 1, .. })
    ));
    for camera in &mut source.cameras {
        camera.pose.position = Vector3::zeros();
    }
    target = mapped(&source, transform());
    assert!(matches!(
        align_scenes(&source, &target, AlignmentConfig::default()),
        Err(SceneAlignmentError::Baseline)
    ));
}
#[test]
fn tiny_overlap_motion_and_conflicting_coordinate_frames_are_rejected() {
    let mut source = source();
    for camera in &mut source.cameras {
        camera.pose.position *= 1e-6;
    }
    let mut far = source.cameras[0].clone();
    far.observation_sha256 = "f".repeat(64);
    far.pose.position = Vector3::new(100.0, 0.0, 0.0);
    source.cameras.push(far);
    let mut target = mapped(&source, transform());
    target.cameras.pop();
    assert!(matches!(
        align_scenes(&source, &target, AlignmentConfig::default()),
        Err(SceneAlignmentError::Baseline)
    ));
    let source = super::tests::source();
    let mut target = mapped(&source, transform());
    for (i, camera) in target.cameras.iter_mut().enumerate() {
        camera.pose.orientation *= UnitQuaternion::from_euler_angles(i as f64 * 0.2, 0.0, 0.0);
    }
    assert!(align_scenes(&source, &target, AlignmentConfig::default()).is_err());
}
#[test]
fn duplicate_evidence_and_invalid_limits_are_rejected() {
    let source = source();
    let mut target = mapped(&source, transform());
    target.cameras[0].observation_sha256 = "a".repeat(64);
    target.cameras[1].observation_sha256 = "A".repeat(64);
    assert!(matches!(
        align_scenes(&source, &target, AlignmentConfig::default()),
        Err(SceneAlignmentError::Camera { index: 1, .. })
    ));
    assert!(matches!(
        align_scenes(
            &source,
            &source,
            AlignmentConfig {
                max_rotation_error_rad: f64::NAN,
                ..AlignmentConfig::default()
            }
        ),
        Err(SceneAlignmentError::Config)
    ));
}
