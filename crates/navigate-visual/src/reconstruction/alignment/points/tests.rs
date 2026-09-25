//! Checks for shared-point alignment and evidence identity.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use crate::{LocalSceneCamera, LocalScenePoint, LocalScenePose, ScenePointObservation};
use nalgebra::Vector2;
fn scene() -> LocalScene {
    let cameras = (0..6)
        .map(|i| LocalSceneCamera {
            observation_sha256: format!("{i:064x}"),
            fixed: false,
            pose: LocalScenePose {
                position: Vector3::zeros(),
                orientation: UnitQuaternion::from_euler_angles(
                    i as f64 * 0.02,
                    i as f64 * -0.03,
                    i as f64 * 0.04,
                ),
            },
        })
        .collect::<Vec<_>>();
    let points = (0..24)
        .map(|i| {
            let position = Vector3::new(
                (i % 6) as f64 * 0.6 - 1.5,
                (i / 6) as f64 * 0.5 - 0.8,
                -8.0 - (i % 3) as f64,
            );
            let observations = cameras
                .iter()
                .enumerate()
                .map(|(camera_index, c)| {
                    let eye = c.pose.orientation.inverse() * position;
                    ScenePointObservation {
                        camera_index,
                        pixel: Vector2::new(
                            480.0 - 700.0 * eye.x / eye.z,
                            272.0 + 700.0 * eye.y / eye.z,
                        ),
                    }
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
fn expected() -> SceneTransform {
    SceneTransform {
        scale: 3.7,
        rotation: UnitQuaternion::from_euler_angles(0.4, -0.7, 1.3),
        translation: Vector3::new(4.0, 8.0, -3.0),
    }
}
fn mapped(source: &LocalScene, transform: SceneTransform) -> LocalScene {
    LocalScene {
        cameras: source
            .cameras
            .iter()
            .map(|c| LocalSceneCamera {
                pose: transform.pose(c.pose),
                ..c.clone()
            })
            .collect(),
        points: source
            .points
            .iter()
            .map(|p| LocalScenePoint {
                feature_id: p.feature_id + 100,
                position: transform.point(p.position),
                observations: p.observations.clone(),
            })
            .collect(),
    }
}
fn pairs(source: &LocalScene, target: &LocalScene) -> Vec<ScenePointAssociation> {
    associate_scene_points(source, target, PointAssociationConfig::default())
        .expect("shared observations")
}
#[test]
fn scene_points_supply_scale_without_shared_camera_translation() {
    let source = scene();
    let target = mapped(&source, expected());
    assert!(matches!(
        super::super::align_scenes(&source, &target, Default::default()),
        Err(SceneAlignmentError::Baseline)
    ));
    let links = pairs(&source, &target);
    assert_eq!(links.len(), 24);
    let result = align_scenes_with_points(&source, &target, &links, Default::default())
        .expect("supported point depth");
    assert_eq!(result.candidates.len(), 1);
    assert!(!result.candidate_budget_exhausted);
    let first = &result.candidates[0];
    let fit = first.alignment.transform;
    assert!((fit.scale - 3.7).abs() < 1e-10);
    assert!(fit.rotation.angle_to(&expected().rotation) < 1e-10);
    assert!((fit.translation - expected().translation).norm() < 1e-10);
    assert_eq!(first.alignment.target_baseline_scene_units, 0.0);
    assert_eq!(first.associations.len(), 24);
    assert!(first.point_rms_scene_units < 1e-10);
    assert_eq!(source.cameras[0].pose.position, Vector3::zeros());
    let again = align_scenes_with_points(&source, &target, &links, Default::default())
        .expect("same evidence");
    assert_eq!(again.candidates[0].associations, first.associations);
    assert_eq!(
        again.candidates[0].alignment.observation_sha256,
        first.alignment.observation_sha256
    );
}
#[test]
fn wrong_point_depth_is_excluded_without_moving_camera_rotation() {
    let source = scene();
    let mut target = mapped(&source, expected());
    target.points[0].position += Vector3::new(20.0, -15.0, 10.0);
    let result = align_scenes_with_points(
        &source,
        &target,
        &pairs(&source, &target),
        Default::default(),
    )
    .expect("other points agree");
    assert_eq!(result.candidates[0].associations.len(), 23);
    assert_eq!(
        result.candidates[0].excluded_associations[0].source_feature_id,
        0
    );
    assert!(
        result.candidates[0]
            .alignment
            .transform
            .rotation
            .angle_to(&expected().rotation)
            < 1e-10
    );
}
#[test]
fn different_supported_scales_remain_alternatives_and_limits_are_reported() {
    let source = scene();
    let mut target = mapped(&source, expected());
    let other = SceneTransform {
        scale: 7.4,
        ..expected()
    };
    for (i, p) in target.points.iter_mut().enumerate().skip(12) {
        p.position = other.point(source.points[i].position)
    }
    let links = pairs(&source, &target);
    let result = align_scenes_with_points(&source, &target, &links, Default::default())
        .expect("two depth alternatives");
    assert_eq!(result.candidates.len(), 2);
    assert!(
        result
            .candidates
            .iter()
            .any(|c| (c.alignment.transform.scale - 3.7).abs() < 1e-10)
    );
    assert!(
        result
            .candidates
            .iter()
            .any(|c| (c.alignment.transform.scale - 7.4).abs() < 1e-10)
    );
    let limited = align_scenes_with_points(
        &source,
        &target,
        &links,
        PointAlignmentConfig {
            max_candidates: 1,
            ..Default::default()
        },
    )
    .expect("bounded output");
    assert_eq!(limited.candidates.len(), 1);
    assert!(limited.candidate_budget_exhausted);
}
#[test]
fn repeated_points_or_image_pixels_cannot_manufacture_support() {
    let source = scene();
    let target = mapped(&source, expected());
    let mut links = pairs(&source, &target);
    links.push(links[0]);
    assert!(matches!(
        align_scenes_with_points(&source, &target, &links, Default::default()),
        Err(SceneAlignmentError::Association { .. })
    ));
    let mut duplicated = source.clone();
    duplicated.points[1].observations = duplicated.points[0].observations.clone();
    let target = mapped(&duplicated, expected());
    let links: Vec<_> = duplicated
        .points
        .iter()
        .map(|p| ScenePointAssociation {
            source_feature_id: p.feature_id,
            target_feature_id: p.feature_id + 100,
        })
        .collect();
    assert!(matches!(
        align_scenes_with_points(&duplicated, &target, &links, Default::default()),
        Err(SceneAlignmentError::Association { .. })
    ));
    assert_eq!(
        pairs(&duplicated, &target).len(),
        22,
        "equal nearest pixels do not choose an arbitrary point"
    );
}
#[test]
fn wrong_images_bad_pixels_and_conflicting_camera_rotations_do_not_join() {
    let source = scene();
    let mut target = mapped(&source, expected());
    let links = pairs(&source, &target);
    for point in &mut target.points {
        for o in &mut point.observations {
            o.pixel.x += 20.0
        }
    }
    assert!(pairs(&source, &target).is_empty());
    assert!(matches!(
        align_scenes_with_points(&source, &target, &links, Default::default()),
        Err(SceneAlignmentError::PointSupport { retained: 0, .. })
    ));
    target = mapped(&source, expected());
    for (i, c) in target.cameras.iter_mut().enumerate() {
        c.pose.orientation *= UnitQuaternion::from_euler_angles(i as f64 * 0.2, 0.0, 0.0)
    }
    assert!(align_scenes_with_points(&source, &target, &links, Default::default()).is_err());
    target = mapped(&source, expected());
    for c in &mut target.cameras {
        c.observation_sha256 = "f".repeat(64)
    }
    assert!(associate_scene_points(&source, &target, Default::default()).is_err());
}
#[test]
fn invalid_records_and_weak_point_extent_are_rejected() {
    let source = scene();
    let mut target = mapped(&source, expected());
    let repeated = target.points[0].observations[0].clone();
    target.points[0].observations.push(repeated);
    assert!(matches!(
        associate_scene_points(&source, &target, Default::default()),
        Err(SceneAlignmentError::Point { .. })
    ));
    let mut small = source.clone();
    for p in &mut small.points {
        p.position = Vector3::new(0.0, 0.0, -8.0) + p.position * 1e-8
    }
    let target = mapped(&small, expected());
    assert!(matches!(
        align_scenes_with_points(&small, &target, &pairs(&small, &target), Default::default()),
        Err(SceneAlignmentError::Baseline)
    ));
    assert!(matches!(
        align_scenes_with_points(
            &source,
            &mapped(&source, expected()),
            &[],
            PointAlignmentConfig {
                max_point_error_ratio: f64::NAN,
                ..Default::default()
            }
        ),
        Err(SceneAlignmentError::Config)
    ));
}
