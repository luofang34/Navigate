//! Tests for traceable scene-to-reference association and geometric alternatives.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use crate::{
    CameraPose, LocalFrame, LocalSceneCamera, LocalScenePoint, LocalScenePose, MapRevision,
    ScenePointObservation,
};
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};

fn camera() -> CameraModel {
    CameraModel {
        width: 96,
        height: 72,
        fx: 70.0,
        fy: 70.0,
        cx: 47.5,
        cy: 35.5,
    }
}
fn transform() -> SceneTransform {
    SceneTransform {
        scale: 7.0,
        rotation: UnitQuaternion::from_euler_angles(0.3, -0.4, 1.1),
        translation: Vector3::new(20.0, -10.0, 100.0),
    }
}
fn fixture() -> (LocalScene, ReferenceView, Vec<PixelMatch>) {
    let camera = camera();
    let t = transform();
    let origin = CameraPose {
        position: Vector3::zeros(),
        orientation: UnitQuaternion::identity(),
    };
    let mut scene = LocalScene {
        cameras: (0..2)
            .map(|i| LocalSceneCamera {
                observation_sha256: format!("{i:064x}"),
                pose: LocalScenePose {
                    position: Vector3::new(i as f64 * 0.1, 0.0, 0.0),
                    orientation: UnitQuaternion::identity(),
                },
                fixed: i == 0,
            })
            .collect(),
        points: Vec::new(),
    };
    let mut reference = ReferenceView {
        map: MapRevision {
            release_id: "synthetic".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(40.0, -74.0).expect("valid anchor"),
        pose: CameraPose {
            position: t.translation,
            orientation: t.rotation,
        },
        image: GrayImage::new(camera.width, camera.height),
        depth_m: vec![0.0; (camera.width * camera.height) as usize],
    };
    let mut pairs = Vec::new();
    for i in 0..48 {
        let pixel = Vector2::new(8.0 + (i % 8) as f64 * 11.0, 6.0 + (i / 8) as f64 * 12.0);
        let depth = 5.0 + (i % 3) as f64;
        let position = camera.unproject(&origin, pixel, depth);
        scene.points.push(LocalScenePoint {
            feature_id: i as u64,
            position,
            observations: (0..2)
                .map(|j| ScenePointObservation {
                    camera_index: j,
                    pixel: camera
                        .project(
                            &CameraPose {
                                position: scene.cameras[j].pose.position,
                                orientation: scene.cameras[j].pose.orientation,
                            },
                            position,
                        )
                        .expect("visible point"),
                })
                .collect(),
        });
        reference.depth_m[pixel.y as usize * camera.width as usize + pixel.x as usize] =
            (depth * t.scale) as f32;
        pairs.push(PixelMatch {
            query: pixel,
            reference: pixel,
        });
    }
    (scene, reference, pairs)
}
fn propose(
    scene: &LocalScene,
    reference: &ReferenceView,
    pairs: &[PixelMatch],
) -> SceneRegistrationProposals {
    propose_scene_registration(
        &camera(),
        scene,
        &"0".repeat(64),
        reference,
        pairs,
        RegistrationConfig {
            trials: 256,
            ..RegistrationConfig::default()
        },
    )
    .expect("valid registration input")
}
#[test]
fn depth_links_recover_similarity_with_outliers_and_preserve_identity() {
    let (scene, mut reference, pairs) = fixture();
    for pair in &pairs[..8] {
        reference.depth_m[pair.reference.y as usize * 96 + pair.reference.x as usize] *= 3.0;
    }
    let result = propose(&scene, &reference, &pairs);
    let best = result.candidates.first().expect("supported transform");
    let expected = transform();
    assert_eq!(best.inlier_indices.len(), 40);
    assert!(best.occupied_cells >= 6);
    assert!((best.transform.scale - expected.scale).abs() < 1e-10);
    assert!(best.transform.rotation.angle_to(&expected.rotation) < 1e-10);
    assert!((best.transform.translation - expected.translation).norm() < 1e-10);
    assert!(best.inlier_rms_m < 1e-10);
    assert_eq!(result.map, reference.map);
    assert_eq!(result.frame, reference.frame);
    assert_eq!(result.observation_sha256, "0".repeat(64));
    assert_eq!(scene.cameras[0].pose.position, Vector3::zeros());
    assert_eq!(reference.pose.position, expected.translation);
}
#[test]
fn repeated_pairs_or_processing_do_not_add_support() {
    let (scene, reference, pairs) = fixture();
    let once = propose(&scene, &reference, &pairs);
    let repeated = propose(&scene, &reference, &pairs.repeat(3));
    assert_eq!(once.associations.len(), 48);
    assert_eq!(repeated.associations.len(), 48);
    assert_eq!(once.candidates.len(), repeated.candidates.len());
    assert_eq!(
        once.candidates[0].inlier_indices,
        repeated.candidates[0].inlier_indices
    );
    assert_eq!(
        once.candidates[0].inlier_rms_m,
        repeated.candidates[0].inlier_rms_m
    );
    assert_eq!(
        once.candidates[0].inlier_rms_m,
        propose(&scene, &reference, &pairs).candidates[0].inlier_rms_m
    );
}
#[test]
fn missing_surface_and_invalid_pairs_cannot_create_map_points() {
    let (scene, mut reference, mut pairs) = fixture();
    for (i, depth) in reference.depth_m.iter_mut().enumerate() {
        *depth = [0.0, -1.0, f32::NAN, f32::INFINITY][i % 4];
    }
    pairs.extend([
        PixelMatch {
            query: Vector2::new(f64::NAN, 0.0),
            reference: Vector2::zeros(),
        },
        PixelMatch {
            query: Vector2::zeros(),
            reference: Vector2::new(95.9, 71.9),
        },
    ]);
    assert!(matches!(
        propose_scene_registration(
            &camera(),
            &scene,
            &"0".repeat(64),
            &reference,
            &pairs,
            RegistrationConfig::default()
        ),
        Err(SceneRegistrationError::Support { found: 0, .. })
    ));
}
#[test]
fn last_pixel_strip_is_a_valid_scene_observation() {
    let (mut scene, mut reference, mut pairs) = fixture();
    let pixel = Vector2::new(95.25, 50.0);
    scene.points[0].position = camera().unproject(
        &CameraPose {
            position: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
        },
        pixel,
        5.0,
    );
    for observation in &mut scene.points[0].observations {
        observation.pixel = pixel;
    }
    pairs[0] = PixelMatch {
        query: pixel,
        reference: pixel,
    };
    reference.depth_m[50 * 96 + 95] = 35.0;
    assert_eq!(propose(&scene, &reference, &pairs).associations.len(), 48);
}
#[test]
fn duplicate_source_identities_are_rejected() {
    let (original, reference, pairs) = fixture();
    let mut scene = original.clone();
    scene.cameras[1].observation_sha256 = scene.cameras[0].observation_sha256.clone();
    assert!(matches!(
        propose_scene_registration(
            &camera(),
            &scene,
            &"0".repeat(64),
            &reference,
            &pairs,
            RegistrationConfig::default()
        ),
        Err(SceneRegistrationError::Input { .. })
    ));
    scene = original.clone();
    scene.points[1].feature_id = scene.points[0].feature_id;
    assert!(matches!(
        propose_scene_registration(
            &camera(),
            &scene,
            &"0".repeat(64),
            &reference,
            &pairs,
            RegistrationConfig::default()
        ),
        Err(SceneRegistrationError::Input { .. })
    ));
    scene = original;
    let duplicate = scene.points[0].observations[0].clone();
    scene.points[0].observations.push(duplicate);
    assert!(matches!(
        propose_scene_registration(
            &camera(),
            &scene,
            &"0".repeat(64),
            &reference,
            &pairs,
            RegistrationConfig::default()
        ),
        Err(SceneRegistrationError::Input { .. })
    ));
}
#[test]
fn unknown_observation_and_invalid_reference_are_rejected() {
    let (scene, mut reference, pairs) = fixture();
    assert!(matches!(
        propose_scene_registration(
            &camera(),
            &scene,
            &"f".repeat(64),
            &reference,
            &pairs,
            RegistrationConfig::default()
        ),
        Err(SceneRegistrationError::Observation { .. })
    ));
    reference.map.manifest_sha256 = "invalid".into();
    assert!(matches!(
        propose_scene_registration(
            &camera(),
            &scene,
            &"0".repeat(64),
            &reference,
            &pairs,
            RegistrationConfig::default()
        ),
        Err(SceneRegistrationError::Reference { .. })
    ));
    reference.map.manifest_sha256 = "a".repeat(64);
    reference.depth_m.pop();
    assert!(matches!(
        propose_scene_registration(
            &camera(),
            &scene,
            &"0".repeat(64),
            &reference,
            &pairs,
            RegistrationConfig::default()
        ),
        Err(SceneRegistrationError::Reference { .. })
    ));
}
#[test]
fn conflicting_locations_remain_separate_and_truncation_is_reported() {
    let (scene, reference, pairs) = fixture();
    let mut points = propose(&scene, &reference, &pairs).associations;
    let shift = Vector3::new(300.0, -100.0, 70.0);
    for p in &mut points[24..] {
        p.map_point += shift;
    }
    let config = RegistrationConfig {
        trials: 512,
        min_occupied_cells: 4,
        ..RegistrationConfig::default()
    };
    let (candidates, truncated) = consensus::propose(&camera(), &points, config);
    assert!(!truncated);
    assert_eq!(candidates.len(), 2);
    for expected in [transform().translation, transform().translation + shift] {
        assert!(
            candidates
                .iter()
                .any(|c| (c.transform.translation - expected).norm() < 1e-9)
        );
    }
    assert!(candidates.iter().all(|c| c.inlier_indices.len() == 24));
    let (one, truncated) = consensus::propose(
        &camera(),
        &points,
        RegistrationConfig {
            max_candidates: 1,
            ..config
        },
    );
    assert_eq!(one.len(), 1);
    assert!(truncated);
}
#[test]
fn collinear_points_and_concentrated_support_do_not_propose_a_transform() {
    let (scene, reference, pairs) = fixture();
    let original = propose(&scene, &reference, &pairs).associations;
    let mut points = original.clone();
    for (i, p) in points.iter_mut().enumerate() {
        p.scene_point = Vector3::new(i as f64, 0.0, 0.0);
        p.map_point = transform().point(p.scene_point);
    }
    assert!(
        consensus::propose(&camera(), &points, RegistrationConfig::default())
            .0
            .is_empty()
    );
    points = original;
    for p in &mut points {
        p.query_pixel = Vector2::new(1.0, 1.0);
    }
    assert!(
        consensus::propose(&camera(), &points, RegistrationConfig::default())
            .0
            .is_empty()
    );
}
