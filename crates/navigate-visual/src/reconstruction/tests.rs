//! Behavioral checks for local reconstruction and observation identity.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use nalgebra::{Matrix3, UnitQuaternion};
fn fixture() -> (CameraModel, ImageTracks, Vec<Pose>, ReconstructionSeed) {
    let camera = CameraModel {
        width: 1280,
        height: 720,
        fx: 820.0,
        fy: 810.0,
        cx: 639.5,
        cy: 359.5,
    };
    let poses: Vec<_> = (0..5)
        .map(|i| Pose {
            r: UnitQuaternion::from_euler_angles(
                0.01 * i as f64,
                -0.035 * i as f64,
                0.018 * i as f64,
            )
            .to_rotation_matrix()
            .into_inner(),
            t: Vector3::new(-0.65 * i as f64, 0.02 * i as f64, 0.015 * i as f64),
        })
        .collect();
    let tracks = (0..480)
        .filter_map(|id| {
            let world = Vector3::new(
                (id % 24) as f64 * 0.28 - 2.8,
                (id / 24) as f64 * 0.22 - 1.9,
                8.0 + (id % 11) as f64 * 0.41,
            );
            let observations: Vec<_> = poses
                .iter()
                .enumerate()
                .filter_map(|(camera_index, p)| {
                    let pixel = p.project(&camera, world)?;
                    ((0.0..1280.0).contains(&pixel.x) && (0.0..720.0).contains(&pixel.y)).then_some(
                        ScenePointObservation {
                            camera_index,
                            pixel,
                        },
                    )
                })
                .collect();
            (observations.len() >= 2).then_some(ImageTrack {
                feature_id: 9_007_199_254_740_999 + id as u64,
                observations,
            })
        })
        .collect();
    let graph = ImageTracks {
        observation_sha256: (0..5).map(|i| format!("{i:064x}")).collect(),
        tracks,
    };
    let seed = ReconstructionSeed {
        camera_indices: [0, 1],
        second_pose: poses[1].to_scene(),
    };
    (camera, graph, poses, seed)
}
#[test]
fn reconstructs_free_poses_and_retains_source_links() {
    let (camera, graph, poses, seed) = fixture();
    let before = graph.clone();
    let result = reconstruct(&camera, &graph, seed).expect("supported local scene");
    assert_eq!(result.source_camera_indices, vec![0, 1, 2, 3, 4]);
    assert!(result.unresolved_camera_indices.is_empty());
    assert!(result.refinement.final_cost <= result.refinement.initial_cost);
    for (found, expected) in result.scene.cameras.iter().zip(&poses) {
        assert!((found.pose.position - expected.center()).norm() < 1e-4);
        assert!(
            found
                .pose
                .orientation
                .angle_to(&expected.to_scene().orientation)
                < 1e-5
        );
    }
    for point in &result.scene.points {
        let input = before
            .tracks
            .iter()
            .find(|t| t.feature_id == point.feature_id)
            .expect("retained identity");
        for o in &point.observations {
            let source = result.source_camera_indices[o.camera_index];
            assert_eq!(pixel(input, source), Some(o.pixel));
        }
    }
    assert_eq!(graph.observation_sha256, before.observation_sha256);
    assert!(graph.tracks.iter().zip(before.tracks).all(|(a, b)| {
        a.observations
            .iter()
            .zip(b.observations)
            .all(|(x, y)| x.pixel == y.pixel)
    }));
}
#[test]
fn absent_scene_support_remains_an_explicit_gap() {
    let (camera, mut graph, _, seed) = fixture();
    graph.observation_sha256.push(format!("{:064x}", 19));
    let result = reconstruct(&camera, &graph, seed).expect("supported subset");
    assert_eq!(result.unresolved_camera_indices, vec![5]);
    assert!(!result.source_camera_indices.contains(&5));
}
#[test]
fn five_point_proposes_supported_seed_without_geographic_acceptance() {
    let (camera, graph, _, _) = fixture();
    let proposals = propose_seeds(&camera, &graph, [0, 2]).expect("two-view model");
    assert!(!proposals.is_empty());
    for p in proposals {
        assert_eq!(p.seed.camera_indices, [0, 2]);
        assert!(p.triangulated_points >= 20);
        assert!((p.seed.second_pose.position.norm() - 1.0).abs() < 1e-8);
    }
}
#[test]
fn repeated_image_evidence_and_feature_ids_are_rejected() {
    let (camera, mut graph, _, seed) = fixture();
    graph.observation_sha256[0] = "a".repeat(64);
    graph.observation_sha256[1] = "A".repeat(64);
    assert!(matches!(
        reconstruct(&camera, &graph, seed),
        Err(ReconstructionError::Observation { index: 1, .. })
    ));
    graph.observation_sha256[1] = "b".repeat(64);
    graph.tracks[1].feature_id = graph.tracks[0].feature_id;
    assert!(matches!(
        reconstruct(&camera, &graph, seed),
        Err(ReconstructionError::Track { .. })
    ));
}
#[test]
fn invalid_links_seed_and_resource_bounds_return_errors() {
    let (camera, graph, _, seed) = fixture();
    for pixel in [
        Vector2::new(f64::NAN, 0.0),
        Vector2::new(1280.0, 20.0),
        Vector2::new(-0.01, 20.0),
    ] {
        let mut input = graph.clone();
        input.tracks[0].observations[0].pixel = pixel;
        assert!(matches!(
            reconstruct(&camera, &input, seed),
            Err(ReconstructionError::Track { .. })
        ));
    }
    let mut input = graph.clone();
    let repeated = input.tracks[0].observations[0].clone();
    input.tracks[0].observations.push(repeated);
    assert!(matches!(
        reconstruct(&camera, &input, seed),
        Err(ReconstructionError::Track { .. })
    ));
    let mut bad = seed;
    bad.second_pose.position = Vector3::zeros();
    assert!(matches!(
        reconstruct(&camera, &graph, bad),
        Err(ReconstructionError::Seed { .. })
    ));
    bad = seed;
    bad.camera_indices = [0, 99];
    assert!(matches!(
        reconstruct(&camera, &graph, bad),
        Err(ReconstructionError::Seed { .. })
    ));
    input.observation_sha256 = (0..130).map(|i| format!("{i:064x}")).collect();
    assert!(matches!(
        reconstruct(&camera, &input, seed),
        Err(ReconstructionError::Limits { cameras: 130, .. })
    ));
}
#[test]
fn a_wrong_relative_pose_does_not_create_a_scene() {
    let (camera, graph, _, mut seed) = fixture();
    seed.second_pose = Pose {
        r: Matrix3::identity(),
        t: Vector3::new(1.0, 0.0, 0.0),
    }
    .to_scene();
    assert!(matches!(
        reconstruct(&camera, &graph, seed),
        Err(ReconstructionError::Support { .. })
    ));
}

#[test]
fn continuation_seed_preserves_relative_pose_in_any_scene_coordinates() {
    let (_, _, poses, _) = fixture();
    let world = UnitQuaternion::from_euler_angles(0.4, -0.2, 1.0);
    let shift = Vector3::new(30.0, 40.0, -20.0);
    let pair = [poses[1].to_scene(), poses[4].to_scene()].map(|p| crate::LocalScenePose {
        position: world * p.position * 7.0 + shift,
        orientation: world * p.orientation,
    });
    let seed = ReconstructionSeed::between([5, 8], pair).expect("shared estimates");
    let first = Pose::from_scene(pair[0]);
    let second = Pose::from_scene(pair[1]);
    let relative = Pose::from_scene(seed.second_pose);
    assert!((relative.r - second.r * first.r.transpose()).norm() < 1e-10);
    assert!((relative.t - (second.t - relative.r * first.t)).norm() < 1e-10);
    assert!(ReconstructionSeed::between([0, 1], [pair[0], pair[0]]).is_err());
}

#[test]
fn reconstructs_a_right_angle_camera_turn_without_an_attitude_prior() {
    let (camera, _, _, _) = fixture();
    let rotation =
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -std::f64::consts::FRAC_PI_2)
            .to_rotation_matrix()
            .into_inner();
    let poses = [
        Pose {
            r: Matrix3::identity(),
            t: Vector3::zeros(),
        },
        Pose {
            r: Matrix3::identity(),
            t: Vector3::new(-1.0, 0.0, 0.0),
        },
        Pose {
            r: rotation,
            t: -rotation * Vector3::new(-10.0, 0.0, 10.0),
        },
    ];
    let tracks = (0..240)
        .map(|id| {
            let world = Vector3::new(
                (id % 20) as f64 * 0.2 - 1.9,
                (id / 20) as f64 * 0.15 - 0.8,
                8.0 + (id % 13) as f64 * 0.3,
            );
            ImageTrack {
                feature_id: id,
                observations: poses
                    .iter()
                    .enumerate()
                    .map(|(camera_index, pose)| ScenePointObservation {
                        camera_index,
                        pixel: pose.project(&camera, world).expect("visible point"),
                    })
                    .collect(),
            }
        })
        .collect();
    let graph = ImageTracks {
        observation_sha256: (0..3).map(|i| format!("{i:064x}")).collect(),
        tracks,
    };
    let seed = ReconstructionSeed {
        camera_indices: [0, 1],
        second_pose: poses[1].to_scene(),
    };
    let result = reconstruct(&camera, &graph, seed).expect("nonplanar scene through a turn");
    assert_eq!(result.source_camera_indices, vec![0, 1, 2]);
    let found = result.scene.cameras[2].pose;
    assert!((found.position - poses[2].center()).norm() < 1e-3);
    assert!(found.orientation.angle_to(&poses[2].to_scene().orientation) < 1e-4);
}
