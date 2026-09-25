//! Conditional triangulation controls for general camera orientations.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use crate::reconstruction::ImageTrack;
use crate::{LocalScenePose, ScenePointObservation};
use nalgebra::{UnitQuaternion, Vector3};

pub(crate) fn fixture() -> (CameraModel, Vec<LocalSceneCamera>, Vec<Vector3<f64>>) {
    let camera = CameraModel {
        width: 960,
        height: 544,
        fx: 700.0,
        fy: 690.0,
        cx: 479.5,
        cy: 271.5,
    };
    let origin = Vector3::new(17.0, -8.0, 6.0);
    let rotation = UnitQuaternion::from_euler_angles(1.2, -0.4, 0.7);
    let cameras = (0..4)
        .map(|i| LocalSceneCamera {
            observation_sha256: format!("{i:064x}"),
            pose: LocalScenePose {
                position: origin + rotation * Vector3::new(i as f64 * 0.2, 0.0, 0.0),
                orientation: rotation
                    * UnitQuaternion::from_euler_angles(0.0, i as f64 * 0.005, 0.0),
            },
            fixed: i == 0,
        })
        .collect();
    let points = (0..60)
        .map(|i| {
            origin
                + rotation
                    * Vector3::new(
                        (i % 10) as f64 * 0.1 - 0.45,
                        (i / 10) as f64 * 0.1 - 0.25,
                        if i < 30 {
                            -4.0
                        } else {
                            -3.0 - (i % 7) as f64 * 0.2
                        },
                    )
        })
        .collect();
    (camera, cameras, points)
}
pub(crate) fn graph(
    camera: &CameraModel,
    cameras: &[LocalSceneCamera],
    points: &[Vector3<f64>],
) -> ImageTracks {
    ImageTracks {
        observation_sha256: cameras
            .iter()
            .map(|c| c.observation_sha256.clone())
            .collect(),
        tracks: points
            .iter()
            .enumerate()
            .map(|(i, &p)| ImageTrack {
                feature_id: u64::MAX - i as u64,
                observations: cameras
                    .iter()
                    .enumerate()
                    .map(|(camera_index, c)| ScenePointObservation {
                        camera_index,
                        pixel: Pose::from_scene(c.pose)
                            .project(camera, p)
                            .expect("visible point"),
                    })
                    .collect(),
            })
            .collect(),
    }
}
#[test]
fn recovers_wall_and_nonplanar_points_without_changing_cameras() {
    let (camera, cameras, expected) = fixture();
    let graph = graph(&camera, &cameras, &expected);
    let result = triangulate_scene_tracks(&camera, &graph, &cameras).expect("valid scene");
    assert!(result.unresolved_feature_ids.is_empty());
    assert_eq!(result.scene.points.len(), expected.len());
    for (i, (point, expected)) in result.scene.points.iter().zip(expected).enumerate() {
        assert!((point.position - expected).norm() < 1e-10);
        assert_eq!(point.feature_id, u64::MAX - i as u64);
        assert_eq!(point.observations.len(), 4);
    }
    for (before, after) in cameras.iter().zip(&result.scene.cameras) {
        assert_eq!(before.pose.position, after.pose.position);
        assert_eq!(before.pose.orientation, after.pose.orientation);
        assert_eq!(before.fixed, after.fixed);
        assert_eq!(before.observation_sha256, after.observation_sha256);
    }
    let repeated = triangulate_scene_tracks(&camera, &graph, &cameras).expect("repeat");
    for (a, b) in result.scene.points.iter().zip(repeated.scene.points) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.observations.len(), b.observations.len());
    }
}
#[test]
fn insufficient_parallax_stays_unresolved() {
    let (camera, mut cameras, points) = fixture();
    let origin = cameras[0].pose.position;
    for c in &mut cameras {
        c.pose.position = origin;
    }
    for offset in [0.0, 1e-7] {
        cameras[1].pose.position.x += offset;
        let graph = graph(&camera, &cameras, &points);
        let result = triangulate_scene_tracks(&camera, &graph, &cameras).expect("valid weak scene");
        assert!(result.scene.points.is_empty());
        assert_eq!(
            result.unresolved_feature_ids,
            graph
                .tracks
                .iter()
                .map(|t| t.feature_id)
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn inconsistent_pixels_do_not_become_supported_scene_points() {
    let (camera, cameras, points) = fixture();
    let mut graph = graph(&camera, &cameras, &points);
    for track in &mut graph.tracks {
        track.observations[0].pixel.y += 35.0;
        track.observations[1].pixel.y -= 35.0;
    }
    let result = triangulate_scene_tracks(&camera, &graph, &cameras).expect("valid noisy graph");
    assert!(result.scene.points.is_empty());
    assert_eq!(result.unresolved_feature_ids.len(), points.len());
}
#[test]
fn rejects_source_mismatch_invalid_pose_and_repeated_evidence() {
    let (camera, mut cameras, points) = fixture();
    let mut graph = graph(&camera, &cameras, &points);
    assert!(matches!(
        triangulate_scene_tracks(&camera, &graph, &cameras[..3]),
        Err(ReconstructionError::CameraCount { .. })
    ));
    cameras.swap(0, 1);
    assert!(matches!(
        triangulate_scene_tracks(&camera, &graph, &cameras),
        Err(ReconstructionError::Observation { index: 0, .. })
    ));
    cameras.swap(0, 1);
    cameras[1].pose.position.x = f64::NAN;
    let error = triangulate_scene_tracks(&camera, &graph, &cameras).expect_err("invalid pose");
    assert!(matches!(
        error,
        ReconstructionError::SceneCamera { index: 1, .. }
    ));
    assert!(std::error::Error::source(&error).is_some());
    let repeated = graph.tracks[0].observations[0].clone();
    graph.tracks[0].observations.push(repeated);
    assert!(matches!(
        triangulate_scene_tracks(&camera, &graph, &cameras),
        Err(ReconstructionError::Track { .. })
    ));
}
