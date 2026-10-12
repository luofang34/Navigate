use super::*;
use nalgebra::{UnitQuaternion, Vector3};

fn scene() -> Scene {
    let cameras: Vec<_> = (0..3)
        .map(|i| SceneCamera {
            observation_sha256: format!("{i:064x}"),
            camera: Camera {
                size: [640, 480],
                focal: [500.0 + i as f64 * 20.0; 2],
                principal: [319.5, 239.5],
            },
            pose: Pose {
                position: [i as f64 * 2.0, 0.0, 10.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            fixed: i < 2,
        })
        .collect();
    let points = (0..40)
        .map(|i| {
            let world = Vector3::new(
                (i % 8) as f64 * 0.4 - 1.4,
                (i / 8) as f64 * 0.4 - 0.8,
                (i % 3) as f64 * 0.1,
            );
            let observations = cameras
                .iter()
                .enumerate()
                .map(|(camera_index, c)| Observation {
                    camera_index,
                    pixel: c
                        .camera
                        .model()
                        .project(
                            &CameraPose {
                                position: c.pose.position.into(),
                                orientation: UnitQuaternion::identity(),
                            },
                            world,
                        )
                        .expect("visible fixture point")
                        .into(),
                })
                .collect();
            ScenePoint {
                feature_id: i,
                position_prior: None,
                position: (world + Vector3::new(0.01, -0.02, 0.03)).into(),
                observations,
            }
        })
        .collect();
    let mut input = Scene {
        cameras,
        points,
        coordinate_gauge: None,
        iterations: 60,
    };
    input.cameras[2].pose.position = [4.04, -0.02, 10.03];
    input
}

#[test]
fn offline_scene_round_trip_refines_geometry_and_preserves_observations() {
    let root = tempfile::tempdir().expect("fixture directory");
    let input = root.path().join("scene.json");
    let output = root.path().join("refined.json");
    let source = scene();
    assets::write_json_blocking(&input, &source).expect("input");
    run_blocking(&input, &output).expect("native refinement");
    let report: serde_json::Value = assets::read_json_blocking(&output).expect("output");
    let fitted: Scene = serde_json::from_value(report["scene"].clone()).expect("scene wire format");
    let position = Vector3::from(fitted.cameras[2].pose.position);
    assert!((position - Vector3::new(4.0, 0.0, 10.0)).norm() < 1e-4);
    assert!(report["final_cost"].as_f64().expect("cost") < 1e-5);
    assert_eq!(
        report["source_sha256"],
        hex::encode(Sha256::digest(std::fs::read(input).expect("source bytes")))
    );
    for (a, b) in source.points.iter().zip(fitted.points) {
        assert_eq!(a.feature_id, b.feature_id);
        assert_eq!(
            serde_json::to_value(&a.observations).expect("source observations"),
            serde_json::to_value(b.observations).expect("retained observations")
        );
    }
}

#[test]
fn offline_map_constraints_keep_their_original_centers() {
    let root = tempfile::tempdir().expect("fixture directory");
    let input = root.path().join("scene.json");
    let output = root.path().join("fitted.json");
    let mut source = scene();
    for point in &mut source.points {
        point.position_prior = Some(PositionPrior {
            position: point.position,
            scale: 0.01,
        });
    }
    assets::write_json_blocking(&input, &source).expect("input");
    run_blocking(&input, &output).expect("constrained native refinement");
    let report: serde_json::Value = assets::read_json_blocking(&output).expect("output");
    let fitted: Scene = serde_json::from_value(report["scene"].clone()).expect("scene");
    for (a, b) in source.points.iter().zip(&fitted.points) {
        assert_eq!(
            serde_json::to_value(&a.position_prior).expect("prior"),
            serde_json::to_value(&b.position_prior).expect("retained prior")
        );
    }
    source.coordinate_gauge = Some([0, 1]);
    let invalid = root.path().join("invalid.json");
    let rejected = root.path().join("rejected.json");
    assets::write_json_blocking(&invalid, &source).expect("invalid input");
    assert!(run_blocking(&invalid, &rejected).is_err());
    assert!(!rejected.exists());
}
