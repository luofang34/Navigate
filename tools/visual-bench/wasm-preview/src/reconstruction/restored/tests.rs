//! Saved group reconstruction and overlap alignment through the worker boundary.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use nalgebra::{UnitQuaternion, Vector3};
use navigate_visual::ScenePointObservation;
use navigate_visual::reconstruction::ImageTrack;
#[test]
fn saved_graph_reconstructs_without_images_or_model_loading() {
    let camera = CameraModel {
        width: 320,
        height: 240,
        fx: 250.0,
        fy: 250.0,
        cx: 159.5,
        cy: 119.5,
    };
    let tracks = (0..300)
        .map(|id| {
            let world = Vector3::new(
                (id % 20) as f64 * 0.28 - 2.5,
                (id / 20) as f64 * 0.23 - 1.5,
                -10.0 - (id % 9) as f64 * 0.4,
            );
            ImageTrack {
                feature_id: u64::MAX - id,
                observations: (0..3)
                    .map(|camera_index| {
                        let rotation =
                            UnitQuaternion::from_euler_angles(0.0, camera_index as f64 * 0.02, 0.0);
                        let eye = rotation.inverse()
                            * (world - Vector3::new(camera_index as f64 * 0.5, 0.0, 0.0));
                        ScenePointObservation {
                            camera_index,
                            pixel: [
                                camera.fx * eye.x / (-eye.z) + camera.cx,
                                camera.cy - camera.fy * eye.y / (-eye.z),
                            ]
                            .into(),
                        }
                    })
                    .collect(),
            }
        })
        .collect();
    let graph = ImageTracks {
        observation_sha256: (0..3).map(|i| format!("{i:064x}")).collect(),
        tracks,
    };
    let stored = graph_json(&graph);
    let restored = graph::decode(&stored).expect("stored links");
    assert_eq!(restored.tracks[0].feature_id, u64::MAX);
    let proposals: serde_json::Value =
        serde_json::from_str(&proposals(&camera, &restored, [0, 2]).expect("seeds")).expect("JSON");
    let seed = proposals[0]["seed"].to_string();
    let output: serde_json::Value =
        serde_json::from_str(&solve(&camera, &restored, &seed).expect("scene")).expect("JSON");
    assert_eq!(
        output["scene"]["cameras"]
            .as_array()
            .expect("cameras")
            .len(),
        3
    );
    assert_eq!(output["geographic_acceptance"], false);
    assert!(
        output["unresolved_camera_indices"]
            .as_array()
            .expect("gaps")
            .is_empty()
    );
    assert!(
        output["scene"]["points"]
            .as_array()
            .expect("points")
            .iter()
            .all(|p| p["feature_id"].is_string())
    );
}
#[test]
fn overlap_alignment_is_conditional_and_preserves_exact_shared_identities() {
    let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.3);
    let source:Vec<_>=(0..8).map(|i|json!({"observation_sha256":format!("{i:064x}"),"position_scene_units":[i as f64,0.0,0.0],"eye_to_scene_xyzw":[0.0,0.0,0.0,1.0],"fixed":false})).collect();
    let target:Vec<_>=(0..8).map(|i|json!({"observation_sha256":format!("{i:064x}"),"position_scene_units":(rotation*Vector3::new(i as f64,0.0,0.0)*3.0+Vector3::new(12.0,-4.0,2.0)).as_slice(),"eye_to_scene_xyzw":rotation.coords.as_slice(),"fixed":false})).collect();
    let output = alignment(
        &json!({"cameras":source,"points":[]}).to_string(),
        &json!({"cameras":target,"points":[]}).to_string(),
    )
    .expect("shared poses");
    let result: serde_json::Value = serde_json::from_str(&output).expect("JSON");
    assert!((result["scale"].as_f64().expect("scale") - 3.0).abs() < 1e-10);
    assert_eq!(result["geographic_acceptance"], false);
    assert_eq!(
        result["observation_sha256"]
            .as_array()
            .expect("sources")
            .len(),
        8
    );
    assert!(
        result["uncertainty"]
            .as_str()
            .expect("uncertainty")
            .contains("unknown")
    );
}
