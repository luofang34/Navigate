//! Source identity and uncertainty controls at the WASM boundary.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
fn input() -> (String, String, String) {
    let camera = json!({"width":960,"height":544,"fx":700.0,"fy":690.0,"cx":479.5,"cy":271.5});
    let ids = ["ab".repeat(32), "cd".repeat(32)];
    let graph = json!({"observation_sha256":ids,"tracks":[{"feature_id":u64::MAX.to_string(),
        "observations":[{"camera_index":0,"pixel":[479.5,271.5]},{"camera_index":1,"pixel":[409.5,271.5]}]}]});
    let cameras: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            json!({
        "observation_sha256":id,"position_scene_units":[i as f64*0.3,0.0,0.0],
        "eye_to_scene_xyzw":[0.0,0.0,0.0,1.0],"fixed":i==0})
        })
        .collect();
    let scene = json!({"coordinate_gauge":{"origin_camera":0,"scale_camera":1},"cameras":cameras,"points":[]});
    (camera.to_string(), graph.to_string(), scene.to_string())
}
#[test]
fn preserves_large_ids_gauge_and_conditional_status() {
    let (camera, graph, source) = input();
    let output: serde_json::Value =
        serde_json::from_str(&run(&camera, &graph, &source).expect("valid input")).expect("JSON");
    let source: serde_json::Value = serde_json::from_str(&source).expect("source");
    assert_eq!(output["stage"], "conditional_scene_triangulation");
    assert_eq!(output["geographic_acceptance"], false);
    assert_eq!(output["scene"]["cameras"], source["cameras"]);
    assert_eq!(
        output["scene"]["coordinate_gauge"],
        source["coordinate_gauge"]
    );
    assert_eq!(
        output["scene"]["points"][0]["feature_id"],
        u64::MAX.to_string()
    );
    assert!(
        output["unresolved_feature_ids"]
            .as_array()
            .expect("array")
            .is_empty()
    );
    for field in ["uncertainty", "evidence_correlation"] {
        assert!(
            output[field]
                .as_str()
                .expect("unknown errors")
                .starts_with("unknown")
        );
    }
}
#[test]
fn rejects_changed_source_identity() {
    let (camera, graph, source) = input();
    let mut source: serde_json::Value = serde_json::from_str(&source).expect("source");
    source["cameras"][0]["observation_sha256"] = json!("ef".repeat(32));
    assert!(run(&camera, &graph, &source.to_string()).is_err());
}

#[test]
fn retains_supplied_camera_values_without_normalization_roundoff() {
    let (camera, graph, source) = input();
    let mut source: serde_json::Value = serde_json::from_str(&source).expect("source");
    source["cameras"][0]["eye_to_scene_xyzw"][3] = json!(1.0 + 1e-10);
    let output: serde_json::Value = serde_json::from_str(
        &run(&camera, &graph, &source.to_string()).expect("valid unit tolerance"),
    )
    .expect("JSON");
    assert_eq!(output["scene"]["cameras"], source["cameras"]);
    assert_eq!(
        output["scene"]["points"].as_array().expect("points").len(),
        1
    );
}
