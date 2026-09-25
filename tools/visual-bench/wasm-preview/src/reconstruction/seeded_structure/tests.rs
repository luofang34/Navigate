//! Traced seed initialization retains source identities at the WASM boundary.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
fn input() -> (String, String, String, String) {
    let camera = json!({"width":960,"height":544,"fx":700.0,"fy":690.0,"cx":479.5,"cy":271.5});
    let ids = ["ab".repeat(32), "cd".repeat(32)];
    let graph = json!({"observation_sha256":ids,"tracks":[{"feature_id":u64::MAX.to_string(),
        "observations":[{"camera_index":0,"pixel":[479.5,281.5]},{"camera_index":1,"pixel":[409.5,261.5]}]}]});
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
    let seeds = json!([{"feature_id":u64::MAX.to_string(),"position_scene_units":[0.0,0.0,-3.0]}]);
    (
        camera.to_string(),
        graph.to_string(),
        scene.to_string(),
        seeds.to_string(),
    )
}
#[test]
fn initializes_traced_points_without_accepting_geometry_or_changing_cameras() {
    let (camera, graph, scene, seeds) = input();
    let result: serde_json::Value =
        serde_json::from_str(&run(&camera, &graph, &scene, &seeds).expect("valid seeds"))
            .expect("result");
    let original: serde_json::Value = serde_json::from_str(&scene).expect("source");
    assert_eq!(result["stage"], "conditional_scene_initialization");
    assert_eq!(result["geographic_acceptance"], false);
    assert_eq!(result["scene"]["cameras"], original["cameras"]);
    assert_eq!(
        result["scene"]["coordinate_gauge"],
        original["coordinate_gauge"]
    );
    assert_eq!(
        result["scene"]["points"][0]["feature_id"],
        u64::MAX.to_string()
    );
    assert_eq!(
        result["scene"]["points"][0]["observations"]
            .as_array()
            .expect("links")
            .len(),
        2
    );
    assert_eq!(
        result["scene"]["points"][0]["position_scene_units"],
        json!([0.0, 0.0, -3.0])
    );
    assert!(
        result["uncertainty"]
            .as_str()
            .expect("uncertainty")
            .starts_with("unknown")
    );
}
#[test]
fn rejects_invalid_seed_ids_and_changed_observation_identity() {
    let (camera, graph, scene, seeds) = input();
    for invalid in ["invalid", "7"] {
        let mut value: serde_json::Value = serde_json::from_str(&seeds).expect("source");
        value[0]["feature_id"] = json!(invalid);
        assert!(run(&camera, &graph, &scene, &value.to_string()).is_err());
    }
    let mut scene: serde_json::Value = serde_json::from_str(&scene).expect("scene");
    scene["cameras"][0]["observation_sha256"] = json!("ef".repeat(32));
    assert!(run(&camera, &graph, &scene.to_string(), &seeds).is_err());
}
