#![allow(clippy::expect_used, clippy::panic)]
use super::*;
fn camera() -> String {
    json!({"width":960,"height":544,"fx":700,"fy":690,"cx":479.5,"cy":271.5}).to_string()
}
fn scene() -> String {
    let cameras = vec![
        json!({"observation_sha256":"a".repeat(64),"position_scene_units":[0,0,0],"eye_to_scene_xyzw":[1,0,0,0],"fixed":true}),
        json!({"observation_sha256":"b".repeat(64),"position_scene_units":[1,0,0],"eye_to_scene_xyzw":[1,0,0,0],"fixed":true}),
    ];
    let point = json!({"feature_id":u64::MAX.to_string(),"position_scene_units":[0.01,0.01,4.03],"observations":[{"camera_index":0,"pixel":[479.5,271.5]},{"camera_index":1,"pixel":[304.5,271.5]}]});
    json!({"cameras":cameras,"points":[point]}).to_string()
}
#[test]
fn preserves_large_point_ids_and_returns_only_conditional_geometry() {
    let output = run(&camera(), &scene(), 40).expect("scene refinement");
    let result: serde_json::Value = serde_json::from_str(&output).expect("result JSON");
    assert_eq!(result["geographic_acceptance"], false);
    assert_eq!(
        result["scene"]["points"][0]["feature_id"],
        u64::MAX.to_string()
    );
    assert!(result["final_cost"].as_f64().expect("cost") < 1e-7);
    assert_eq!(
        result["scene"]["cameras"][0]["position_scene_units"],
        json!([0.0, 0.0, 0.0])
    );
}
#[test]
fn duplicate_observation_is_a_typed_error() {
    let mut value: serde_json::Value = serde_json::from_str(&scene()).expect("scene");
    value["cameras"][1]["observation_sha256"] = value["cameras"][0]["observation_sha256"].clone();
    assert!(matches!(
        run(&camera(), &value.to_string(), 10),
        Err(PreviewError::LocalScene(
            navigate_visual::LocalSceneError::Observation { index: 1, .. }
        ))
    ));
}

#[test]
fn preserves_fixed_pose_numbers_across_json_boundary() {
    let mut input: serde_json::Value = serde_json::from_str(&scene()).expect("scene");
    let rotation: [f64; 4] = [
        0.7104830756218631,
        -0.03404654428293617,
        0.045928282532831564,
        0.7013880701443906,
    ];
    input["cameras"][0]["eye_to_scene_xyzw"] = json!(rotation);
    input["cameras"][1]["eye_to_scene_xyzw"] = json!(rotation);
    let output = run(&camera(), &input.to_string(), 10).expect("refinement");
    let result: serde_json::Value = serde_json::from_str(&output).expect("result JSON");
    let value = result["scene"]["cameras"][0]["eye_to_scene_xyzw"][2]
        .as_f64()
        .expect("rotation component");
    assert_eq!(value.to_bits(), rotation[2].to_bits());
    let before: Scene = serde_json::from_value(input).expect("input scene");
    let after: Scene = serde_json::from_value(result["scene"].clone()).expect("output scene");
    for (a, b) in before.cameras.iter().zip(after.cameras) {
        assert_eq!(a.position_scene_units, b.position_scene_units);
        assert_eq!(a.eye_to_scene_xyzw, b.eye_to_scene_xyzw);
        assert_eq!(a.observation_sha256, b.observation_sha256);
        assert_eq!(a.fixed, b.fixed);
    }
}

#[test]
fn arbitrary_gauge_is_explicit_and_does_not_accept_a_location() {
    let mut input: serde_json::Value = serde_json::from_str(&scene()).expect("scene");
    input["cameras"][1]["fixed"] = json!(false);
    input["coordinate_gauge"] = json!({"origin_camera":0,"scale_camera":1});
    let output = run(&camera(), &input.to_string(), 10).expect("coordinate gauge");
    let result: serde_json::Value = serde_json::from_str(&output).expect("result");
    assert_eq!(result["geographic_acceptance"], false);
    assert_eq!(
        result["scene"]["coordinate_gauge"],
        input["coordinate_gauge"]
    );
    let before: Scene = serde_json::from_value(input.clone()).expect("input scene");
    let after: Scene = serde_json::from_value(result["scene"].clone()).expect("output scene");
    assert_eq!(
        before.cameras[0].position_scene_units,
        after.cameras[0].position_scene_units
    );
    assert_eq!(
        before.cameras[0].eye_to_scene_xyzw,
        after.cameras[0].eye_to_scene_xyzw
    );
    assert_eq!(
        before.cameras[0].observation_sha256,
        after.cameras[0].observation_sha256
    );
    assert_eq!(
        result["scene"]["points"][0]["feature_id"],
        u64::MAX.to_string()
    );
    assert!(
        result["uncertainty"]
            .as_str()
            .expect("uncertainty")
            .contains("no measured scale")
    );
    input["coordinate_gauge"]["scale_camera"] = json!(0);
    assert!(matches!(
        run(&camera(), &input.to_string(), 10),
        Err(PreviewError::LocalScene(
            navigate_visual::LocalSceneError::Gauge { .. }
        ))
    ));
}
