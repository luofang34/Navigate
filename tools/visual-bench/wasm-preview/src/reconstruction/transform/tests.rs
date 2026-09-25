//! Cloud restoration preserves source observations and the saved coordinate fit.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn stored_transform_moves_cloud_and_cameras_together() {
    let scene = json!({
        "cameras":[{"observation_sha256":"source", "position_scene_units":[1.0,2.0,3.0],
            "eye_to_scene_xyzw":[0.0,0.0,0.0,1.0], "fixed":false}],
        "points":[{"feature_id":"18446744073709551615", "position_scene_units":[4.0,5.0,6.0],
            "observations":[{"camera_index":0,"pixel":[10.5,12.25]}]}]
    });
    let alignment = json!({"scale":2.0,"source_to_target_xyzw":[0.0,0.0,1.0,0.0],
        "translation_target_scene_units":[10.0,20.0,30.0]});
    let restored: serde_json::Value = serde_json::from_str(
        &restore(&scene.to_string(), &alignment.to_string()).expect("restore"),
    )
    .expect("JSON");
    assert_eq!(
        restored["cameras"][0]["position_scene_units"],
        json!([8.0, 16.0, 36.0])
    );
    assert_eq!(
        restored["cameras"][0]["eye_to_scene_xyzw"],
        json!([0.0, 0.0, 1.0, 0.0])
    );
    assert_eq!(
        restored["points"][0]["position_scene_units"],
        json!([2.0, 10.0, 42.0])
    );
    assert_eq!(
        restored["points"][0]["observations"],
        scene["points"][0]["observations"]
    );
    assert_eq!(
        restored["points"][0]["feature_id"],
        scene["points"][0]["feature_id"]
    );
    assert_eq!(
        restored["cameras"][0]["observation_sha256"],
        scene["cameras"][0]["observation_sha256"]
    );
    for (field, value) in [
        ("scale", json!(0.0)),
        ("scale", json!(-2.0)),
        ("source_to_target_xyzw", json!([0.0, 0.0, 0.0, 0.5])),
    ] {
        let mut invalid = alignment.clone();
        invalid[field] = value;
        assert!(restore(&scene.to_string(), &invalid.to_string()).is_err());
    }
}
