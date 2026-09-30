use super::*;
fn row(t: u64, anchor: &str, supported: bool) -> Value {
    json!({"capture_time_ns":t,"observation_sha256":format!("frame-{t}"),"context":{"map_manifest_sha256":"map"},
        "candidate_hypotheses":[{"candidate_id":0,"accepted":false,"tracking_supported":supported,
            "anchor_observation_sha256":anchor,"longitude_deg":-74.0,"latitude_deg":40.0}]})
}
#[test]
fn gaps_and_reacquired_anchors_split_conditional_paths() {
    let output = collection(&[
        row(0, "a", true),
        row(1, "a", true),
        row(2, "a", false),
        row(3, "a", true),
        row(4, "b", true),
        row(5, "b", true),
    ])
    .expect("valid track");
    let lines = output["features"].as_array().map(|values| {
        values
            .iter()
            .filter(|f| f["geometry"]["type"] == "LineString")
            .count()
    });
    assert_eq!(lines, Some(2));
    assert_eq!(output["gaps"].as_array().map(Vec::len), Some(1));
}
#[test]
fn diagnostic_map_check_does_not_become_a_path_vertex() {
    let first = row(0, "a", true);
    let mut second = row(1, "a", true);
    second["candidate_hypotheses"][0]["map_check"] =
        json!({"accepted":true,"latitude_deg":50.0,"longitude_deg":-80.0});
    let output = collection(&[first, second]).expect("valid track");
    for feature in output["features"].as_array().into_iter().flatten() {
        if feature["geometry"]["type"] == "LineString" {
            assert_eq!(
                feature["geometry"]["coordinates"],
                json!([[-74.0, 40.0], [-74.0, 40.0]])
            );
        }
    }
}

#[test]
fn supported_pose_with_missing_coordinates_cannot_be_exported() {
    let mut invalid = row(0, "a", true);
    invalid["candidate_hypotheses"][0]["longitude_deg"] = Value::Null;
    assert!(collection(&[invalid.clone()]).is_err());
    invalid["candidate_hypotheses"][0]["longitude_deg"] = 181.0.into();
    assert!(collection(&[invalid]).is_err());
}
