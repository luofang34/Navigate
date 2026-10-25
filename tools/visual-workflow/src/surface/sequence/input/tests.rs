use super::*;
#[test]
fn sequence_rejects_truth_fields_and_reordered_evidence() {
    let value = serde_json::json!({"job":"job.json","refresh_interval":30,"frames":[
        {"image":{"path":"0.png","sha256":"a".repeat(64)},"frame_id":0,"capture_time_ns":0},
        {"image":{"path":"1.png","sha256":"b".repeat(64)},"frame_id":1,"capture_time_ns":33333333}]});
    let spec: Manifest = serde_json::from_value(value.clone()).unwrap();
    assert!(spec.validate().is_ok());
    let mut truth = value.clone();
    truth["frames"][0]["true_pose"] = serde_json::json!([1, 2, 3]);
    assert!(serde_json::from_value::<Manifest>(truth).is_err());
    for (key, invalid) in [("frame_id", 0), ("capture_time_ns", 0)] {
        let mut v = value.clone();
        v["frames"][1][key] = serde_json::json!(invalid);
        assert!(
            serde_json::from_value::<Manifest>(v)
                .unwrap()
                .validate()
                .is_err()
        );
    }
}
