use super::*;

#[test]
fn arbitrary_camera_orientation_survives_and_invalid_quaternion_is_rejected() {
    let dir = tempfile::tempdir().expect("dir");
    let path = dir.path().join("orientation.json");
    let q = UnitQuaternion::from_euler_angles(0.4, 0.7, 1.2);
    std::fs::write(&path, serde_json::json!([q.coords.as_slice()]).to_string()).expect("write");
    let actual = orientations_blocking(Some(&path)).expect("orientation");
    assert!(q.angle_to(&actual[0]) < 1e-12);
    std::fs::write(&path, "[[0,0,0,2]]").expect("write");
    assert!(orientations_blocking(Some(&path)).is_err());
}
#[test]
fn map_catalog_and_descriptor_checksums_are_required_before_model_loading() {
    let dir = tempfile::tempdir().expect("dir");
    let index = dir.path().join("index.json");
    let catalog_path = dir.path().join("catalog.json");
    let descriptor_path = dir.path().join("descriptors.f32");
    let catalog=serde_json::json!({"release":"fixture","gallery":[{"id":3,"width_m":100.0,"center_enu_m":[0,0]}]}).to_string();
    let mut descriptor = vec![0_u8; 4096];
    descriptor[..4].copy_from_slice(&1_f32.to_le_bytes());
    std::fs::write(&catalog_path, &catalog).expect("catalog");
    std::fs::write(&descriptor_path, &descriptor).expect("descriptor");
    let revision = MapRevision {
        release_id: "fixture".into(),
        manifest_sha256: "a".repeat(64),
    };
    let mut value = serde_json::json!({"map_release":"fixture","map_manifest_sha256":revision.manifest_sha256,
        "catalog_manifest_sha256":digest(catalog.as_bytes()),"model_sha256":"b".repeat(64),"ids":[3],
        "descriptors":"descriptors.f32","descriptors_sha256":digest(&descriptor)});
    std::fs::write(&index, value.to_string()).expect("index");
    assert!(load_blocking(&index, &catalog_path, &revision).is_ok());
    value["map_release"] = "different".into();
    std::fs::write(&index, value.to_string()).expect("index");
    assert!(load_blocking(&index, &catalog_path, &revision).is_err());
    value["map_release"] = "fixture".into();
    value["ids"] = serde_json::json!([7]);
    std::fs::write(&index, value.to_string()).expect("index");
    assert!(load_blocking(&index, &catalog_path, &revision).is_err());
    value["ids"] = serde_json::json!([3]);
    std::fs::write(&index, value.to_string()).expect("index");
    descriptor[4] = 1;
    std::fs::write(&descriptor_path, &descriptor).expect("corruption");
    assert!(load_blocking(&index, &catalog_path, &revision).is_err());
}
