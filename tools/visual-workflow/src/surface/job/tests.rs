use super::*;
#[test]
fn query_truth_is_not_a_surface_solver_input() {
    let value = serde_json::json!({
        "query":{"path":"q.png","sha256":"a".repeat(64)},
        "camera":{"size":[64,64],"focal":[55.0,55.0],"principal":[31.5,31.5]},
        "prior":{"position":[0.0,0.0,100.0],"rotation":[0.0,0.0,0.0,1.0]},
        "position_radius_m":1000.0,"attitude_radius_rad":std::f64::consts::PI,
        "reference":{"image":{"path":"r.png","sha256":"a".repeat(64)},
        "heights":{"path":"h.bin","sha256":"b".repeat(64)},
        "origin":[0.0,0.0],"column":[1.0,0.0],"row":[0.0,-1.0],
        "map_release":"test","map_sha256":"c".repeat(64)}});
    assert!(serde_json::from_value::<SurfaceJob>(value.clone()).is_ok());
    let mut with_truth = value;
    with_truth["ground_truth_pose"] = serde_json::json!({"position":[0.0,0.0,100.0]});
    assert!(serde_json::from_value::<SurfaceJob>(with_truth).is_err());
}
