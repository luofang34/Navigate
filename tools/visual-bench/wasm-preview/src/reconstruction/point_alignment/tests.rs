//! Worker serialization retains point evidence without geographic acceptance.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
#[test]
fn worker_returns_conditional_camera_alternatives_and_point_ids() {
    let cameras:Vec<_>=(0..6).map(|i|json!({"observation_sha256":format!("{i:064x}"),"position_scene_units":[0.0,0.0,0.0],"eye_to_scene_xyzw":[0.0,0.0,0.0,1.0],"fixed":false})).collect();
    let points:Vec<_>=(0..12).map(|i|json!({"feature_id":i.to_string(),"position_scene_units":[i as f64*0.3-1.5,(i%3) as f64*0.4,-8.0],"observations":(0..6).map(|camera_index|json!({"camera_index":camera_index,"pixel":[i as f64*4.0+20.0,50.0]})).collect::<Vec<_>>()})).collect();
    let source = json!({"cameras":cameras,"points":points});
    let mut target = source.clone();
    for point in target["points"].as_array_mut().expect("points") {
        for v in point["position_scene_units"]
            .as_array_mut()
            .expect("coordinates")
        {
            *v = json!(v.as_f64().expect("number") * 4.0)
        }
        point["feature_id"] = json!(format!("1{}", point["feature_id"].as_str().expect("ID")));
    }
    let report: serde_json::Value = serde_json::from_str(
        &align(&source.to_string(), &target.to_string()).expect("aligned scene"),
    )
    .expect("result");
    assert_eq!(report["geographic_acceptance"], false);
    assert_eq!(report["candidate_budget_exhausted"], false);
    let candidates = report["candidates"].as_array().expect("alternatives");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0]["scale"], 4.0);
    assert_eq!(candidates[0]["geographic_acceptance"], false);
    assert_eq!(candidates[0]["scene"]["points"], json!([]));
    assert_eq!(candidates[0]["scene"]["cameras"], source["cameras"]);
    assert_eq!(
        candidates[0]["point_associations"]
            .as_array()
            .expect("point links")
            .len(),
        12
    );
    assert!(
        candidates[0]["uncertainty"]
            .as_str()
            .expect("uncertainty")
            .contains("unknown")
    );
    assert!(
        candidates[0]["evidence_correlation"]
            .as_str()
            .expect("correlation")
            .contains("reuse")
    );
}
