//! JSON boundary evidence and failure controls.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;

fn camera() -> String {
    json!({"width":960,"height":544,"fx":700.0,"fy":690.0,"cx":479.5,"cy":271.5}).to_string()
}
fn request() -> serde_json::Value {
    json!({"scene_sha256":"ab".repeat(32),"observation_sha256":"cd".repeat(32),
        "position_scene_units":[0.0,0.0,0.0],"eye_to_scene_xyzw":[0.0,0.0,0.0,1.0],"matches":[]})
}
#[test]
fn unresolved_fit_keeps_identities_and_unknown_errors() {
    let request = request();
    let output: serde_json::Value =
        serde_json::from_str(&run(&camera(), &request.to_string()).expect("valid input"))
            .expect("JSON output");
    assert_eq!(output["stage"], "conditional_scene_resection");
    assert_eq!(output["geographic_acceptance"], false);
    assert_eq!(output["scene_sha256"], request["scene_sha256"]);
    assert_eq!(output["observation_sha256"], request["observation_sha256"]);
    assert!(output["fit"].is_null());
    assert!(
        output["uncertainty"]
            .as_str()
            .expect("uncertainty")
            .starts_with("unknown")
    );
    assert!(
        output["evidence_correlation"]
            .as_str()
            .expect("correlation")
            .starts_with("unknown")
    );
}
#[test]
fn invalid_feature_identity_retains_parse_context() {
    let mut input = request();
    input["matches"] = json!([{"feature_id":"bad-point-id","position_scene_units":[0.0,0.0,-3.0],"pixel":[479.5,271.5]}]);
    let error = run(&camera(), &input.to_string()).expect_err("invalid point ID");
    assert!(matches!(&error, PreviewError::ScenePointId { id, .. } if id == "bad-point-id"));
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn supported_fit_preserves_feature_ids_beyond_javascript_integer_precision() {
    let mut input = request();
    let matches: Vec<_> = (0..60_u64)
        .map(|i| {
            let x = (i % 10) as f64 * 0.1 - 0.45;
            let y = (i / 10) as f64 * 0.1 - 0.25;
            let depth = 3.0 + (i % 7) as f64 * 0.2;
            json!({"feature_id":(u64::MAX-i).to_string(),"position_scene_units":[x,y,-depth],
            "pixel":[700.0*x/depth+479.5,271.5-690.0*y/depth]})
        })
        .collect();
    input["matches"] = json!(matches);
    let output: serde_json::Value =
        serde_json::from_str(&run(&camera(), &input.to_string()).expect("valid point links"))
            .expect("fit JSON");
    let fit = &output["fit"];
    assert!(!fit.is_null());
    assert_eq!(output["geographic_acceptance"], false);
    assert_eq!(
        fit["inlier_feature_ids"],
        json!(
            (0..60_u64)
                .map(|i| (u64::MAX - i).to_string())
                .collect::<Vec<_>>()
        )
    );
    assert!(fit["reprojection_rms_px"].as_f64().expect("image error") < 1e-10);
}
