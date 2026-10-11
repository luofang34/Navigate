use super::Request;

#[test]
fn dense_requests_keep_refinement_opt_in() {
    let ordinary: Request =
        serde_json::from_str(r#"{"reference":"reference.png","query":"query.png","dense":true}"#)
            .unwrap();
    assert!(!ordinary.refinement);
    let refinement: Request = serde_json::from_str(
        r#"{"reference":"reference.png","query":"query.png","dense":true,"refinement":true}"#,
    )
    .unwrap();
    assert!(refinement.dense && refinement.refinement);
}
