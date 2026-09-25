//! Worker graph identity, repeated fitting, and cache invalidation controls.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
fn fixture() -> (String, String, String) {
    let camera = json!({"width":960,"height":544,"fx":700.0,"fy":690.0,"cx":479.5,"cy":271.5});
    let ids = ["ab".repeat(32), "cd".repeat(32)];
    let mut tracks = Vec::new();
    let mut points = Vec::new();
    for i in 0..60_u64 {
        let x = (i % 10) as f64 * 0.1 - 0.45;
        let y = (i / 10) as f64 * 0.1 - 0.25;
        let depth = 3.0 + (i % 7) as f64 * 0.2;
        let observations = vec![
            json!({"camera_index":0,"pixel":[700.0*x/depth+479.5,271.5-690.0*y/depth]}),
            json!({"camera_index":1,"pixel":[700.0*(x-0.3)/depth+479.5,271.5-690.0*y/depth]}),
        ];
        tracks.push(json!({"feature_id":(u64::MAX-i).to_string(),"observations":observations}));
        points.push(json!({"feature_id":i.to_string(),"position_scene_units":[x,y,-depth],"observations":observations}));
    }
    let cameras=ids.iter().enumerate().map(|(i,id)|json!({"observation_sha256":id,"position_scene_units":[i as f64*0.3,0.0,0.0],"eye_to_scene_xyzw":[0.0,0.0,0.0,1.0],"fixed":true})).collect::<Vec<_>>();
    (
        camera.to_string(),
        json!({"observation_sha256":ids,"tracks":tracks}).to_string(),
        json!({"cameras":cameras,"points":points}).to_string(),
    )
}
fn initial() -> String {
    json!({"position_scene_units":[0.02,0.01,0.0],"eye_to_scene_xyzw":[0.0,0.0,0.0,1.0]})
        .to_string()
}
#[test]
fn joined_selection_preserves_large_source_ids_and_does_not_accept_a_pose() {
    let (camera, graph, _) = fixture();
    let mut state = SceneTrackGraph::create(&camera).expect("camera");
    state.append(&"ef".repeat(32), &graph).expect("source");
    let selected: serde_json::Value = serde_json::from_str(
        &state
            .selection(
                &json!(["ab".repeat(32), "cd".repeat(32)]).to_string(),
                60,
                1,
            )
            .expect("selection"),
    )
    .expect("JSON");
    assert_eq!(selected["geographic_acceptance"], false);
    assert_eq!(
        selected["graph"]["tracks"]
            .as_array()
            .expect("tracks")
            .len(),
        60
    );
    assert_eq!(
        selected["association_sources"][0]["source_tracks"][0]["feature_id"],
        u64::MAX.to_string()
    );
    assert!(state.append(&"EF".repeat(32), &graph).is_err());
}
#[test]
fn repeated_fits_keep_exact_scene_identity_and_do_not_add_support() {
    let (camera, graph, scene) = fixture();
    let mut state = SceneTrackGraph::create(&camera).expect("camera");
    state.append(&"ef".repeat(32), &graph).expect("source");
    let sha = state.set_points(&scene).expect("scene");
    assert_eq!(sha, format!("{:x}", Sha256::digest(scene.as_bytes())));
    let first = state.fit(&"ab".repeat(32), &initial()).expect("camera fit");
    assert_eq!(state.set_points(&scene).expect("same scene"), sha);
    assert_eq!(
        state.fit(&"ab".repeat(32), &initial()).expect("same fit"),
        first
    );
    let result: serde_json::Value = serde_json::from_str(&first).expect("JSON");
    assert_eq!(result["scene_sha256"], sha);
    assert_eq!(result["geographic_acceptance"], false);
    assert_eq!(
        result["fit"]["inlier_feature_ids"]
            .as_array()
            .expect("fit")
            .len(),
        60
    );
    assert_eq!(result["match_count"], 60);
    assert!(result["fit"]["reprojection_rms_px"].as_f64().expect("RMS") < 1e-8);
}
#[test]
fn changed_graph_cannot_reuse_selected_points() {
    let (camera, graph, scene) = fixture();
    let mut state = SceneTrackGraph::create(&camera).expect("camera");
    state.append(&"ef".repeat(32), &graph).expect("source");
    state.set_points(&scene).expect("scene");
    assert!(state.append(&"ef".repeat(32), &graph).is_err());
    assert!(
        state.fit(&"ab".repeat(32), &initial()).is_ok(),
        "failed append leaves the graph and point selection unchanged"
    );
    state
        .append(&"01".repeat(32), &graph)
        .expect("new source group");
    assert!(matches!(
        state.fit(&"ab".repeat(32), &initial()),
        Err(PreviewError::Input { .. })
    ));
}
#[test]
fn invalid_scene_does_not_replace_selected_points() {
    let (camera, graph, scene) = fixture();
    let mut state = SceneTrackGraph::create(&camera).expect("camera");
    state.append(&"ef".repeat(32), &graph).expect("source");
    state.set_points(&scene).expect("scene");
    let before = state.fit(&"ab".repeat(32), &initial()).expect("fit");
    let mut invalid: serde_json::Value = serde_json::from_str(&scene).expect("scene JSON");
    invalid["points"][0]["feature_id"] = json!("99999");
    assert!(matches!(
        state.set_points(&invalid.to_string()),
        Err(PreviewError::TrackMerge(_))
    ));
    assert_eq!(
        state
            .fit(&"ab".repeat(32), &initial())
            .expect("retained fit"),
        before
    );
    assert!(state.fit("unknown", &initial()).is_err());
}
