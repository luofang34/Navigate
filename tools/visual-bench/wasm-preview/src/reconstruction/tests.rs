//! Worker conversion checks retain evidence and reject invalid input.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
const CAMERA: &str = r#"{"width":320,"height":240,"fx":250,"fy":250,"cx":159.5,"cy":119.5}"#;
#[test]
fn scene_group_has_no_implicit_geographic_acceptance() {
    let mut session = SceneTracks::create(CAMERA).expect("camera");
    session.append("a".repeat(64), "[]").expect("first");
    session.append("b".repeat(64), "[]").expect("second");
    let report: serde_json::Value = serde_json::from_str(&session.snapshot()).expect("snapshot");
    assert_eq!(report["observation_sha256"][0], "a".repeat(64));
    assert_eq!(report["geographic_acceptance"], false);
    assert!(session.proposals([0, 1]).is_err());
    assert!(session.append("A".repeat(64), "[]").is_err());
}
#[test]
fn u64_feature_identity_is_lossless_in_javascript() {
    let graph = ImageTracks {
        observation_sha256: vec!["a".repeat(64)],
        tracks: vec![navigate_visual::reconstruction::ImageTrack {
            feature_id: u64::MAX,
            observations: vec![],
        }],
    };
    let json: serde_json::Value = serde_json::from_str(&graph_json(&graph)).expect("json");
    assert_eq!(json["tracks"][0]["feature_id"], u64::MAX.to_string());
}
#[test]
fn invalid_seed_rotation_is_not_silently_normalized() {
    let seed = Seed {
        camera_indices: [0, 1],
        position_scene_units: [1.0, 0.0, 0.0],
        eye_to_scene_xyzw: [0.0, 0.0, 0.0, 2.0],
    };
    assert!(seed.model().is_err());
}

#[test]
fn lost_two_frame_links_release_capacity_through_the_worker_boundary() {
    let mut scene = SceneTracks::create(CAMERA).expect("camera");
    scene.append(format!("{:064x}", 0), "[]").expect("first");
    let links: Vec<_> = (0..12)
        .flat_map(|y| {
            (0..12).map(move |x| json!({"reference":[30+x*12,30+y*12],"query":[32+x*12,30+y*12]}))
        })
        .collect();
    let links = serde_json::to_string(&links).expect("pairs");
    for i in 1..120 {
        scene
            .append(format!("{i:064x}"), if i % 2 == 0 { "[]" } else { &links })
            .expect("group capacity");
        if i % 2 == 0 {
            let graph: serde_json::Value = serde_json::from_str(&scene.snapshot()).expect("graph");
            assert!(graph["tracks"].as_array().expect("tracks").is_empty());
        }
    }
    let graph: serde_json::Value = serde_json::from_str(&scene.snapshot()).expect("graph");
    assert_eq!(
        graph["observation_sha256"]
            .as_array()
            .expect("cameras")
            .len(),
        120
    );
    assert!(
        graph["tracks"][0]["feature_id"]
            .as_str()
            .expect("identity")
            .parse::<u64>()
            .expect("id")
            > 5000
    );
}
