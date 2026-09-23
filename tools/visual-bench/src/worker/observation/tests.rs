#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use nalgebra::{UnitQuaternion, Vector3};
use navigate_visual::{CameraPose, MapRevision};

#[test]
fn refinements_keep_alternatives_and_failed_evidence_invalidates_a_candidate() {
    let dir = tempfile::tempdir().expect("directory");
    let (mut observation, mut reference) = fixture(dir.path());
    let matches = dir.path().join("matches.json");
    let pairs: Vec<_> = (10..65)
        .step_by(10)
        .flat_map(|y| {
            (10..90)
                .step_by(10)
                .map(move |x| serde_json::json!({"reference":[x,y],"query":[x,y]}))
        })
        .collect();
    let mut payload = serde_json::json!({"backend_identity":"synthetic-coordinates",
        "reference_image_sha256":crate::package::digest(reference.image.as_raw()),
        "query_image_sha256":crate::package::digest(reference.image.as_raw()),
        "reference_depth_sha256":crate::matches::depth_digest(&reference.depth_m),"matches":pairs});
    std::fs::write(&matches, payload.to_string()).expect("pairs");
    let context = serde_json::json!({"map_release":"fixture"});
    let fit = |o: &mut Observation, r: &ReferenceView, id, name: &str| {
        o.refine_blocking(Refinement {
            id,
            reference: r,
            matches: &matches,
            output: &dir.path().join(name),
            map_context: &context,
            map_frame: MapFrame::new([40.0, -74.0]),
        })
    };
    let first = fit(&mut observation, &reference, 1, "first.json").expect("first");
    assert_eq!(first["accepted"], true);
    let repeated = fit(&mut observation, &reference, 1, "repeated.json").expect("repeat");
    assert_eq!(
        first["geometry_covariance"],
        repeated["geometry_covariance"]
    );
    assert_eq!(
        observation.select()["candidate_hypotheses"]
            .as_array()
            .expect("array")
            .len(),
        1
    );
    reference.pose.position.x = 500.0;
    fit(&mut observation, &reference, 2, "alternative.json").expect("alternative");
    let unresolved = observation.select();
    assert_eq!(unresolved["decision"], "unresolved");
    assert!(unresolved.get("position_enu_m").is_none());
    payload["query_image_sha256"] = "b".repeat(64).into();
    std::fs::write(&matches, payload.to_string()).expect("corrupt pairs");
    assert!(fit(&mut observation, &reference, 2, "bad.json").is_err());
    assert_eq!(observation.select()["candidate_id"], 1);
    observation.invalidate(1).expect("incomplete render");
    assert_eq!(observation.select()["decision"], "rejected");
}

fn fixture(dir: &Path) -> (Observation, ReferenceView) {
    let camera = CameraModel {
        width: 96,
        height: 72,
        fx: 90.0,
        fy: 90.0,
        cx: 47.5,
        cy: 35.5,
    };
    let pose = CameraPose {
        position: Vector3::new(0.0, 0.0, 100.0),
        orientation: UnitQuaternion::identity(),
    };
    let prior = PosePrior {
        pose,
        position_radius_m: 1000.0,
        attitude_radius_rad: 3.0,
    };
    let image = image::GrayImage::new(96, 72);
    let query = dir.join("query.png");
    image.save(&query).expect("query");
    let observation = Observation::new_blocking(
        &query,
        FrameStamp {
            sequence: 0,
            capture_time_ns: 1,
        },
        camera,
        prior,
    )
    .expect("observation");
    let reference = ReferenceView {
        pose,
        image,
        depth_m: vec![80.0; 96 * 72],
        map: MapRevision {
            release_id: "fixture".into(),
            manifest_sha256: "a".repeat(64),
        },
    };
    (observation, reference)
}
