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
        })
    };
    let first = fit(&mut observation, &reference, 1, "first.json").expect("first");
    assert_eq!(first["accepted"], true);
    assert_eq!(first["refinement_pose"]["inliers"], first["inliers"]);
    assert_eq!(
        first["refinement_pose"]["spatial_support"],
        first["spatial_support"]
    );
    assert!(
        first["refinement_pose"]
            .get("geometry_covariance")
            .is_none()
    );
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
        frame: navigate_visual::LocalFrame::anchor_mercator(40.0, -74.0).expect("valid anchor"),
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

struct MemoryMatcher {
    fail: bool,
}
impl ImageMatcher for MemoryMatcher {
    fn identity(&self) -> &str {
        "memory-test-adapter"
    }
    fn match_images_blocking(
        &mut self,
        _: &image::GrayImage,
        _: &image::GrayImage,
    ) -> Result<Vec<navigate_visual::PixelMatch>, navigate_visual::VisualError> {
        if self.fail {
            return Err(navigate_visual::VisualError::Invalid {
                field: "test backend failure",
            });
        }
        Ok((10..65)
            .step_by(10)
            .flat_map(|y| {
                (10..90).step_by(10).map(move |x| {
                    let pixel = nalgebra::Vector2::new(f64::from(x), f64::from(y));
                    navigate_visual::PixelMatch {
                        reference: pixel,
                        query: pixel,
                    }
                })
            })
            .collect())
    }
}
#[test]
fn in_memory_backend_preserves_alternatives_and_requires_rendered_depth() {
    let dir = tempfile::tempdir().expect("directory");
    let (mut observation, mut reference) = fixture(dir.path());
    let context = serde_json::json!({"map_release":"fixture"});
    let mut matcher = MemoryMatcher { fail: false };
    let (first, _) = observation
        .match_candidate_blocking(3, &reference, &mut matcher, &context)
        .expect("fit");
    assert_eq!(first["accepted"], true);
    let (repeated, _) = observation
        .match_candidate_blocking(3, &reference, &mut matcher, &context)
        .expect("repeat");
    assert_eq!(
        first["geometry_covariance"],
        repeated["geometry_covariance"]
    );
    reference.pose.position.x = 500.0;
    observation
        .match_candidate_blocking(4, &reference, &mut matcher, &context)
        .expect("alternative");
    assert_eq!(observation.select()["decision"], "unresolved");
    reference.depth_m.fill(0.0);
    let (unsupported, seed) = observation
        .match_candidate_blocking(4, &reference, &mut matcher, &context)
        .expect("missing surface report");
    assert_eq!(unsupported["accepted"], false);
    assert!(seed.is_none());
    assert_eq!(observation.select()["candidate_id"], 3);
    matcher.fail = true;
    assert!(
        observation
            .match_candidate_blocking(3, &reference, &mut matcher, &context)
            .is_err()
    );
    assert_eq!(observation.select()["decision"], "rejected");
}
