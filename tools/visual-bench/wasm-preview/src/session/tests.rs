#![allow(clippy::expect_used)]
use super::*;
use navigate_visual::MapRevision;
fn scene() -> (Session, ReferenceView, Vec<PixelMatch>) {
    let camera = Camera {
        width: 160,
        height: 120,
        fx: 150.0,
        fy: 150.0,
        cx: 79.5,
        cy: 59.5,
    };
    let pose = Pose {
        position_enu_m: [0.0, 0.0, 150.0],
        eye_to_enu_xyzw: [0.0, 0.0, 0.0, 1.0],
    };
    let prior = Prior {
        pose,
        position_radius_m: 50.0,
        attitude_radius_rad: 0.5,
    };
    let session =
        Session::new(camera, vec![100; 160 * 120], prior, 0, 0.0).expect("fixture observation");
    let reference = ReferenceView {
        frame: navigate_visual::LocalFrame::anchor_mercator(40.0, -74.0).expect("valid anchor"),
        map: MapRevision {
            release_id: "surface-test".into(),
            manifest_sha256: "a".repeat(64),
        },
        pose: pose.model().expect("fixture pose"),
        image: GrayImage::from_pixel(160, 120, image::Luma([100])),
        depth_m: (0..160 * 120)
            .map(|i| 100.0 + (i / 160 % 23) as f32)
            .collect(),
    };
    let pairs = (1..7)
        .flat_map(|x| {
            (1..6).map(move |y| {
                let p = Vector2::new(f64::from(x * 20), f64::from(y * 18));
                PixelMatch {
                    reference: p,
                    query: p,
                }
            })
        })
        .collect();
    (session, reference, pairs)
}
#[test]
fn repeat_refinement_replaces_shared_evidence_without_gaining_precision() {
    let (mut session, reference, pairs) = scene();
    session.invalidate(1).expect("candidate");
    let first = session
        .refine(1, &reference, &pairs, "custom-classical-test")
        .expect("fit");
    assert_eq!(first["accepted"], true);
    let second = session
        .refine(1, &reference, &pairs, "custom-classical-test")
        .expect("repeat fit");
    assert_eq!(first["geometry_covariance"], second["geometry_covariance"]);
    assert_eq!(
        session.select()["candidate_hypotheses"]
            .as_array()
            .expect("results")
            .len(),
        1
    );
    session.invalidate(2).expect("alternative");
    session
        .refine(2, &reference, &pairs, "another-backend")
        .expect("alternative fit");
    assert_eq!(session.select()["decision"], "unresolved");
    assert!(session.select().get("position_enu_m").is_none());
}
#[test]
fn missing_depth_and_interrupted_replacements_cannot_reuse_an_accepted_pose() {
    let (mut session, mut reference, pairs) = scene();
    session.invalidate(7).expect("candidate");
    assert_eq!(
        session
            .refine(7, &reference, &pairs, "matcher")
            .expect("fit")["accepted"],
        true
    );
    session.invalidate(7).expect("replace");
    assert_eq!(session.select()["decision"], "rejected");
    reference.depth_m.fill(0.0);
    assert_eq!(
        session
            .refine(7, &reference, &pairs, "matcher")
            .expect("rejected report")["accepted"],
        false
    );
    assert_eq!(session.select()["decision"], "rejected");
}

#[test]
fn a_refinement_seed_remains_rejected_until_new_geometry_passes() {
    let (mut session, reference, full_pairs) = scene();
    let restricted = (5..115)
        .step_by(10)
        .flat_map(|x| {
            (5..35).step_by(10).map(move |y| {
                let p = Vector2::new(f64::from(x), f64::from(y));
                PixelMatch {
                    reference: p,
                    query: p,
                }
            })
        })
        .collect::<Vec<_>>();
    session.invalidate(0).expect("candidate");
    let weak = session
        .refine(0, &reference, &restricted, "matcher")
        .expect("report");
    assert_eq!(weak["accepted"], false);
    assert_eq!(weak["refinement_proposal"]["accepted"], false);
    assert_eq!(session.select()["decision"], "rejected");
    assert!(session.select().get("position_enu_m").is_none());
    assert!(weak.get("geometry_covariance").is_none());
    let complete = session
        .refine(0, &reference, &full_pairs, "matcher")
        .expect("report");
    assert_eq!(complete["accepted"], true);
    assert!(complete.get("refinement_proposal").is_none());
}

#[test]
fn relative_tracking_never_becomes_an_accepted_map_candidate() {
    let (mut session, reference, pairs) = scene();
    let previous = Frame {
        camera: session.frame.camera,
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
        image: GrayImage::from_pixel(160, 120, image::Luma([99])),
    };
    session.invalidate(0).expect("candidate");
    let tracked = session
        .track(0, &previous, &reference, &pairs, "test")
        .expect("tracking");
    assert_eq!(tracked["accepted"], false);
    assert_eq!(tracked["tracking_supported"], true);
    assert!(tracked.get("geometry_covariance").is_none());
    assert_eq!(session.select()["decision"], "rejected");
    assert_eq!(
        tracked["reference_observation_sha256"],
        previous.evidence_sha256()
    );
    assert!(
        session
            .track(1, &previous, &reference, &pairs, "test")
            .is_err()
    );
}
