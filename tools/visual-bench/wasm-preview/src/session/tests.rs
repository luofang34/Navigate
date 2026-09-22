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
        .refine(
            1,
            &reference,
            &pairs,
            "custom-classical-test",
            [40.0, -74.0],
        )
        .expect("fit");
    assert_eq!(first["accepted"], true);
    let second = session
        .refine(
            1,
            &reference,
            &pairs,
            "custom-classical-test",
            [40.0, -74.0],
        )
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
        .refine(2, &reference, &pairs, "another-backend", [40.0, -74.0])
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
            .refine(7, &reference, &pairs, "matcher", [40.0, -74.0])
            .expect("fit")["accepted"],
        true
    );
    session.invalidate(7).expect("replace");
    assert_eq!(session.select()["decision"], "rejected");
    reference.depth_m.fill(0.0);
    assert_eq!(
        session
            .refine(7, &reference, &pairs, "matcher", [40.0, -74.0])
            .expect("rejected report")["accepted"],
        false
    );
    assert_eq!(session.select()["decision"], "rejected");
}
