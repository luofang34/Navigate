//! Odometry continuity, anchor revisions, hover, seek, and late results.

use crate::support::*;
use nalgebra::Vector3;
use navigate_visual_session::{
    AnchorDecision, AnchorRejection, Confirmation, FusionEligibility, MapPose, RevisionCause,
    SegmentStart, SessionError, SessionEvent, Unlocated,
};

fn straight(flight: &mut Flight, frames: u64, step_m: f64) {
    for i in 0..frames {
        flight.fly_to(nadir(Vector3::new(step_m * i as f64, 0.0, 100.0), 0.0));
    }
}

#[test]
fn map_corrections_revise_map_poses_without_moving_odometry() {
    let mut flight = Flight::standard();
    flight.heading_bias_rad = 0.002;
    flight.scale_error = 0.03;
    straight(&mut flight, 60, 5.0);
    let keys: Vec<_> = flight.truth.iter().map(|(k, _, _)| *k).collect();
    let before: Vec<_> = keys
        .iter()
        .map(|k| odometry_of(&flight.session, k))
        .collect();
    assert!(
        matches!(
            flight.session.pose_at(&keys[10]).unwrap().map,
            MapPose::Unlocated(Unlocated::NoAnchor)
        ),
        "no map pose before an anchor"
    );
    flight.session.drain_events();

    for index in [0, 30, 59] {
        let observation = anchor(&flight, &keys[index], Vector3::zeros());
        let decision = flight.session.submit_anchor(observation).unwrap();
        assert!(
            matches!(decision, AnchorDecision::Accepted { .. }),
            "anchor {index}: {decision:?}"
        );
    }
    let after: Vec<_> = keys
        .iter()
        .map(|k| odometry_of(&flight.session, k))
        .collect();
    assert_eq!(before, after, "odometry poses never change");

    let worst = flight
        .truth
        .iter()
        .filter_map(|(k, truth, _)| error_m(&flight.session, k, truth))
        .fold(0.0_f64, f64::max);
    assert!(
        worst < 6.0,
        "map poses between anchors follow the anchors: worst {worst} m"
    );
    let batch = flight.session.drain_events();
    let revisions: Vec<_> = batch
        .events
        .iter()
        .filter_map(|e| match e {
            SessionEvent::TrajectoryRevised(r) => Some(r),
            _ => None,
        })
        .collect();
    assert!(
        revisions.len() >= 2,
        "each anchor after the first revises earlier map poses"
    );
    assert!(
        revisions.iter().all(|r| r.cause == RevisionCause::Anchor
            && r.ranges.iter().all(|range| range.to == keys[59]))
    );
    match flight.session.pose_at(&keys[45]).unwrap().map {
        MapPose::Located {
            confirmation,
            position_bound_m,
            ..
        } => {
            // Frames are 1 s apart: the anchors at 30 s and 59 s are within the
            // 30 s window of frame 45; the anchor at 0 s is not.
            assert_eq!(confirmation, Confirmation::Confirmed { anchors: 2 });
            assert!(position_bound_m.is_finite() && position_bound_m > 0.0);
        }
        other => panic!("frame 45 is located: {other:?}"),
    }
}

#[test]
fn hovering_frames_do_not_add_anchor_evidence() {
    let mut flight = Flight::standard();
    for _ in 0..20 {
        flight.fly_to(nadir(Vector3::new(0.0, 0.0, 80.0), 0.3));
    }
    let keys: Vec<_> = flight.truth.iter().map(|(k, _, _)| *k).collect();
    let first = flight
        .session
        .submit_anchor(anchor(&flight, &keys[0], Vector3::zeros()))
        .unwrap();
    assert_eq!(
        first,
        AnchorDecision::Accepted {
            eligibility: FusionEligibility::Independent
        }
    );
    for k in &keys[1..] {
        let decision = flight
            .session
            .submit_anchor(anchor(&flight, k, Vector3::zeros()))
            .unwrap();
        assert_eq!(
            decision,
            AnchorDecision::SameViewpoint,
            "a hovering camera keeps one keyframe"
        );
    }
    assert_eq!(flight.session.usage().anchors, 1);
    assert_eq!(flight.session.usage().keyframes, 1);
    let repeated = anchor(&flight, &keys[0], Vector3::zeros());
    assert!(
        matches!(
            flight.session.submit_anchor(repeated),
            Err(SessionError::RepeatedEvidence(_))
        ),
        "one frame is one piece of evidence"
    );
}

#[test]
fn anchors_in_one_map_cell_share_map_error() {
    let mut flight = Flight::standard();
    straight(&mut flight, 30, 5.0);
    let keys: Vec<_> = flight.truth.iter().map(|(k, _, _)| *k).collect();
    flight
        .session
        .submit_anchor(anchor(&flight, &keys[0], Vector3::zeros()))
        .unwrap();
    let second = flight
        .session
        .submit_anchor(anchor(&flight, &keys[29], Vector3::zeros()))
        .unwrap();
    match second {
        AnchorDecision::Accepted {
            eligibility: FusionEligibility::SharedMapError { earlier, .. },
        } => assert_eq!(earlier, keys[0]),
        other => panic!("second anchor in the cell shares map error: {other:?}"),
    }
}

#[test]
fn a_seek_starts_an_unlocated_segment_and_late_anchors_revise_only_their_segment() {
    let mut flight = Flight::standard();
    straight(&mut flight, 20, 5.0);
    let first_epoch: Vec<_> = flight.truth.iter().map(|(k, _, _)| *k).collect();
    flight
        .session
        .submit_anchor(anchor(&flight, &first_epoch[0], Vector3::zeros()))
        .unwrap();
    flight
        .session
        .submit_anchor(anchor(&flight, &first_epoch[19], Vector3::zeros()))
        .unwrap();

    flight.seek();
    for i in 0..20 {
        flight.fly_to(nadir(
            Vector3::new(500.0 + 5.0 * i as f64, 200.0, 100.0),
            1.0,
        ));
    }
    let second_epoch: Vec<_> = flight.truth[20..].iter().map(|(k, _, _)| *k).collect();
    let started = flight.session.drain_events().events.into_iter().any(|e| matches!(e, SessionEvent::SegmentStarted { first, reason: SegmentStart::NewEpoch, .. } if first == second_epoch[0]));
    assert!(started, "a new epoch starts a new segment");
    assert!(
        matches!(
            flight.session.pose_at(&second_epoch[10]).unwrap().map,
            MapPose::Unlocated(Unlocated::NoAnchor)
        ),
        "the first epoch does not locate the second"
    );

    let before = error_m(&flight.session, &first_epoch[10], &flight.truth[10].1).unwrap();
    let late = anchor(&flight, &second_epoch[5], Vector3::zeros());
    assert!(
        matches!(
            flight.session.submit_anchor(late).unwrap(),
            AnchorDecision::Accepted { .. }
        ),
        "a result for an older frame is accepted"
    );
    let after = error_m(&flight.session, &first_epoch[10], &flight.truth[10].1).unwrap();
    assert!(
        (after - before).abs() < 1e-6,
        "the other segment keeps its poses"
    );
    let second_segment = flight
        .session
        .pose_at(&second_epoch[0])
        .unwrap()
        .odometry
        .unwrap()
        .segment;
    let revisions: Vec<_> = flight
        .session
        .drain_events()
        .events
        .into_iter()
        .filter_map(|e| match e {
            SessionEvent::TrajectoryRevised(r) => Some(r),
            _ => None,
        })
        .collect();
    assert!(
        !revisions.is_empty()
            && revisions
                .iter()
                .all(|r| r.ranges.iter().all(|range| range.segment == second_segment)),
        "only the late anchor's segment is revised: {revisions:?}"
    );
    let error = error_m(&flight.session, &second_epoch[19], &flight.truth[39].1).unwrap();
    assert!(
        error < 3.0,
        "the newest frame follows the late anchor: {error} m"
    );
}

#[test]
fn motion_without_scale_is_refused_and_an_anchor_after_a_gap_starts_a_segment() {
    let mut flight = Flight::standard();
    straight(&mut flight, 5, 5.0);
    let (k4, _, f4) = flight.last().clone();
    let k4: navigate_visual_session::FrameKey = k4;
    let gap_frame = image_frame(100);
    let gap = key(0, 10);
    flight
        .session
        .observe_frame(record(gap, &gap_frame))
        .unwrap();
    let motion = navigate_visual_session::RelativeMotion {
        from: k4,
        to: gap,
        from_to: navigate_visual_session::Pose::translation(1.0, 0.0, 0.0),
        scale: navigate_visual_session::ScaleSource::Unknown,
        from_observation_sha256: f4.evidence_sha256(),
        to_observation_sha256: gap_frame.evidence_sha256(),
    };
    assert!(matches!(
        flight.session.apply_motion(motion),
        Err(SessionError::ScaleUnobservable { .. })
    ));
    assert!(matches!(
        flight.session.pose_at(&gap).unwrap().map,
        MapPose::Unlocated(Unlocated::NoOdometry)
    ));

    let truth = nadir(Vector3::new(300.0, 0.0, 100.0), 0.0);
    let estimate = map_estimate(&gap_frame, &truth, Vector3::zeros());
    let observation =
        navigate_visual_session::AnchorObservation::from_estimate(gap, &estimate, reliability());
    assert!(matches!(
        flight.session.submit_anchor(observation).unwrap(),
        AnchorDecision::Accepted { .. }
    ));
    let events = flight.session.drain_events().events;
    assert!(events.iter().any(|e| matches!(e, SessionEvent::SegmentStarted { first, reason: SegmentStart::AnchorAfterGap, .. } if *first == gap)));
    assert!(error_m(&flight.session, &gap, &truth).unwrap() < 1.0);
    assert!(
        matches!(
            flight
                .session
                .submit_anchor(anchor(&flight, &k4, Vector3::new(0.0, 0.0, 0.0))),
            Ok(AnchorDecision::Accepted { .. })
        ),
        "the earlier segment can still take anchors"
    );
}

#[test]
fn changed_areas_and_wrong_places_are_refused_without_moving_the_track() {
    let mut flight = Flight::standard();
    straight(&mut flight, 40, 5.0);
    let keys: Vec<_> = flight.truth.iter().map(|(k, _, _)| *k).collect();
    flight
        .session
        .submit_anchor(anchor(&flight, &keys[0], Vector3::zeros()))
        .unwrap();
    flight
        .session
        .submit_anchor(anchor(&flight, &keys[20], Vector3::zeros()))
        .unwrap();
    let mut changed = anchor(&flight, &keys[25], Vector3::zeros());
    changed.reliability.change = navigate_visual_session::RegionChange::Changed;
    assert_eq!(
        flight.session.submit_anchor(changed).unwrap(),
        AnchorDecision::Rejected(AnchorRejection::ChangedRegion)
    );
    let revision = flight.session.revision();
    let wrong = anchor(&flight, &keys[39], Vector3::new(60.0, 0.0, 0.0));
    let decision = flight.session.submit_anchor(wrong).unwrap();
    assert!(
        matches!(
            decision,
            AnchorDecision::Rejected(AnchorRejection::Inconsistent { .. })
        ),
        "a repeated building far from the track is refused: {decision:?}"
    );
    assert_eq!(
        flight.session.revision(),
        revision,
        "a refused anchor does not revise poses"
    );
}

#[test]
fn returned_map_poses_stay_valid_through_the_host_tracking_loop() {
    // The host renders each tracking reference at the returned map pose, and
    // the visual crate refuses quaternions whose norm differs from one by more
    // than 1e-8. Blended corrections and composed motions must not drift.
    use navigate_visual::{LocalizerConfig, PosePrior, PoseVerifier, TrackingReference};
    let mut flight = Flight::standard();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let mut previous: Option<(
        navigate_visual_session::FrameKey,
        navigate_visual_session::Pose,
        navigate_visual::Frame,
    )> = None;
    for i in 0..80_u64 {
        let truth = nadir(
            Vector3::new(2.0 * i as f64, 0.3 * i as f64, 100.0),
            0.004 * i as f64,
        );
        let frame = image_frame(i);
        let k = key(0, i);
        flight.session.observe_frame(record(k, &frame)).unwrap();
        if let Some((pk, pt, pf)) = &previous
            && let MapPose::Located { pose, .. } = flight.session.pose_at(pk).unwrap().map
        {
            let reference = ground_view(&pose, 0.0);
            let surface = TrackingReference {
                observation: pf,
                surface: &reference,
            };
            let prior = PosePrior {
                pose: navigate_visual_session::to_camera(&pose),
                position_radius_m: 500.0,
                attitude_radius_rad: 1.0,
            };
            let proposal = verifier
                .track(
                    &frame,
                    surface,
                    &prior,
                    &ground_matches(pt, &truth, 0.0),
                    "synthetic",
                )
                .unwrap_or_else(|e| panic!("frame {i}: {e}"));
            let motion = navigate_visual_session::RelativeMotion::from_tracking(
                *pk,
                k,
                &reference.pose,
                &proposal,
            );
            flight.session.apply_motion(motion).unwrap();
        }
        if i == 0 || i == 20 {
            let offset = if i == 0 {
                Vector3::zeros()
            } else {
                Vector3::new(3.0, -2.0, 0.0)
            };
            let estimate = map_estimate(&frame, &truth, offset);
            let observation = navigate_visual_session::AnchorObservation::from_estimate(
                k,
                &estimate,
                reliability(),
            );
            flight.session.submit_anchor(observation).unwrap();
        }
        previous = Some((k, truth, frame));
    }
    // Hosts can also read the returned isometries directly.
    for i in 0..80_u64 {
        let pose = flight.session.pose_at(&key(0, i)).unwrap();
        if let MapPose::Located {
            pose,
            map_from_odom,
            ..
        } = pose.map
        {
            for rotation in [pose.rotation, map_from_odom.rotation] {
                assert!(
                    (rotation.norm() - 1.0).abs() <= 1e-12,
                    "frame {i}: norm {}",
                    rotation.norm()
                );
            }
        }
    }
}
