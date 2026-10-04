//! Input contracts: refused inputs change no state, and evidence is not counted twice.

use crate::support::*;
use nalgebra::{Matrix3, Vector3};
use navigate_visual_session::{
    AnchorDecision, Confirmation, MapPose, Pose, RevisitConstraint, ScaleSource, SessionConfig,
    SessionError, SessionEvent, SessionLimits, SurfaceModel,
};

fn small(max_keyframes: usize) -> SessionConfig {
    let limits = SessionLimits {
        max_keyframes,
        ..SessionLimits::default()
    };
    SessionConfig {
        limits,
        ..SessionConfig::standard()
    }
}

fn fly(flight: &mut Flight, frames: u64, step_m: f64) {
    for i in 0..frames {
        flight.fly_to(nadir(Vector3::new(step_m * i as f64, 0.0, 100.0), 0.0));
    }
}

#[test]
fn kept_priors_of_evicted_keyframes_do_not_confirm_a_single_anchor() {
    let mut flight = Flight::new(small(4));
    fly(&mut flight, 7, 5.0);
    let k6 = flight.truth[6].0;
    flight
        .session
        .submit_anchor(anchor(&flight, &k6, Vector3::zeros()))
        .unwrap();
    for i in 7..16 {
        flight.fly_to(nadir(Vector3::new(5.0 * i as f64, 0.0, 100.0), 0.0));
    }
    let newest = flight.last().0;
    match flight.session.pose_at(&newest).unwrap().map {
        MapPose::Located { confirmation, .. } => {
            assert_eq!(confirmation, Confirmation::Unconfirmed)
        }
        other => panic!("the track stays located: {other:?}"),
    }
}

#[test]
fn an_agreeing_anchor_near_the_consensus_is_not_retracted() {
    let mut config = SessionConfig::standard();
    config.anchors.direct_bound_m = 0.01;
    let mut flight = Flight::new(config);
    fly(&mut flight, 10, 2.0);
    let keys: Vec<_> = flight.truth.iter().map(|(k, _, _)| *k).collect();
    flight
        .session
        .submit_anchor(anchor(&flight, &keys[0], Vector3::zeros()))
        .unwrap();
    flight
        .session
        .submit_anchor(anchor(&flight, &keys[1], Vector3::zeros()))
        .unwrap();
    let decision = flight
        .session
        .submit_anchor(anchor(&flight, &keys[6], Vector3::zeros()))
        .unwrap();
    // Both agreeing frames are within one keyframe of the first anchor, so
    // they add no new evidence.
    assert_eq!(decision, AnchorDecision::SameViewpoint);
    let events = flight.session.drain_events().events;
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, SessionEvent::AnchorRetracted { frame } if *frame == keys[0])),
        "{events:?}"
    );
}

#[test]
fn a_non_finite_closure_is_refused_without_state_change() {
    let mut flight = Flight::standard();
    fly(&mut flight, 40, 5.0);
    let (earlier, later) = (flight.truth[0].0, flight.truth[39].0);
    flight
        .session
        .submit_anchor(anchor(&flight, &earlier, Vector3::zeros()))
        .unwrap();
    let (usage, revision) = (flight.session.usage(), flight.session.revision());
    let mut nan = Pose::identity();
    nan.translation.vector.x = f64::NAN;
    let constraint = RevisitConstraint {
        earlier,
        later,
        earlier_to_later: nan,
        sigma_m: 1.0,
        sigma_rad: 0.01,
        quality: map_estimate(&flight.truth[0].2, &flight.truth[0].1, Vector3::zeros()).quality,
        scale: ScaleSource::MapDepth { map: map() },
        backend: "test".into(),
    };
    assert!(matches!(
        flight.session.submit_revisit(constraint),
        Err(SessionError::Invalid { .. })
    ));
    assert_eq!(
        (flight.session.usage(), flight.session.revision()),
        (usage, revision)
    );
}

#[test]
fn refused_anchors_change_no_state_and_attachment_keeps_its_keyframe() {
    let mut flight = Flight::new(small(4));
    fly(&mut flight, 16, 5.0);
    // Tracking skips frame 16 and continues from frame 15 to frame 17.
    let (k15, truth15, f15) = flight.last().clone();
    let (skipped, skipped_frame) = (key(0, 16), image_frame(16));
    let (next, next_frame) = (key(0, 17), image_frame(17));
    flight
        .session
        .observe_frame(record(skipped, &skipped_frame))
        .unwrap();
    flight
        .session
        .observe_frame(record(next, &next_frame))
        .unwrap();
    let motion = navigate_visual_session::RelativeMotion {
        from: k15,
        to: next,
        from_to: Pose::translation(10.0, 0.0, 0.0),
        scale: ScaleSource::MapDepth { map: map() },
        from_observation_sha256: f15.evidence_sha256(),
        to_observation_sha256: next_frame.evidence_sha256(),
    };
    flight.session.apply_motion(motion).unwrap();
    let (usage, revision) = (flight.session.usage(), flight.session.revision());
    let estimate = map_estimate(&skipped_frame, &truth15, Vector3::zeros());
    let refused = navigate_visual_session::AnchorObservation::from_estimate(
        skipped,
        &estimate,
        reliability(),
    );
    assert!(matches!(
        flight.session.submit_anchor(refused),
        Err(SessionError::NoOdometry(_))
    ));
    assert_eq!(
        flight.session.local_frame(),
        None,
        "a refused anchor does not fix the local frame"
    );
    assert_eq!(
        (flight.session.usage(), flight.session.revision()),
        (usage, revision)
    );

    // Frame 0 is older than every kept keyframe. Attaching it must not evict its own keyframe.
    let k0 = flight.truth[0].0;
    let decision = flight
        .session
        .submit_anchor(anchor(&flight, &k0, Vector3::zeros()))
        .unwrap();
    assert!(
        matches!(decision, AnchorDecision::Accepted { .. }),
        "{decision:?}"
    );
    assert_eq!(flight.session.usage().keyframes, 4);
}

#[test]
fn an_anchor_covariance_without_a_square_root_is_refused() {
    let mut flight = Flight::standard();
    fly(&mut flight, 3, 5.0);
    let k = flight.truth[0].0;
    let mut observation = anchor(&flight, &k, Vector3::zeros());
    observation.reliability.surface = SurfaceModel::Surface;
    observation.geometry_position_m2 = Matrix3::zeros();
    assert!(matches!(
        flight.session.submit_anchor(observation),
        Err(SessionError::Invalid { .. })
    ));
    assert_eq!(
        flight.session.local_frame(),
        None,
        "a refused first anchor does not fix the local frame"
    );
    assert_eq!(flight.session.usage().anchors, 0);
}

fn wrong_ground(
    flight: &Flight,
    index: usize,
    error_rad: f64,
) -> navigate_visual_session::GroundPlaneObservation {
    let (k, truth, _) = &flight.truth[index];
    // The true ground normal in the camera, tilted by `error_rad`, with an understated error.
    let up =
        nalgebra::UnitQuaternion::from_euler_angles(error_rad, 0.0, 0.0) * nalgebra::Vector3::z();
    navigate_visual_session::GroundPlaneObservation {
        frame: *k,
        normal_camera: nalgebra::Unit::new_normalize(truth.rotation.inverse() * up),
        geometry_sigma_rad: 0.005,
        terrain: navigate_visual_session::TerrainNormal {
            normal: nalgebra::Vector3::z_axis(),
            sigma_rad: 0.0,
        },
        relief_ratio: 0.0,
    }
}

#[test]
fn wrong_ground_planes_cannot_hold_the_attitude_or_lock_out_anchors() {
    use navigate_visual_session::GroundDecision;
    let mut flight = Flight::standard();
    fly(&mut flight, 30, 5.0);
    let (usage, revision) = (flight.session.usage(), flight.session.revision());
    for index in 5..10 {
        let decision = flight
            .session
            .submit_ground_plane(wrong_ground(&flight, index, 0.35))
            .unwrap();
        assert_eq!(
            decision,
            GroundDecision::Unanchored,
            "the odometry frame has no map attitude"
        );
    }
    assert_eq!(
        (flight.session.usage(), flight.session.revision()),
        (usage, revision),
        "refusals change nothing"
    );

    for index in [0, 29] {
        let k = flight.truth[index].0;
        let decision = flight
            .session
            .submit_anchor(anchor(&flight, &k, Vector3::zeros()))
            .unwrap();
        assert!(
            matches!(decision, AnchorDecision::Accepted { .. }),
            "{decision:?}"
        );
    }
    for index in 10..16 {
        flight
            .session
            .submit_ground_plane(wrong_ground(&flight, index, 0.09))
            .unwrap();
    }
    assert_eq!(flight.session.usage().anchors, 2, "correct anchors stay");
    for (k, truth, _) in &flight.truth {
        if let MapPose::Located {
            pose,
            tilt_bound_rad: Some(bound),
            ..
        } = flight.session.pose_at(k).unwrap().map
        {
            let axis = nalgebra::Vector3::new(0.0, 0.0, -1.0);
            let error = (pose.rotation * axis).angle(&(truth.rotation * axis));
            assert!(
                error <= 3.0 * bound + 0.5_f64.to_radians(),
                "frame {}: {:.2} deg error, {:.2} deg bound",
                k.index,
                error.to_degrees(),
                bound.to_degrees()
            );
        }
    }
}
