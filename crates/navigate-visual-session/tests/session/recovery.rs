//! Recovery from a wrong first anchor and from a stale map.

use crate::support::*;
use nalgebra::Vector3;
use navigate_visual_session::{
    AnchorDecision, AnchorRejection, Confirmation, MapPose, RevisionCause, SessionEvent,
};

#[test]
fn agreeing_frames_relocate_a_track_after_a_wrong_first_anchor() {
    let mut flight = Flight::standard();
    flight.heading_bias_rad = 0.001;
    for i in 0..50 {
        flight.fly_to(nadir(
            Vector3::new(4.0 * i as f64, 2.0 * i as f64, 120.0),
            0.4,
        ));
    }
    let keys: Vec<_> = flight.truth.iter().map(|(k, _, _)| *k).collect();
    // A repeated roof pattern matches 80 m away from the true place.
    let wrong = anchor(&flight, &keys[0], Vector3::new(80.0, -20.0, 0.0));
    assert!(matches!(
        flight.session.submit_anchor(wrong).unwrap(),
        AnchorDecision::Accepted { .. }
    ));
    assert!(
        matches!(
            flight.session.pose_at(&keys[5]).unwrap().map,
            MapPose::Located {
                confirmation: Confirmation::Unconfirmed,
                ..
            }
        ),
        "one anchor is not confirmed"
    );

    let first_true = flight
        .session
        .submit_anchor(anchor(&flight, &keys[20], Vector3::zeros()))
        .unwrap();
    assert!(
        matches!(
            first_true,
            AnchorDecision::Rejected(AnchorRejection::Inconsistent { .. })
        ),
        "one disagreeing frame does not move the track: {first_true:?}"
    );
    let second_true = flight
        .session
        .submit_anchor(anchor(&flight, &keys[40], Vector3::zeros()))
        .unwrap();
    assert_eq!(
        second_true,
        AnchorDecision::Relocated {
            agreeing: 2,
            retracted: 1
        }
    );

    let events = flight.session.drain_events().events;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SessionEvent::AnchorRetracted { frame } if *frame == keys[0]))
    );
    assert!(events.iter().any(
        |e| matches!(e, SessionEvent::TrajectoryRevised(r) if r.cause == RevisionCause::Relocation)
    ));
    for (k, truth, _) in &flight.truth[15..] {
        let error = error_m(&flight.session, k, truth).unwrap();
        assert!(
            error < 5.0,
            "frame {} follows the agreeing anchors: {error} m",
            k.index
        );
    }
    assert!(matches!(
        flight.session.pose_at(&keys[45]).unwrap().map,
        MapPose::Located {
            confirmation: Confirmation::Confirmed { anchors: 2 },
            ..
        }
    ));
}

#[test]
fn a_shared_map_offset_moves_the_track_as_one_body() {
    // An old orthoimage moved 6 m east. Anchors agree with each other but not
    // with the truth. The session follows the map and does not claim more.
    let mut flight = Flight::standard();
    for i in 0..40 {
        flight.fly_to(nadir(Vector3::new(0.0, 5.0 * i as f64, 90.0), 1.57));
    }
    let keys: Vec<_> = flight.truth.iter().map(|(k, _, _)| *k).collect();
    let shift = Vector3::new(6.0, 0.0, 0.0);
    for index in [0, 13, 26, 39] {
        let decision = flight
            .session
            .submit_anchor(anchor(&flight, &keys[index], shift))
            .unwrap();
        assert!(
            matches!(decision, AnchorDecision::Accepted { .. }),
            "{decision:?}"
        );
    }
    for (k, truth, _) in &flight.truth {
        let MapPose::Located {
            pose,
            position_bound_m,
            ..
        } = flight.session.pose_at(k).unwrap().map
        else {
            panic!("frame {} is located", k.index);
        };
        let offset = pose.translation.vector.xy() - truth.translation.vector.xy();
        assert!(
            (offset - shift.xy()).norm() < 1.5,
            "frame {} keeps the map offset: {offset:?}",
            k.index
        );
        assert!(
            position_bound_m >= 1.0,
            "the bound includes the declared map error"
        );
    }
}
