//! NAV-TT-003: a fly-by fix sequences on the line abeam the distance of
//! turn anticipation, wherever the vehicle is across the track, and the
//! fly-by limit is a deliberate step a degree past 120°.

use navigate_contract::TurnDirection;

use super::*;

/// DTA at 60 m/s under the 18° default for the 90° dogleg at W1
/// (`60² / (9.80665 · tan 18°) ≈ 1129.803 m`).
const DTA_M: f64 = 1_129.803;

/// A position `to_go_m` short of W1 along the equatorial inbound track
/// and `left_m` north of it, left of the eastbound course.
fn abeam(to_go_m: f64, left_m: f64) -> GeodeticPosition {
    pos(left_m / M_PER_DEG, 0.5 - to_go_m / M_PER_DEG)
}

/// The distance to go at which a vehicle `left_m` off the track, closing
/// on W1 in steps of 5 m, first sequences, with the reason.
fn sequencing_station(left_m: f64) -> (f64, SequenceReason) {
    let wps = dogleg(wp("W1", 0.0, 0.5));
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    let mut to_go = 1_330.0;
    while to_go > -500.0 {
        if let SequenceEvent::LegAdvanced { reason, .. } = exec.advance(&abeam(to_go, left_m), 60.0)
        {
            return (to_go, reason);
        }
        to_go -= 5.0;
    }
    panic!("{left_m}: never sequenced");
}

#[test]
fn nav_tt_003_off_the_track_the_fix_sequences_on_the_abeam_line() {
    for fraction in [0.0, 0.3, -0.3, 0.8, -0.8, 1.5, -1.5] {
        let left_m = fraction * DTA_M;
        let (to_go, reason) = sequencing_station(left_m);
        assert_eq!(reason, SequenceReason::Anticipated, "{fraction}");
        // The first sample past the line, within one step of it.
        assert!(
            to_go < DTA_M && to_go > DTA_M - 10.0,
            "{fraction}: sequenced {to_go} m short"
        );
    }
}

#[test]
fn nav_tt_006_a_vehicle_beyond_the_fix_is_passed_not_a_fresh_approach() {
    // Beyond the perpendicular through W1, off the track and moving
    // away: the distance to go is negative, so it is no turn to
    // anticipate, and the leg ends as passed with its miss.
    let wps = dogleg(wp("W1", 0.0, 0.5));
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    let event = exec.advance(&abeam(-300.0, 2_500.0), 60.0);
    let SequenceEvent::LegAdvanced {
        to_index: 2,
        reason: SequenceReason::Passed { cross_track_m },
        ..
    } = event
    else {
        panic!("{event:?}");
    };
    // North of an eastbound course is left: negative.
    assert!((cross_track_m + 2_500.0).abs() < 5.0, "{cross_track_m}");
}

#[test]
fn a_direct_to_leg_does_not_anticipate_off_its_track() {
    // The first leg runs from the present position: no inbound course,
    // so no abeam line. Off the fix by more than the capture radius it
    // does not sequence however close along the equator it is.
    let wps = dogleg(wp("W1", 0.0, 0.5));
    let mut exec = execution(&wps, ExecutionConfig::default());
    assert_eq!(exec.advance(&pos(0.01, 0.0), 60.0), SequenceEvent::None);
    assert_eq!(exec.active_index(), 0);
}

/// The point `distance_deg` of arc from `from` on the true bearing
/// `bearing_deg`, on the sphere.
fn destination(from: (f64, f64), bearing_deg: f64, distance_deg: f64) -> (f64, f64) {
    let (lat, lon) = (from.0.to_radians(), from.1.to_radians());
    let (theta, delta) = (bearing_deg.to_radians(), distance_deg.to_radians());
    let lat2 = (lat.sin() * delta.cos() + lat.cos() * delta.sin() * theta.cos()).asin();
    let lon2 =
        lon + (theta.sin() * delta.sin() * lat.cos()).atan2(delta.cos() - lat.sin() * lat2.sin());
    (lat2.to_degrees(), lon2.to_degrees())
}

/// The corner at W1 of a turn of `change_deg` to the right from the
/// eastbound inbound track, and whether a vehicle on the track inside
/// the DTA of that turn anticipates it.
fn turn_of(change_deg: f64) -> (Corner, bool) {
    let (lat, lon) = destination((0.0, 0.5), 90.0 + change_deg, 0.5);
    let wps = [wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5), wp("W2", lat, lon)];
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    let corner = exec.corner(1);
    // Inside the capture-free stretch of any DTA of these turns.
    let event = exec.advance(&short_of_w1(1_500.0), 60.0);
    let anticipated = matches!(
        event,
        SequenceEvent::LegAdvanced {
            reason: SequenceReason::Anticipated,
            ..
        }
    );
    (corner, anticipated)
}

#[test]
fn nav_tt_003_the_fly_by_limit_is_a_degree_past_120() {
    for change_deg in [119.9, 120.0 - 1e-9, 120.0, 120.0 + 1e-9, 120.5, 120.9] {
        let (corner, anticipated) = turn_of(change_deg);
        assert!(
            matches!(corner, Corner::Anticipated { .. }) && anticipated,
            "{change_deg}: {corner:?}"
        );
    }
    let (corner, anticipated) = turn_of(121.1);
    assert!(
        matches!(corner, Corner::PastFlyByLimit { .. }) && !anticipated,
        "{corner:?}"
    );
}

#[test]
fn mandatory_turn_direction_opposing_the_short_arc_refuses_anticipation() {
    // W0(0,0) -> W1(0,0.5) -> W2(0.5,0.5) is a 90° left turn.
    let matching = [
        wp("W0", 0.0, 0.0),
        wp("W1", 0.0, 0.5),
        wp("W2", 0.5, 0.5).with_turn_direction(TurnDirection::Left),
    ];
    let mut exec_left = execution(&matching, ExecutionConfig::default());
    capture_w0(&mut exec_left);
    assert!(matches!(
        exec_left.corner(1),
        Corner::Anticipated { track_change_rad } if (track_change_rad.to_degrees() - 90.0).abs() < 0.1
    ));
    assert_eq!(
        exec_left.advance(&short_of_w1(1_120.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::Anticipated,
        }
    );
    assert_eq!(
        exec_left.active_leg().expect("active leg").turn_direction(),
        Some(TurnDirection::Left)
    );

    // Declaring a mandatory right turn on W2 requires a 270° loop (> 121°),
    // so the corner is PastFlyByLimit and waits for the capture radius.
    let opposing = [
        wp("W0", 0.0, 0.0),
        wp("W1", 0.0, 0.5),
        wp("W2", 0.5, 0.5).with_turn_direction(TurnDirection::Right),
    ];
    let mut exec_right = execution(&opposing, ExecutionConfig::default());
    capture_w0(&mut exec_right);
    assert!(matches!(
        exec_right.corner(1),
        Corner::PastFlyByLimit { track_change_rad } if (track_change_rad.to_degrees() - 270.0).abs() < 0.1
    ));
    assert_eq!(
        exec_right.advance(&short_of_w1(1_120.0), 60.0),
        SequenceEvent::None
    );
    assert_eq!(
        exec_right.advance(&short_of_w1(90.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::Overflown,
        }
    );
    assert_eq!(
        exec_right
            .active_leg()
            .expect("active leg")
            .turn_direction(),
        Some(TurnDirection::Right)
    );
}

#[test]
fn the_corners_of_a_plan_are_reported_per_fix() {
    let wps = [
        wp("W0", 0.0, 0.0),
        wp("W1", 0.0, 0.5),
        wp("W2", 0.5, 0.5).with_turn(TurnType::FlyOver),
        wp("W3", 0.5, 1.0),
    ];
    let exec = execution(&wps, ExecutionConfig::default());
    let corners: std::vec::Vec<Corner> = exec.corners().collect();
    assert_eq!(corners.len(), 4);
    assert_eq!(
        corners[0],
        Corner::Captured,
        "the first fix has no inbound track"
    );
    assert!(
        matches!(corners[1], Corner::Anticipated { track_change_rad } if (track_change_rad.to_degrees() - 90.0).abs() < 0.1)
    );
    assert_eq!(corners[2], Corner::Captured, "a fly-over fix");
    assert_eq!(corners[3], Corner::Captured, "the terminal fix");
}
