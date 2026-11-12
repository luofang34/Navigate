//! A fix that the vehicle passes outside its capture radius (NAV-TT-006).

use navigate_contract::CourseReference;

use super::leg_paths::eastbound_course;
use super::*;

/// The measured case: the estimated track passes a fly-over fix 156 m
/// abeam, and the capture radius is 100 m.
const ABEAM_M: f64 = 156.0;

/// A position `abeam_m` north of the equatorial inbound track and
/// `beyond_m` east of W1 at (0°, 0.5°E). A negative `beyond_m` is short
/// of W1.
fn abeam_of_w1(abeam_m: f64, beyond_m: f64) -> GeodeticPosition {
    pos(abeam_m / M_PER_DEG, 0.5 + beyond_m / M_PER_DEG)
}

/// Flies a track `abeam_m` north of W1 from 2 km short of it to 2 km
/// beyond it in 10 m steps, and returns each event that is not `None`
/// with the along-track position of its sample.
fn fly_abeam(exec: &mut PlanExecution<'_>, abeam_m: f64) -> std::vec::Vec<(f64, SequenceEvent)> {
    let mut events = std::vec::Vec::new();
    for step in 0_i32..=400 {
        let beyond_m = f64::from(step).mul_add(10.0, -2000.0);
        let event = exec.advance(&abeam_of_w1(abeam_m, beyond_m), 60.0);
        if event != SequenceEvent::None {
            events.push((beyond_m, event));
        }
    }
    events
}

#[test]
fn nav_tt_006_a_fly_over_fix_passed_abeam_sequences_at_the_perpendicular() {
    let wps = dogleg(wp("W1", 0.0, 0.5).with_turn(TurnType::FlyOver));
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    let events = fly_abeam(&mut exec, ABEAM_M);
    assert_eq!(exec.active_index(), 2, "W1 sequenced: {events:?}");
    assert_eq!(events.len(), 1, "one event only: {events:?}");
    let Some(&(beyond_m, event)) = events.first() else {
        return;
    };
    assert!(
        beyond_m > 0.0 && beyond_m <= 10.0,
        "first sample beyond the perpendicular: {beyond_m}"
    );
    let SequenceEvent::LegAdvanced {
        to_index: 2,
        turn: TurnType::FlyOver,
        reason: SequenceReason::Passed { cross_track_m },
    } = event
    else {
        panic!("a passed fix: {event:?}");
    };
    // North of an eastbound course is left of it: a negative miss.
    assert!(
        (cross_track_m + ABEAM_M).abs() < 0.5,
        "miss {cross_track_m}"
    );
}

#[test]
fn nav_tt_006_a_capture_on_a_sample_beyond_the_fix_is_one_capture_event() {
    // 50 m abeam and 50 m beyond W1: inside the capture radius and beyond
    // the perpendicular. Capture has priority and only one fix sequences.
    let wps = dogleg(wp("W1", 0.0, 0.5).with_turn(TurnType::FlyOver));
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    assert_eq!(
        exec.advance(&abeam_of_w1(50.0, 50.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyOver,
            reason: SequenceReason::Overflown,
        }
    );
    assert_eq!(exec.active_index(), 2);
}

#[test]
fn nav_tt_006_a_fly_by_fix_past_the_limit_missed_abeam_is_passed() {
    // A 135° corner at W1 exceeds the 121° fly-by boundary, so W1 does
    // not anticipate on the abeam line. A track 2 km to the right of the
    // inbound course misses the capture disc and sequences on the
    // perpendicular through W1 as Passed.
    let wps = [wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5), wp("W2", 0.5, 0.0)];
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    let events = fly_abeam(&mut exec, -2000.0);
    assert_eq!(events.len(), 1, "one event only: {events:?}");
    assert!(matches!(
        events.first(),
        Some(&(beyond_m, SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::Passed { cross_track_m },
        })) if beyond_m > 0.0 && (cross_track_m - 2000.0).abs() < 1.0
    ));
}

#[test]
fn nav_tt_006_a_passed_terminal_fix_completes_the_plan() {
    let wps = [wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5)];
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    let events = fly_abeam(&mut exec, ABEAM_M);
    assert_eq!(events.len(), 1, "one event only: {events:?}");
    assert!(matches!(
        events.first(),
        Some(&(
            _,
            SequenceEvent::PlanComplete {
                reason: SequenceReason::Passed { .. },
                ..
            }
        ))
    ));
    assert!(exec.is_complete());
}

#[test]
fn nav_tt_006_a_course_to_fix_leg_ends_at_the_perpendicular_to_its_course() {
    let wps = [
        wp("W0", 0.0, 0.0),
        wp("W1", 0.0, 0.5).with_path(eastbound_course(CourseReference::True)),
        wp("W2", 0.5, 0.5),
    ];
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    let events = fly_abeam(&mut exec, ABEAM_M);
    assert_eq!(events.len(), 1, "one event only: {events:?}");
    assert!(matches!(
        events.first(),
        Some(&(beyond_m, SequenceEvent::LegAdvanced {
            reason: SequenceReason::Passed { cross_track_m },
            ..
        })) if beyond_m > 0.0 && (cross_track_m + ABEAM_M).abs() < 0.5
    ));
}

#[test]
fn nav_tt_006_a_direct_to_leg_has_no_inbound_course_to_pass() {
    // W0 is on a direct-to leg: a position beyond it on any line does
    // not sequence it, because guidance steers back to it.
    let wps = dogleg(wp("W1", 0.0, 0.5));
    let mut exec = execution(&wps, ExecutionConfig::default());
    for lon_deg in [0.01, -0.01, 0.1] {
        assert_eq!(
            exec.advance(&pos(0.002, lon_deg), 60.0),
            SequenceEvent::None
        );
    }
    assert_eq!(exec.active_index(), 0);
}

#[test]
fn nav_tt_006_a_far_position_beyond_the_fix_sequences_it() {
    // The rule is a half-space test with no distance limit: a position
    // 50 km beyond W1 and 20 km off the course sequences W1.
    let wps = dogleg(wp("W1", 0.0, 0.5).with_turn(TurnType::FlyOver));
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    assert!(matches!(
        exec.advance(&abeam_of_w1(20_000.0, 50_000.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            reason: SequenceReason::Passed { .. },
            ..
        }
    ));
}

#[test]
fn nav_tt_006_a_position_beyond_several_fixes_sequences_one_per_call() {
    // Three collinear eastbound legs of 5 km. A position 2 km north of
    // the course and beyond W3 is beyond W1, W2 and W3.
    let step = 5000.0 / M_PER_DEG;
    let wps = [
        wp("W0", 0.0, 0.0),
        wp("W1", 0.0, step),
        wp("W2", 0.0, 2.0 * step),
        wp("W3", 0.0, 3.0 * step),
        wp("W4", 0.5, 3.0 * step),
    ];
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    let far = pos(2000.0 / M_PER_DEG, 3.2 * step);
    for to_index in 2..=4 {
        assert!(
            matches!(
                exec.advance(&far, 60.0),
                SequenceEvent::LegAdvanced { to_index: index, reason: SequenceReason::Passed { .. }, .. }
                    if index == to_index
            ),
            "one fix per call, now toward {to_index}"
        );
    }
    assert_eq!(exec.advance(&far, 60.0), SequenceEvent::None, "W4 is ahead");
}

#[test]
fn nav_tt_006_a_non_finite_position_passes_nothing() {
    let wps = dogleg(wp("W1", 0.0, 0.5).with_turn(TurnType::FlyOver));
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);
    let nan = GeodeticPosition::new(f64::NAN, 0.0, 0.0);
    assert_eq!(exec.advance(&nan, 60.0), SequenceEvent::None);
    assert_eq!(exec.active_index(), 1);
}
