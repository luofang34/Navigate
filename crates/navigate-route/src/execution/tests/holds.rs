//! Racetrack entry, hold circuit sequencing (`HA`, `HF`, `HM`), and
//! course-to-altitude (`CA`) termination tests (`docs/leg-requirements.md`).

use core::f64::consts::FRAC_PI_2;

use navigate_contract::{
    AltitudeConstraint, CourseReference, HoldGeometry, HoldLegLength, LateralReference,
    TurnDirection,
};
use navigate_geodesy::{
    cross_track_from_course_m, cross_track_m, destination_point, initial_bearing_rad,
};

use super::hold::holding_side_offset_m;
use super::*;

const EAST_RAD: f64 = FRAC_PI_2;

fn pos_alt(lat_deg: f64, lon_deg: f64, alt_m: f64) -> GeodeticPosition {
    GeodeticPosition::new(lat_deg.to_radians(), lon_deg.to_radians(), alt_m)
}

fn eastbound_hold(turn: TurnDirection, leg: HoldLegLength) -> HoldGeometry {
    HoldGeometry::new(EAST_RAD, CourseReference::True, turn, leg)
}

/// Integrates a proportional cross-track follower on `leg.lateral_reference()`
/// for one `dt_s` step on the WGS84 sphere, climbing at `climb_mps`.
fn step_guided(
    exec: &PlanExecution<'_>,
    position: GeodeticPosition,
    speed_mps: f64,
    climb_mps: f64,
    dt_s: f64,
) -> GeodeticPosition {
    let leg = exec.active_leg().expect("active leg");
    let ellipsoid = exec.ellipsoid();
    let (course_rad, xtk_m) = match leg.lateral_reference() {
        LateralReference::PresentPosition => {
            (initial_bearing_rad(&position, &leg.to.position), 0.0)
        }
        LateralReference::Track { from } => (
            initial_bearing_rad(&from, &leg.to.position),
            cross_track_m(&position, &from, &leg.to.position, ellipsoid).unwrap_or(0.0),
        ),
        LateralReference::Course { course_rad } => (
            course_rad,
            cross_track_from_course_m(&position, &leg.to.position, course_rad, ellipsoid),
        ),
        LateralReference::DisplacedCourse { anchor, course_rad } => (
            course_rad,
            cross_track_from_course_m(&position, &anchor, course_rad, ellipsoid),
        ),
        _ => (EAST_RAD, 0.0),
    };
    let intercept_rad = (-0.05 * xtk_m).atan().clamp(-1.0, 1.0);
    let track_rad = course_rad + intercept_rad;
    let moved = destination_point(&position, track_rad, speed_mps * dt_s, ellipsoid);
    GeodeticPosition::new(
        moved.latitude_rad,
        moved.longitude_rad,
        position.altitude_m + climb_mps * dt_s,
    )
}

#[test]
fn hold_entry_sectors_follow_icao_and_aim_for_right_and_left_turns() {
    // Inbound course due east (090°); outbound course due west (270°).
    // Right turns:
    //   arrival 090° (180° from outbound) -> Direct
    //   arrival 300° (30° CW of 270°)    -> Parallel
    //   arrival 020° (110° CW of 270°)   -> Parallel (inclusive boundary)
    //   arrival 021° (111° CW of 270°)   -> Direct
    //   arrival 199° (289° CW of 270°)   -> Direct
    //   arrival 200° (290° CW of 270°)   -> Teardrop (inclusive boundary)
    //   arrival 270° (0° from outbound)  -> Teardrop
    let check_right = |deg: f64| hold_entry(EAST_RAD, TurnDirection::Right, deg.to_radians());
    assert_eq!(check_right(90.0), HoldEntry::Direct);
    assert_eq!(check_right(300.0), HoldEntry::Parallel);
    assert_eq!(check_right(20.0), HoldEntry::Parallel);
    assert_eq!(check_right(21.0), HoldEntry::Direct);
    assert_eq!(check_right(199.0), HoldEntry::Direct);
    assert_eq!(check_right(200.0), HoldEntry::Teardrop);
    assert_eq!(check_right(270.0), HoldEntry::Teardrop);

    // Left turns mirror about the 090°/270° line:
    //   arrival 240° (30° CCW of 270°)   -> Parallel
    //   arrival 160° (110° CCW of 270°)  -> Parallel
    //   arrival 159°                     -> Direct
    //   arrival 341°                     -> Direct
    //   arrival 340° (290° CCW of 270°)  -> Teardrop
    let check_left = |deg: f64| hold_entry(EAST_RAD, TurnDirection::Left, deg.to_radians());
    assert_eq!(check_left(90.0), HoldEntry::Direct);
    assert_eq!(check_left(240.0), HoldEntry::Parallel);
    assert_eq!(check_left(160.0), HoldEntry::Parallel);
    assert_eq!(check_left(159.0), HoldEntry::Direct);
    assert_eq!(check_left(341.0), HoldEntry::Direct);
    assert_eq!(check_left(340.0), HoldEntry::Teardrop);
    assert_eq!(check_left(270.0), HoldEntry::Teardrop);
}

#[test]
fn nav_lg_012_coincident_hold_fix_after_track_leg_activates_cleanly() {
    let hold = eastbound_hold(
        TurnDirection::Right,
        HoldLegLength::Distance { meters: 4_000.0 },
    );
    let wps = [
        wp("W0", 0.0, 0.0),
        wp("FIX", 0.0, 0.5),
        wp("FIX", 0.0, 0.5).with_path(LegPath::HoldToFix { hold }),
        wp("W2", 0.0, 1.0),
    ];
    let exec = execution(&wps, ExecutionConfig::default());
    assert_eq!(exec.active_index(), 0);
}

#[test]
fn hf_flies_one_closed_loop_circuit_on_the_holding_side_and_sequences_across_resume() {
    let hold = eastbound_hold(
        TurnDirection::Right,
        HoldLegLength::Distance { meters: 3_500.0 },
    );
    let wps = [
        wp("W0", 0.0, 0.0),
        wp("HOLD", 0.0, 0.1).with_path(LegPath::HoldToFix { hold }),
        wp("OUT", 0.0, 0.3),
    ];
    let plan = FlightPlan::new(id("hf"), PlanRole::Mission, Ellipsoid::WGS84, &wps);
    let config = ExecutionConfig::default();
    let mut exec = PlanExecution::new(plan, config).expect("valid HF plan");
    capture_w0(&mut exec);
    assert_eq!(
        exec.hold_status(),
        Some(HoldStatus {
            entry: None,
            stage: HoldStage::ToFix,
            circuits_completed: 0,
            exit_armed: false,
        })
    );

    let speed = 50.0;
    let mut pos_now = pos(0.0, 0.05);
    let mut saw_outbound = false;
    let mut saw_inbound = false;
    let mut max_holding_offset_m = 0.0_f64;
    let mut sequenced = None;

    for _ in 0..800 {
        let state = exec.state();
        let mut resumed = PlanExecution::resume_state(plan, config, state).expect("resume_state");
        let event = resumed.advance(&pos_now, speed);
        let direct_event = exec.advance(&pos_now, speed);
        assert_eq!(event, direct_event);
        assert_eq!(resumed.state(), exec.state());

        if let SequenceEvent::LegAdvanced {
            to_index, reason, ..
        } = event
        {
            sequenced = Some((to_index, reason));
            break;
        }
        if let Some(status) = exec.hold_status() {
            if status.stage == HoldStage::Outbound {
                saw_outbound = true;
                assert_eq!(status.entry, Some(HoldEntry::Direct));
                let offset =
                    holding_side_offset_m(&pos_now, &wps[1].position, hold, exec.ellipsoid());
                max_holding_offset_m = max_holding_offset_m.max(offset);
            } else if status.stage == HoldStage::Inbound {
                saw_inbound = true;
            }
        }
        pos_now = step_guided(&exec, pos_now, speed, 0.0, 1.0);
    }

    assert!(saw_outbound, "HF must fly the outbound leg");
    assert!(saw_inbound, "HF must fly the inbound leg");
    // At 50 m/s and 18° bank, 2 * r ≈ 1569 m on the holding (south/right) side.
    assert!(
        max_holding_offset_m > 1_200.0,
        "outbound must displace onto the holding side: {max_holding_offset_m}"
    );
    assert_eq!(sequenced, Some((2, SequenceReason::HoldCircuitComplete)));
    assert_eq!(exec.hold_status(), None);
}

#[test]
fn hf_parallel_and_teardrop_entries_select_expected_stages() {
    let hold = eastbound_hold(TurnDirection::Right, HoldLegLength::Time { seconds: 60.0 });
    // Parallel arrival: approaching HOLD(0.0, 0.1) from SSE on track ~330°.
    let parallel_wps = [
        wp("S0", -0.05, 0.13),
        wp("HOLD", 0.0, 0.1).with_path(LegPath::HoldToFix { hold }),
        wp("OUT", 0.0, 0.3),
    ];
    let mut par = execution(&parallel_wps, ExecutionConfig::default());
    assert!(matches!(
        par.advance(&pos(-0.05, 0.13), 50.0),
        SequenceEvent::LegAdvanced { to_index: 1, .. }
    ));
    assert_eq!(par.advance(&pos(-0.002, 0.101), 50.0), SequenceEvent::None);
    assert_eq!(par.advance(&pos(0.0, 0.1), 50.0), SequenceEvent::None);
    assert_eq!(
        par.hold_status().expect("hold status").stage,
        HoldStage::EntryOutbound {
            entry: HoldEntry::Parallel
        }
    );

    // Teardrop arrival: approaching HOLD(0.0, 0.1) from ENE on track ~240°.
    let teardrop_wps = [
        wp("E0", 0.03, 0.15),
        wp("HOLD", 0.0, 0.1).with_path(LegPath::HoldToFix { hold }),
        wp("OUT", 0.0, 0.3),
    ];
    let mut tear = execution(&teardrop_wps, ExecutionConfig::default());
    assert!(matches!(
        tear.advance(&pos(0.03, 0.15), 50.0),
        SequenceEvent::LegAdvanced { to_index: 1, .. }
    ));
    assert_eq!(tear.advance(&pos(0.001, 0.102), 50.0), SequenceEvent::None);
    assert_eq!(tear.advance(&pos(0.0, 0.1), 50.0), SequenceEvent::None);
    assert_eq!(
        tear.hold_status().expect("hold status").stage,
        HoldStage::EntryOutbound {
            entry: HoldEntry::Teardrop
        }
    );
}

#[test]
fn ha_holds_until_target_altitude_is_reached_and_exits_immediately_when_already_above() {
    let hold = eastbound_hold(
        TurnDirection::Left,
        HoldLegLength::Distance { meters: 2_500.0 },
    );
    let wps = [
        wp("W0", 0.0, 0.0),
        wp("HAFIX", 0.0, 0.1)
            .with_altitude(AltitudeConstraint::AtOrAbove(600.0))
            .with_path(LegPath::HoldToAltitude { hold }),
        wp("OUT", 0.0, 0.3),
    ];
    let mut exec = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut exec);

    // Arrive at the fix at 200 m (< 600 m), fly outbound and inbound back
    // to the fix still at 450 m: must NOT sequence yet.
    assert_eq!(
        exec.advance(&pos_alt(0.0, 0.1, 200.0), 50.0),
        SequenceEvent::None
    );
    assert_eq!(exec.hold_status().expect("hold").stage, HoldStage::Outbound);
    assert_eq!(
        exec.advance(&pos_alt(0.0, 0.1 - 2_600.0 / M_PER_DEG, 350.0), 50.0),
        SequenceEvent::None
    );
    assert_eq!(exec.hold_status().expect("hold").stage, HoldStage::Inbound);
    assert_eq!(
        exec.advance(&pos_alt(0.0, 0.1, 450.0), 50.0),
        SequenceEvent::None
    );
    assert_eq!(exec.hold_status().expect("hold").circuits_completed, 1);

    // Reaching 600 m sequences immediately with AltitudeReached.
    assert_eq!(
        exec.advance(&pos_alt(0.0, 0.09, 605.0), 50.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::AltitudeReached,
        }
    );

    // Arriving at HAFIX already >= 600 m sequences at fix crossing without
    // entering the racetrack.
    let mut already_high = execution(&wps, ExecutionConfig::default());
    capture_w0(&mut already_high);
    assert_eq!(
        already_high.advance(&pos_alt(0.0, 0.1, 650.0), 50.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::AltitudeReached,
        }
    );
}

#[test]
fn hm_repeats_circuits_until_exit_hold_is_armed_then_exits_at_the_fix() {
    let hold = eastbound_hold(
        TurnDirection::Right,
        HoldLegLength::Distance { meters: 2_500.0 },
    );
    let wps = [
        wp("W0", 0.0, 0.0),
        wp("HMFIX", 0.0, 0.1).with_path(LegPath::HoldManual { hold }),
        wp("OUT", 0.0, 0.3),
    ];
    let mut exec = execution(&wps, ExecutionConfig::default());
    assert!(!exec.exit_hold(), "W0 is not an HM leg");
    capture_w0(&mut exec);

    // Complete two full circuits without arming exit: stays in HM.
    assert_eq!(exec.advance(&pos(0.0, 0.1), 50.0), SequenceEvent::None);
    for expected_circuits in 1..=2 {
        assert_eq!(
            exec.advance(&pos(0.0, 0.1 - 2_600.0 / M_PER_DEG), 50.0),
            SequenceEvent::None
        );
        assert_eq!(exec.advance(&pos(0.0, 0.1), 50.0), SequenceEvent::None);
        assert_eq!(
            exec.hold_status().expect("hold").circuits_completed,
            expected_circuits
        );
    }

    // Arm manual exit on the third outbound leg: still flies back to the
    // hold fix before sequencing.
    assert!(exec.exit_hold());
    assert!(exec.hold_status().expect("hold").exit_armed);
    assert_eq!(
        exec.advance(&pos(0.0, 0.1 - 2_600.0 / M_PER_DEG), 50.0),
        SequenceEvent::None
    );
    assert_eq!(
        exec.advance(&pos(0.0, 0.1), 50.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::HoldExited,
        }
    );
}

#[test]
fn ca_skips_minimum_leg_length_and_sequences_at_target_altitude() {
    let wps = [
        wp("RWY", 0.0, 0.0).with_path(LegPath::InitialFix),
        wp("CA400", 0.0, 0.0)
            .with_altitude(AltitudeConstraint::AtOrAbove(400.0))
            .with_path(LegPath::CourseToAltitude {
                course_rad: EAST_RAD,
                reference: CourseReference::True,
            }),
        wp("FIX", 0.0, 0.2).with_path(LegPath::DirectToFix),
    ];
    let mut exec = execution(&wps, ExecutionConfig::default());
    assert!(matches!(
        exec.advance(&pos_alt(0.0, 0.0, 50.0), 55.0),
        SequenceEvent::LegAdvanced { to_index: 1, .. }
    ));
    assert_eq!(
        exec.active_leg().expect("CA leg").lateral_reference(),
        LateralReference::course(EAST_RAD)
    );

    // Below 400 m: stays on CA even kilometers downrange.
    assert_eq!(
        exec.advance(&pos_alt(0.0, 0.03, 390.0), 55.0),
        SequenceEvent::None
    );
    // Reaching 400 m sequences with AltitudeReached onto the DF leg.
    assert_eq!(
        exec.advance(&pos_alt(0.0, 0.04, 400.0), 55.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::AltitudeReached,
        }
    );
}
