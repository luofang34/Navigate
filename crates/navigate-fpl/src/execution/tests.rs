#![allow(clippy::expect_used, clippy::panic)]

use navigate_contract::PlanRole;

use super::*;

/// Meters per degree of arc on the mean-radius sphere the geodesy layer
/// uses; exact along the equator, where these fixtures live.
const M_PER_DEG: f64 = 111_194.926;

fn pos(lat_deg: f64, lon_deg: f64) -> GeodeticPosition {
    GeodeticPosition::new(lat_deg.to_radians(), lon_deg.to_radians(), 0.0)
}

fn wp(ident: &str, lat_deg: f64, lon_deg: f64) -> Waypoint {
    Waypoint::new(ident.to_owned(), pos(lat_deg, lon_deg))
}

fn execution(waypoints: Vec<Waypoint>, config: ExecutionConfig) -> PlanExecution {
    let plan = FlightPlan::new("mission".into(), PlanRole::Mission, waypoints);
    PlanExecution::new(plan, config).expect("valid plan")
}

/// A dogleg with a 90° turn at W1, legs long enough that the 60 m/s DTA
/// (1129.8 m) sits far outside the capture radius.
fn dogleg(turn_fix: Waypoint) -> PlanExecution {
    execution(
        vec![wp("W0", 0.0, 0.0), turn_fix, wp("W2", 0.5, 0.5)],
        ExecutionConfig::default(),
    )
}

/// A position on the inbound equatorial track, `meters` short of W1 at
/// (0°, 0.5°E).
fn short_of_w1(meters: f64) -> GeodeticPosition {
    pos(0.0, 0.5 - meters / M_PER_DEG)
}

fn capture_w0(exec: &mut PlanExecution) {
    assert!(matches!(
        exec.advance(&pos(0.0, 0.0), 60.0),
        SequenceEvent::LegAdvanced { to_index: 1, .. }
    ));
}

#[test]
fn nav_tt_003_a_fly_by_fix_sequences_at_the_anticipation_distance() {
    let mut exec = dogleg(wp("W1", 0.0, 0.5));
    capture_w0(&mut exec);

    // DTA at 60 m/s under the 18° default is 1129.8 m for a 90° turn.
    assert_eq!(
        exec.advance(&short_of_w1(1140.0), 60.0),
        SequenceEvent::None
    );
    assert_eq!(
        exec.advance(&short_of_w1(1120.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::Anticipated,
        }
    );
}

#[test]
fn nav_tt_004_a_fly_over_fix_waits_for_the_capture_radius() {
    let mut exec = dogleg(wp("W1", 0.0, 0.5).with_turn(TurnType::FlyOver));
    capture_w0(&mut exec);

    assert_eq!(
        exec.advance(&short_of_w1(1120.0), 60.0),
        SequenceEvent::None
    );
    assert_eq!(exec.advance(&short_of_w1(110.0), 60.0), SequenceEvent::None);
    assert_eq!(
        exec.advance(&short_of_w1(99.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyOver,
            reason: SequenceReason::Overflown,
        }
    );
}

#[test]
fn nav_tt_003_collinear_legs_degrade_to_the_capture_radius() {
    let mut exec = execution(
        vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5), wp("W2", 0.0, 1.0)],
        ExecutionConfig::default(),
    );
    capture_w0(&mut exec);

    assert_eq!(
        exec.advance(&short_of_w1(1120.0), 60.0),
        SequenceEvent::None
    );
    assert_eq!(
        exec.advance(&short_of_w1(99.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::Overflown,
        }
    );
}

#[test]
fn nav_tt_003_anticipation_is_capped_by_the_inbound_leg_length() {
    // Inbound leg 400 m, 90° turn: DTA (1129.8 m) caps to 400 m.
    let w1_lon = 400.0 / M_PER_DEG;
    let mut exec = execution(
        vec![
            wp("W0", 0.0, 0.0),
            wp("W1", 0.0, w1_lon),
            wp("W2", 0.5, w1_lon),
        ],
        ExecutionConfig::default(),
    );
    capture_w0(&mut exec);

    let short_of_short_w1 = |meters: f64| pos(0.0, w1_lon - meters / M_PER_DEG);
    assert_eq!(
        exec.advance(&short_of_short_w1(390.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::Anticipated,
        }
    );
}

#[test]
fn an_unflyable_groundspeed_anticipates_nothing() {
    for groundspeed in [f64::NAN, -10.0, 0.0] {
        let mut exec = dogleg(wp("W1", 0.0, 0.5));
        capture_w0(&mut exec);
        assert_eq!(
            exec.advance(&short_of_w1(1120.0), groundspeed),
            SequenceEvent::None,
            "groundspeed {groundspeed} must not anticipate"
        );
    }
}

/// NAV-HN-002: a plan built with only `Waypoint::new` under the default
/// config sequences exactly as pure capture-radius behavior — the same
/// stations the pre-turn-type sequencer stepped through.
#[test]
fn nav_hn_002_a_legacy_plan_keeps_capture_radius_behavior() {
    let mut exec = execution(
        vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 1.0), wp("W2", 1.0, 1.0)],
        ExecutionConfig::default(),
    );
    // At 2 m/s the DTA is ~1.26 m, far inside the 100 m capture radius.
    assert_eq!(exec.advance(&pos(0.001, 0.0), 2.0), SequenceEvent::None);
    assert!(matches!(
        exec.advance(&pos(0.0008, 0.0), 2.0),
        SequenceEvent::LegAdvanced {
            to_index: 1,
            reason: SequenceReason::Overflown,
            ..
        }
    ));
    assert_eq!(exec.advance(&pos(0.0, 0.5), 2.0), SequenceEvent::None);
    assert!(matches!(
        exec.advance(&pos(0.0, 1.0), 2.0),
        SequenceEvent::LegAdvanced { to_index: 2, .. }
    ));
    assert_eq!(
        exec.advance(&pos(1.0, 1.0), 2.0),
        SequenceEvent::PlanComplete
    );
    assert!(exec.is_complete());
    assert_eq!(exec.advance(&pos(1.0, 1.0), 2.0), SequenceEvent::None);
}

#[test]
fn nav_vc_003_a_speed_constraint_below_the_floor_is_refused() {
    let plan = FlightPlan::new(
        "slow".into(),
        PlanRole::Mission,
        vec![wp("W0", 0.0, 0.0).with_max_speed(0.1)],
    );
    let refused = PlanExecution::new(plan, ExecutionConfig::default());
    assert!(matches!(
        refused,
        Err(PlanActivationError::SpeedBelowFloor { max_speed_mps, .. })
            if max_speed_mps == 0.1
    ));
}

#[test]
fn an_unflyable_config_is_refused_by_field() {
    let plan = || FlightPlan::new("p".into(), PlanRole::Mission, vec![wp("W0", 0.0, 0.0)]);
    let zero_bank = ExecutionConfig {
        bank_limit_rad: 0.0,
        ..ExecutionConfig::default()
    };
    assert!(matches!(
        PlanExecution::new(plan(), zero_bank),
        Err(PlanActivationError::InvalidConfig {
            field: "bank_limit_rad",
            ..
        })
    ));
    let bad_floor = ExecutionConfig {
        min_approach_speed_mps: f64::NAN,
        ..ExecutionConfig::default()
    };
    assert!(matches!(
        PlanExecution::new(plan(), bad_floor),
        Err(PlanActivationError::InvalidConfig {
            field: "min_approach_speed_mps",
            ..
        })
    ));
}

#[test]
fn a_position_far_from_everything_advances_nothing() {
    let mut exec = dogleg(wp("W1", 0.0, 0.5));
    for _ in 0..3 {
        assert_eq!(exec.advance(&pos(45.0, -120.0), 60.0), SequenceEvent::None);
    }
    assert_eq!(exec.active_index(), 0);
    assert!(!exec.is_complete());
}
