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

/// A dogleg with a 90° turn at W1, legs long enough (~55 km) that the
/// 60 m/s DTA (1129.8 m) sits far inside the half-leg cap.
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
fn nav_tt_005_a_late_sample_on_an_anticipated_fix_still_names_the_rule() {
    // The fix's anticipation (1129.8 m) exceeds the capture radius, so
    // the transition IS anticipated even when the deciding sample lands
    // inside the radius — the reason names the rule, not the sample.
    let mut exec = dogleg(wp("W1", 0.0, 0.5));
    capture_w0(&mut exec);
    assert_eq!(
        exec.advance(&short_of_w1(90.0), 60.0),
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
fn nav_tt_003_anticipation_is_capped_at_half_the_shorter_adjoining_leg() {
    // Inbound leg 400 m, 90° turn onto a long leg: DTA (1129.8 m) caps
    // to 200 m — the leg keeps a flyable middle and never sequences at
    // the instant it begins.
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
    // At the leg start (400 m out) and outside the cap: no sequencing.
    assert_eq!(
        exec.advance(&short_of_short_w1(399.0), 60.0),
        SequenceEvent::None
    );
    assert_eq!(
        exec.advance(&short_of_short_w1(201.0), 60.0),
        SequenceEvent::None
    );
    assert_eq!(
        exec.advance(&short_of_short_w1(199.0), 60.0),
        SequenceEvent::LegAdvanced {
            to_index: 2,
            turn: TurnType::FlyBy,
            reason: SequenceReason::Anticipated,
        }
    );
}

#[test]
fn a_parked_vehicle_never_walks_through_short_legs() {
    // Two 500 m legs with a 90° corner; the vehicle sits at W0. W0
    // captures once, and W1 must NOT sequence while parked 500 m out —
    // the whole-leg skip is the failure this pins against.
    let w1_lon = 500.0 / M_PER_DEG;
    let w2_lat = 500.0 / M_PER_DEG;
    let mut exec = execution(
        vec![
            wp("W0", 0.0, 0.0),
            wp("W1", 0.0, w1_lon),
            wp("W2", w2_lat, w1_lon),
        ],
        ExecutionConfig::default(),
    );
    let parked = pos(0.0, 0.0);
    assert!(matches!(
        exec.advance(&parked, 60.0),
        SequenceEvent::LegAdvanced { to_index: 1, .. }
    ));
    for _ in 0..3 {
        assert_eq!(exec.advance(&parked, 60.0), SequenceEvent::None);
    }
    assert_eq!(exec.active_index(), 1, "W1 is still ahead");
}

#[test]
fn a_direct_to_leg_anticipates_nothing_even_before_a_reversal() {
    // Fly-by first waypoint with a reversal onward: the direct-to leg's
    // inbound geometry is the live position, so anticipation would be
    // self-referential — it degrades to the capture radius, and no fix
    // captures from tens of kilometers away.
    let mut exec = execution(
        vec![wp("W0", 0.0, 0.5), wp("W1", 0.0, 0.0)],
        ExecutionConfig::default(),
    );
    assert_eq!(
        exec.advance(&pos(0.0, 0.05), 60.0),
        SequenceEvent::None,
        "50 km out must not capture"
    );
    assert_eq!(exec.active_index(), 0);
}

#[test]
fn nav_tt_003_a_track_change_beyond_the_limit_falls_back_to_capture() {
    // A 180° reversal at W1: tan(Δ/2) blows up, so anticipation refuses
    // and the fix sequences at the capture radius like a fly-over.
    let mut exec = execution(
        vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5), wp("W2", 0.0, 0.0)],
        ExecutionConfig::default(),
    );
    capture_w0(&mut exec);
    assert_eq!(exec.advance(&short_of_w1(500.0), 60.0), SequenceEvent::None);
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
fn nav_tt_001_the_terminal_waypoint_ignores_its_turn_type() {
    for turn in [TurnType::FlyBy, TurnType::FlyOver] {
        let mut exec = execution(
            vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5).with_turn(turn)],
            ExecutionConfig::default(),
        );
        capture_w0(&mut exec);
        assert_eq!(
            exec.advance(&short_of_w1(1120.0), 60.0),
            SequenceEvent::None,
            "no onward course exists to anticipate"
        );
        assert_eq!(
            exec.advance(&short_of_w1(99.0), 60.0),
            SequenceEvent::PlanComplete {
                turn,
                reason: SequenceReason::Overflown,
            }
        );
    }
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

#[test]
fn a_non_finite_position_captures_nothing() {
    let mut exec = dogleg(wp("W1", 0.0, 0.5));
    let bad = GeodeticPosition::new(f64::NAN, 0.0, 0.0);
    assert_eq!(exec.advance(&bad, 60.0), SequenceEvent::None);
    assert_eq!(exec.active_index(), 0);
}

/// NAV-HN-002: a plan built with only `Waypoint::new` sequences at pure
/// capture-radius stations whenever the DTA stays inside the capture
/// radius (here 1.26 m at 2 m/s against 100 m).
#[test]
fn nav_hn_002_a_legacy_plan_keeps_capture_radius_behavior_at_low_speed() {
    let mut exec = execution(
        vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 1.0), wp("W2", 1.0, 1.0)],
        ExecutionConfig::default(),
    );
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
    assert!(matches!(
        exec.advance(&pos(1.0, 1.0), 2.0),
        SequenceEvent::PlanComplete { .. }
    ));
    assert!(exec.is_complete());
    assert_eq!(exec.advance(&pos(1.0, 1.0), 2.0), SequenceEvent::None);
}

/// NAV-HN-002's honest boundary: fly-by is the DEFAULT, so the same
/// vocabulary-free plan anticipates once the groundspeed makes the DTA
/// exceed the capture radius — intentional, not a regression.
#[test]
fn nav_hn_002_a_legacy_plan_anticipates_at_speed_by_design() {
    let mut exec = execution(
        vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.5), wp("W2", 0.5, 0.5)],
        ExecutionConfig::default(),
    );
    capture_w0(&mut exec);
    assert!(matches!(
        exec.advance(&short_of_w1(1120.0), 60.0),
        SequenceEvent::LegAdvanced {
            reason: SequenceReason::Anticipated,
            ..
        }
    ));
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

#[test]
fn nav_tt_003_a_leg_no_longer_than_the_capture_radius_is_refused_at_activation() {
    // A 50 m fixed leg under the 100 m capture radius would sequence
    // while the vehicle is still at the previous fix.
    let w1_lon = 50.0 / M_PER_DEG;
    let plan = FlightPlan::new(
        "short".into(),
        PlanRole::Mission,
        vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, w1_lon)],
    );
    assert!(matches!(
        PlanExecution::new(plan, ExecutionConfig::default()),
        Err(PlanActivationError::CaptureRadiusExceedsLeg { ident, .. }) if ident == "W1"
    ));
}
