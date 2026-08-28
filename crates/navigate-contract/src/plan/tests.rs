#![allow(clippy::expect_used, clippy::panic)]
//! Plan-vocabulary tests. Every constraint form is pinned against the
//! requirement it implements (`docs/procedure-requirements.md`).

use super::{
    AltitudeConstraint, ConstraintField, CourseReference, FlightPlan, LegPath, PlanRole,
    PlanValidationError, TurnType, Waypoint,
};
use crate::kinematics::GeodeticPosition;

/// Every reserved terminator, so a test walks the whole reserved set
/// rather than a sample of it.
const RESERVED: [LegPath; 8] = [
    LegPath::RadiusToFix,
    LegPath::HoldToAltitude,
    LegPath::HoldToFix,
    LegPath::HoldManual,
    LegPath::VerticalTakeoff,
    LegPath::Hover,
    LegPath::Transition,
    LegPath::VerticalLanding,
];

fn waypoint(ident: &str) -> Waypoint {
    Waypoint::new(ident.to_owned(), GeodeticPosition::new(0.0, 0.0, 0.0))
}

fn plan_of(waypoints: Vec<Waypoint>) -> FlightPlan {
    FlightPlan::new("p".into(), PlanRole::Mission, waypoints)
}

#[test]
fn empty_plan_is_invalid() {
    let plan = FlightPlan::new("p1".into(), PlanRole::Mission, Vec::new());
    assert!(matches!(
        plan.validate(),
        Err(PlanValidationError::Empty { .. })
    ));
}

#[test]
fn implausible_waypoint_is_named() {
    let bad = Waypoint::new("W1".into(), GeodeticPosition::new(f64::NAN, 0.0, 0.0));
    let plan = FlightPlan::new("p2".into(), PlanRole::Mission, vec![bad]);
    assert!(matches!(
        plan.validate(),
        Err(PlanValidationError::ImplausiblePosition { index: 0, .. })
    ));
}

#[test]
fn nav_tt_001_waypoints_default_to_fly_by_and_carry_an_explicit_type() {
    let default = waypoint("W0");
    assert_eq!(default.turn, TurnType::FlyBy, "the RNAV default");
    assert_eq!(TurnType::default(), TurnType::FlyBy);
    let over = waypoint("W1").with_turn(TurnType::FlyOver);
    assert_eq!(over.turn, TurnType::FlyOver);
    // The builder replaces the type rather than accumulating one.
    assert_eq!(over.with_turn(TurnType::FlyBy).turn, TurnType::FlyBy);
}

#[test]
fn nav_hn_002_a_waypoint_built_without_builders_declares_no_constraint() {
    let plain = waypoint("W0");
    assert_eq!(plain.altitude, None);
    assert_eq!(plain.max_speed_mps, None);
    assert_eq!(plain.gradient, None);
    assert_eq!(plain.turn, TurnType::FlyBy);
    assert!(plan_of(vec![plain]).validate().is_ok());
}

#[test]
fn nav_vc_001_a_window_brackets_a_band_and_an_inverted_one_is_refused() {
    let ok = waypoint("W0").with_altitude(AltitudeConstraint::Window {
        lower_m: 300.0,
        upper_m: 600.0,
    });
    assert!(plan_of(vec![ok]).validate().is_ok());

    for (lower_m, upper_m) in [(600.0, 300.0), (400.0, 400.0)] {
        let inverted =
            waypoint("W1").with_altitude(AltitudeConstraint::Window { lower_m, upper_m });
        assert!(
            matches!(
                plan_of(vec![inverted]).validate(),
                Err(PlanValidationError::InvertedAltitudeWindow { index: 0, .. })
            ),
            "lower {lower_m} upper {upper_m} must not validate"
        );
    }
}

#[test]
fn nav_vc_001_a_non_finite_window_bound_is_refused_before_the_ordering_test() {
    let bad = waypoint("W0").with_altitude(AltitudeConstraint::Window {
        lower_m: f64::NAN,
        upper_m: 600.0,
    });
    assert!(matches!(
        plan_of(vec![bad]).validate(),
        Err(PlanValidationError::NonFiniteConstraint {
            field: ConstraintField::Altitude,
            ..
        })
    ));
}

#[test]
fn nav_vc_003_a_non_finite_speed_or_gradient_is_refused_by_field() {
    let speed = waypoint("W0").with_max_speed(f64::INFINITY);
    assert!(matches!(
        plan_of(vec![speed]).validate(),
        Err(PlanValidationError::NonFiniteConstraint {
            field: ConstraintField::MaxSpeed,
            ..
        })
    ));
    let gradient = waypoint("W1").with_gradient(f64::NAN);
    assert!(matches!(
        plan_of(vec![gradient]).validate(),
        Err(PlanValidationError::NonFiniteConstraint {
            field: ConstraintField::Gradient,
            ..
        })
    ));
}

#[test]
fn nav_vc_001_a_non_finite_one_sided_altitude_is_refused() {
    for constraint in [
        AltitudeConstraint::At(f64::NAN),
        AltitudeConstraint::AtOrAbove(f64::INFINITY),
        AltitudeConstraint::AtOrBelow(f64::NEG_INFINITY),
    ] {
        let bad = waypoint("W0").with_altitude(constraint);
        assert!(
            matches!(
                plan_of(vec![bad]).validate(),
                Err(PlanValidationError::NonFiniteConstraint {
                    field: ConstraintField::Altitude,
                    ..
                })
            ),
            "{constraint:?} must not validate"
        );
    }
}

#[test]
fn nav_vc_004_constraints_are_exactly_what_the_source_declared() {
    let waypoint = waypoint("W0")
        .with_max_speed(35.0)
        .with_gradient(0.052)
        .with_turn(TurnType::FlyOver)
        .with_altitude(AltitudeConstraint::AtOrAbove(1_200.0));
    let plan = plan_of(vec![waypoint]);
    plan.validate().expect("declared constraints validate");
    let stored = plan.waypoints.first().expect("one waypoint");
    assert_eq!(stored.max_speed_mps, Some(35.0));
    assert_eq!(stored.gradient, Some(0.052));
    assert_eq!(stored.turn, TurnType::FlyOver);
    assert_eq!(
        stored.altitude,
        Some(AltitudeConstraint::AtOrAbove(1_200.0))
    );
}

#[test]
fn a_defect_names_the_waypoint_that_carries_it() {
    let plan = plan_of(vec![
        waypoint("GOOD"),
        waypoint("BAD").with_gradient(f64::NAN),
    ]);
    let refusal = plan
        .validate()
        .expect_err("the second waypoint is defective");
    assert!(matches!(
        &refusal,
        PlanValidationError::NonFiniteConstraint { index: 1, ident, .. } if ident == "BAD"
    ));
    assert!(
        refusal.to_string().contains("gradient"),
        "the message names the field: {refusal}"
    );
}

#[test]
fn nav_vc_003_a_non_positive_speed_constraint_is_refused() {
    for speed in [0.0, -5.0] {
        let plan = FlightPlan::new(
            "p".into(),
            PlanRole::Mission,
            vec![
                Waypoint::new("W0".into(), GeodeticPosition::new(0.0, 0.0, 0.0))
                    .with_max_speed(speed),
            ],
        );
        assert!(
            matches!(
                plan.validate(),
                Err(PlanValidationError::NonPositiveSpeedConstraint { max_speed_mps, .. })
                    if max_speed_mps == speed
            ),
            "speed {speed} must be refused"
        );
    }
}

#[test]
fn nav_lg_001_a_waypoint_declares_no_path_and_carries_an_explicit_one() {
    assert_eq!(waypoint("W0").path, None, "the source declared none");
    let declared = waypoint("W1").with_path(LegPath::DirectToFix);
    assert_eq!(declared.path, Some(LegPath::DirectToFix));
    // The builder replaces the terminator rather than accumulating one.
    assert_eq!(
        declared.with_path(LegPath::InitialFix).path,
        Some(LegPath::InitialFix)
    );
}

#[test]
fn nav_lg_001_every_terminator_names_itself_distinctly() {
    let flown = [
        LegPath::InitialFix,
        LegPath::TrackToFix,
        LegPath::DirectToFix,
        LegPath::CourseToFix {
            course_rad: 0.0,
            reference: CourseReference::True,
        },
    ];
    let codes: Vec<&str> = flown
        .iter()
        .chain(RESERVED.iter())
        .map(LegPath::code)
        .collect();
    for (position, code) in codes.iter().enumerate() {
        assert!(!code.is_empty(), "terminator {position} has no name");
        assert_eq!(
            codes.iter().filter(|other| *other == code).count(),
            1,
            "{code} names more than one terminator"
        );
    }
    assert_eq!(LegPath::TrackToFix.to_string(), "TF");
}

#[test]
fn nav_lg_004_a_track_leg_at_the_first_waypoint_has_no_fix_to_run_from() {
    let plan = plan_of(vec![waypoint("W0").with_path(LegPath::TrackToFix)]);
    assert!(matches!(
        plan.validate(),
        Err(PlanValidationError::TrackWithoutUpstreamFix { ident, .. }) if ident == "W0"
    ));
}

#[test]
fn nav_lg_004_an_initial_fix_after_the_first_waypoint_is_refused() {
    let plan = plan_of(vec![
        waypoint("W0"),
        waypoint("W1").with_path(LegPath::InitialFix),
    ]);
    assert!(matches!(
        plan.validate(),
        Err(PlanValidationError::InitialFixNotFirst { index: 1, ident, .. }) if ident == "W1"
    ));
    // The same terminator at the first waypoint is exactly its place.
    assert!(
        plan_of(vec![waypoint("W0").with_path(LegPath::InitialFix)])
            .validate()
            .is_ok()
    );
}

#[test]
fn nav_lg_004_a_course_outside_the_full_turn_is_refused() {
    for course_rad in [f64::NAN, f64::INFINITY, -0.1, core::f64::consts::TAU, 10.0] {
        let plan = plan_of(vec![
            waypoint("W0"),
            waypoint("W1").with_path(LegPath::CourseToFix {
                course_rad,
                reference: CourseReference::True,
            }),
        ]);
        assert!(
            matches!(
                plan.validate(),
                Err(PlanValidationError::CourseOutOfRange { index: 1, ident, .. }) if ident == "W1"
            ),
            "course {course_rad} must not validate"
        );
    }
    // The open upper bound admits everything below a full turn.
    let plan = plan_of(vec![
        waypoint("W0"),
        waypoint("W1").with_path(LegPath::CourseToFix {
            course_rad: core::f64::consts::TAU - 1e-9,
            reference: CourseReference::True,
        }),
    ]);
    assert!(plan.validate().is_ok());
}

/// NAV-LG-005: what is implemented is the executor's property, not this
/// type's. A plan that carries a reserved leg is a VALID plan; only
/// activation refuses it.
#[test]
fn nav_lg_005_a_reserved_terminator_is_a_valid_plan_here() {
    for path in RESERVED {
        let plan = plan_of(vec![waypoint("W0").with_path(path)]);
        assert!(
            plan.validate().is_ok(),
            "{path} must validate as an exchange shape"
        );
    }
}

/// NAV-LG-013: a structural refusal names the plan, the index, and the
/// identifier, so an operator reads which leg blocked the plan.
#[test]
fn nav_lg_013_a_leg_refusal_names_the_leg_that_carries_it() {
    let plan = plan_of(vec![
        waypoint("GOOD"),
        waypoint("BAD").with_path(LegPath::InitialFix),
    ]);
    let refusal = plan
        .validate()
        .expect_err("the second waypoint is misplaced");
    let message = refusal.to_string();
    for fragment in ["p", "1", "BAD", "IF"] {
        assert!(
            message.contains(fragment),
            "the message must name {fragment}: {message}"
        );
    }
}

#[test]
fn nav_vc_002_a_zero_gradient_is_refused_absence_expresses_no_limit() {
    let plan = FlightPlan::new(
        "p".into(),
        PlanRole::Mission,
        vec![Waypoint::new("W0".into(), GeodeticPosition::new(0.0, 0.0, 0.0)).with_gradient(0.0)],
    );
    assert!(matches!(
        plan.validate(),
        Err(PlanValidationError::ZeroGradient { .. })
    ));
}
