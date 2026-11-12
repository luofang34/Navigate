//! Validation tests for hold geometries, course-to-altitude legs, and
//! mandatory turn directions.

use super::*;

#[test]
fn nav_lg_015_a_hold_carries_its_racetrack() {
    let path = LegPath::HoldManual { hold: HOLD };
    let fix = waypoint("FIX").with_path(path);
    assert_eq!(fix.path.and_then(|p| p.hold()), Some(HOLD));
    assert_eq!(path.code(), "HM");
    assert!(plan_of(&[waypoint("W0"), fix]).validate().is_ok());
    let distance = HoldGeometry::new(
        1.5,
        CourseReference::True,
        TurnDirection::Left,
        HoldLegLength::Distance { meters: 7_408.0 },
    );
    let fix = waypoint("FIX")
        .with_path(LegPath::HoldToFix { hold: distance })
        .with_turn_direction(TurnDirection::Left);
    assert!(plan_of(&[fix]).validate().is_ok());
}

#[test]
fn nav_lg_004_a_hold_inbound_course_outside_the_full_turn_is_refused() {
    for inbound_course_rad in [f64::NAN, -0.1, core::f64::consts::TAU] {
        let hold = HoldGeometry {
            inbound_course_rad,
            ..HOLD
        };
        let waypoints = [
            waypoint("W0"),
            waypoint("W1").with_path(LegPath::HoldToFix { hold }),
        ];
        assert!(
            matches!(
                plan_of(&waypoints).validate(),
                Err(PlanValidationError::CourseOutOfRange { index: 1, ident, .. }) if ident == "W1"
            ),
            "inbound course {inbound_course_rad} must not validate"
        );
    }
}

#[test]
fn nav_lg_004_a_hold_leg_that_is_not_positive_is_refused() {
    for leg in [
        HoldLegLength::Time { seconds: 0.0 },
        HoldLegLength::Time { seconds: f64::NAN },
        HoldLegLength::Distance { meters: -1.0 },
        HoldLegLength::Distance {
            meters: f64::INFINITY,
        },
    ] {
        let hold = HoldGeometry { leg, ..HOLD };
        let waypoints = [waypoint("W0").with_path(LegPath::HoldManual { hold })];
        let refusal = plan_of(&waypoints).validate();
        assert!(
            matches!(
                refusal,
                Err(PlanValidationError::HoldLegNotPositive { index: 0, .. })
            ),
            "{leg:?} must not validate: {refusal:?}"
        );
    }
}

#[test]
fn nav_lg_015_a_hold_to_altitude_needs_the_waypoint_altitude() {
    let path = LegPath::HoldToAltitude { hold: HOLD };
    let without = [waypoint("W0").with_path(path)];
    assert!(matches!(
        plan_of(&without).validate(),
        Err(PlanValidationError::HoldToAltitudeWithoutAltitude { index: 0, .. })
    ));
    let with = [waypoint("W0")
        .with_altitude(AltitudeConstraint::AtOrAbove(1_500.0))
        .with_path(path)];
    assert!(plan_of(&with).validate().is_ok());
}

#[test]
fn a_course_to_altitude_needs_valid_course_and_altitude() {
    let without = [waypoint("W0").with_path(LegPath::CourseToAltitude {
        course_rad: 1.0,
        reference: CourseReference::True,
    })];
    assert!(matches!(
        plan_of(&without).validate(),
        Err(PlanValidationError::CourseToAltitudeWithoutAltitude { index: 0, .. })
    ));
    let bad_course = [waypoint("W0")
        .with_altitude(AltitudeConstraint::AtOrAbove(500.0))
        .with_path(LegPath::CourseToAltitude {
            course_rad: -0.2,
            reference: CourseReference::True,
        })];
    assert!(matches!(
        plan_of(&bad_course).validate(),
        Err(PlanValidationError::CourseOutOfRange { index: 0, .. })
    ));
    let valid = [waypoint("W0")
        .with_altitude(AltitudeConstraint::AtOrAbove(500.0))
        .with_path(LegPath::CourseToAltitude {
            course_rad: 1.0,
            reference: CourseReference::True,
        })
        .with_turn_direction(TurnDirection::Right)];
    assert!(plan_of(&valid).validate().is_ok());
}

#[test]
fn conflicting_hold_turn_direction_is_refused() {
    let waypoints = [waypoint("W0")
        .with_path(LegPath::HoldManual { hold: HOLD })
        .with_turn_direction(TurnDirection::Left)];
    assert!(matches!(
        plan_of(&waypoints).validate(),
        Err(PlanValidationError::ConflictingTurnDirection {
            index: 0,
            waypoint_turn: TurnDirection::Left,
            hold_turn: TurnDirection::Right,
            ..
        })
    ));
}
