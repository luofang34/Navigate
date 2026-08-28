#![allow(clippy::expect_used, clippy::panic)]
//! Deviation-tracking derivation tests: the lateral reference each
//! flown leg type defines, the vertical profile, and the fail-closed
//! admission refusals.

use navigate_contract::{
    AltitudeConstraint, ClockDomainId, DurationNanos, GeodeticPosition, GuidanceSetpoint,
    LateralReference, MonotonicNanos, SolutionQuality, Waypoint,
};
use navigate_geodesy::{cross_track_from_course_m, cross_track_m, initial_bearing_rad, wgs84};

use crate::config::GuidanceConfig;
use crate::refusal::GuidanceRefusal;
use crate::scenario::{CLOCK, deg, equator_leg, now, solution};

use super::guide;

fn deviation(setpoint: GuidanceSetpoint) -> (f64, f64, f64) {
    match setpoint {
        GuidanceSetpoint::DeviationTracking {
            lateral_m,
            vertical_m,
            course_rad,
        } => (lateral_m, vertical_m, course_rad),
        other => panic!("expected DeviationTracking, got {other:?}"),
    }
}

#[test]
fn on_track_yields_near_zero_lateral_and_the_track_course() {
    let (from, to) = equator_leg();
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now());
    let command = guide(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect("guides");
    let (lateral_m, vertical_m, course_rad) = deviation(command.setpoint);
    assert!(lateral_m.abs() < 1.0, "on-track lateral: {lateral_m}");
    assert!(vertical_m.abs() < 1e-12, "no constraint: {vertical_m}");
    let track_bearing = initial_bearing_rad(&from, &to.position);
    assert!(
        (course_rad - track_bearing).abs() < 1e-12,
        "course {course_rad} vs track bearing {track_bearing}"
    );
    assert_eq!(command.issued_at, now());
    assert_eq!(command.basis, own.stamp);
}

#[test]
fn right_of_track_is_positive_matching_geodesy() {
    let (from, to) = equator_leg();
    // 500 m right of an eastbound equatorial track is 500 m south.
    let offset_rad = 500.0 / wgs84::MEAN_RADIUS_M;
    let own_pos = GeodeticPosition::new(-offset_rad, 0.5_f64.to_radians(), 0.0);
    let own = solution(SolutionQuality::Good, own_pos, now());
    let command = guide(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect("guides");
    let (lateral_m, _, _) = deviation(command.setpoint);
    assert!(
        lateral_m > 0.0,
        "right of track must be positive: {lateral_m}"
    );
    assert!(
        (lateral_m - 500.0).abs() < 1.0,
        "offset magnitude: {lateral_m}"
    );
    let reference = cross_track_m(&own_pos, &from, &to.position).expect("valid track");
    assert!(
        (lateral_m - reference).abs() < 1e-9,
        "sign convention pinned to geodesy: {lateral_m} vs {reference}"
    );
}

#[test]
fn direct_to_has_zero_lateral_and_bearing_to_waypoint() {
    let own_pos = deg(10.0, 20.0, 0.0);
    let to = Waypoint::new("DCT".into(), deg(11.0, 21.0, 0.0));
    let own = solution(SolutionQuality::Good, own_pos, now());
    let command = guide(
        &own,
        LateralReference::PresentPosition,
        &to,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect("guides");
    let (lateral_m, _, course_rad) = deviation(command.setpoint);
    assert!(lateral_m.abs() < 1e-12, "direct-to lateral: {lateral_m}");
    let bearing = initial_bearing_rad(&own_pos, &to.position);
    assert!(
        (course_rad - bearing).abs() < 1e-12,
        "course {course_rad} vs bearing {bearing}"
    );
}

#[test]
fn vertical_deviation_flows_from_the_constraint() {
    let (from, to) = equator_leg();
    let to = to.with_altitude(AltitudeConstraint::At(100.0));
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 120.0), now());
    let command = guide(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect("guides");
    let (_, vertical_m, _) = deviation(command.setpoint);
    assert!(
        (vertical_m - 20.0).abs() < 1e-9,
        "above At target: {vertical_m}"
    );
}

#[test]
fn unusable_quality_is_refused_by_default() {
    let (from, to) = equator_leg();
    let own = solution(SolutionQuality::Unusable, deg(0.0, 0.5, 0.0), now());
    let refusal = guide(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect_err("refuses");
    assert_eq!(
        refusal,
        GuidanceRefusal::IntegrityBelowFloor {
            quality: SolutionQuality::Unusable,
            floor: SolutionQuality::Degraded,
        }
    );
}

#[test]
fn requiring_good_refuses_degraded() {
    let (from, to) = equator_leg();
    let own = solution(SolutionQuality::Degraded, deg(0.0, 0.5, 0.0), now());
    let tightened = GuidanceConfig::new(
        SolutionQuality::Good,
        GuidanceConfig::default().max_solution_age,
    );
    let refusal = guide(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &tightened,
    )
    .expect_err("refuses");
    assert!(matches!(
        refusal,
        GuidanceRefusal::IntegrityBelowFloor {
            quality: SolutionQuality::Degraded,
            floor: SolutionQuality::Good,
        }
    ));
}

#[test]
fn stale_solution_is_refused_and_the_bound_is_inclusive() {
    let (from, to) = equator_leg();
    let solved_at = MonotonicNanos::from_nanos(1_000);
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), solved_at);
    let config = GuidanceConfig::default();
    let at_bound = MonotonicNanos::from_nanos(1_000 + config.max_solution_age.as_nanos());
    assert!(
        guide(
            &own,
            LateralReference::track(from),
            &to,
            at_bound,
            CLOCK,
            &config
        )
        .is_ok()
    );
    let past_bound = MonotonicNanos::from_nanos(at_bound.as_nanos() + 1);
    let refusal = guide(
        &own,
        LateralReference::track(from),
        &to,
        past_bound,
        CLOCK,
        &config,
    )
    .expect_err("refuses");
    assert_eq!(
        refusal,
        GuidanceRefusal::SolutionStale {
            age: DurationNanos::from_nanos(config.max_solution_age.as_nanos() + 1),
            bound: config.max_solution_age,
        }
    );
}

#[test]
fn clock_inversion_is_refused_not_zeroed() {
    let (from, to) = equator_leg();
    let solved_at = MonotonicNanos::from_nanos(10);
    let earlier = MonotonicNanos::from_nanos(5);
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), solved_at);
    let refusal = guide(
        &own,
        LateralReference::track(from),
        &to,
        earlier,
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect_err("refuses");
    assert_eq!(
        refusal,
        GuidanceRefusal::ClockInversion {
            now: earlier,
            solved_at,
        }
    );
}

#[test]
fn now_read_on_another_clock_domain_is_refused_before_age_arithmetic() {
    let (from, to) = equator_leg();
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now());
    let foreign = ClockDomainId::new(9);
    // `now` precedes `solved_at`, so if age arithmetic ran first this
    // would be ClockInversion; the mismatch refusal proves the domain
    // is judged before any age arithmetic.
    let earlier = MonotonicNanos::from_nanos(0);
    let refusal = guide(
        &own,
        LateralReference::track(from),
        &to,
        earlier,
        foreign,
        &GuidanceConfig::default(),
    )
    .expect_err("refuses");
    assert_eq!(
        refusal,
        GuidanceRefusal::ClockDomainMismatch {
            expected: CLOCK,
            got: foreign,
        }
    );
    // The same solution guides once `now` is read on its domain.
    assert!(
        guide(
            &own,
            LateralReference::track(from),
            &to,
            now(),
            CLOCK,
            &GuidanceConfig::default()
        )
        .is_ok()
    );
}

#[test]
fn a_leg_with_coincident_endpoints_is_refused_as_implausible() {
    let anchor = deg(10.0, 20.0, 0.0);
    let to = Waypoint::new("ZERO".into(), anchor);
    let own = solution(SolutionQuality::Good, deg(10.5, 20.5, 0.0), now());
    let refusal = guide(
        &own,
        LateralReference::track(anchor),
        &to,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect_err("refuses");
    assert_eq!(
        refusal,
        GuidanceRefusal::ImplausibleTarget {
            ident: "ZERO".into(),
        }
    );
}

#[test]
fn nav_lg_009_a_course_leg_reports_its_published_course_not_the_track() {
    // A published course 10° off the track through the same fix: the
    // reference course and the deviation must both follow the course
    // line, or the vehicle would fly a geometry it was not given.
    let (from, to) = equator_leg();
    let published_rad = 80.0_f64.to_radians();
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now());
    let config = GuidanceConfig::default();
    let on_course = guide(
        &own,
        LateralReference::course(published_rad),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let (course_lateral_m, _, course_rad) = deviation(on_course.setpoint);
    assert!(
        (course_rad - published_rad).abs() < 1e-15,
        "course {course_rad} vs published {published_rad}"
    );
    let reference = cross_track_from_course_m(&own.position, &to.position, published_rad);
    assert!(
        (course_lateral_m - reference).abs() < 1e-9,
        "sign convention pinned to geodesy: {course_lateral_m} vs {reference}"
    );
    // The same ownship is exactly on the track through that fix, so
    // the two references disagree by the whole divergence.
    let on_track = guide(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let (track_lateral_m, _, _) = deviation(on_track.setpoint);
    assert!(track_lateral_m.abs() < 1.0, "on track: {track_lateral_m}");
    assert!(
        course_lateral_m < -1_000.0,
        "left of the published course: {course_lateral_m}"
    );
}

#[test]
fn nav_lg_009_right_of_a_published_course_is_positive() {
    let (_, to) = equator_leg();
    // 500 m right of an eastbound course line is 500 m south.
    let offset_rad = 500.0 / wgs84::MEAN_RADIUS_M;
    let own_pos = GeodeticPosition::new(-offset_rad, 0.5_f64.to_radians(), 0.0);
    let own = solution(SolutionQuality::Good, own_pos, now());
    let command = guide(
        &own,
        LateralReference::course(core::f64::consts::FRAC_PI_2),
        &to,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect("guides");
    let (lateral_m, _, _) = deviation(command.setpoint);
    assert!(
        (lateral_m - 500.0).abs() < 1.0,
        "right of the course must be positive 500 m: {lateral_m}"
    );
}

/// NAV-LG-012: the coincident-endpoint guard is replaced for a
/// course leg, not relaxed. A course line needs one position and one
/// direction, so the case a two-point track must refuse cannot
/// arise for it.
#[test]
fn nav_lg_012_a_course_leg_has_no_coincident_endpoint_case_to_refuse() {
    let anchor = deg(10.0, 20.0, 0.0);
    let to = Waypoint::new("ZERO".into(), anchor);
    let own = solution(SolutionQuality::Good, anchor, now());
    let config = GuidanceConfig::default();
    let refusal = guide(
        &own,
        LateralReference::track(anchor),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect_err("a track needs two separate ends");
    assert_eq!(
        refusal,
        GuidanceRefusal::ImplausibleTarget {
            ident: "ZERO".into(),
        }
    );
    let published_rad = 1.0;
    let command = guide(
        &own,
        LateralReference::course(published_rad),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("a course line guides from the fix it runs through");
    let (lateral_m, _, course_rad) = deviation(command.setpoint);
    assert!(lateral_m.abs() < 1e-9, "on the course line: {lateral_m}");
    assert!((course_rad - published_rad).abs() < 1e-15);
}

#[test]
fn implausible_target_is_refused_by_name() {
    let (from, _) = equator_leg();
    let bad = Waypoint::new("BAD".into(), GeodeticPosition::new(f64::NAN, 0.0, 0.0));
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now());
    let refusal = guide(
        &own,
        LateralReference::track(from),
        &bad,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect_err("refuses");
    assert_eq!(
        refusal,
        GuidanceRefusal::ImplausibleTarget {
            ident: "BAD".into(),
        }
    );
}
