//! Velocity-derivation refusals, and the proof that one solution draws
//! the same refusal from both derivations.

use navigate_contract::{
    ClockDomainId, DurationNanos, GeodeticPosition, LateralReference, MonotonicNanos,
    SolutionQuality, Waypoint,
};

use crate::config::{GuidanceConfig, VelocityGuidanceConfig};
use crate::derive::guide;
use crate::refusal::GuidanceRefusal;
use crate::scenario::{CLOCK, deg, equator_leg, now, solution};

use super::super::guide_velocity;

#[test]
fn unusable_quality_is_refused() {
    let (from, to) = equator_leg();
    let own = solution(SolutionQuality::Unusable, deg(0.0, 0.5, 0.0), now());
    let refusal = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &VelocityGuidanceConfig::default(),
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
fn a_stale_solution_is_refused() {
    let (from, to) = equator_leg();
    let solved_at = MonotonicNanos::from_nanos(1_000);
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), solved_at);
    let config = VelocityGuidanceConfig::default();
    let bound = config.admission.max_solution_age;
    let at_bound = MonotonicNanos::from_nanos(solved_at.as_nanos() + bound.as_nanos());
    assert!(
        guide_velocity(
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
    let refusal = guide_velocity(
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
            age: DurationNanos::from_nanos(bound.as_nanos() + 1),
            bound,
        }
    );
}

#[test]
fn clock_inversion_is_refused() {
    let (from, to) = equator_leg();
    let solved_at = MonotonicNanos::from_nanos(10);
    let earlier = MonotonicNanos::from_nanos(5);
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), solved_at);
    let refusal = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        earlier,
        CLOCK,
        &VelocityGuidanceConfig::default(),
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
    let earlier = MonotonicNanos::from_nanos(0);
    let refusal = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        earlier,
        foreign,
        &VelocityGuidanceConfig::default(),
    )
    .expect_err("refuses");
    assert_eq!(
        refusal,
        GuidanceRefusal::ClockDomainMismatch {
            expected: CLOCK,
            got: foreign,
        }
    );
}

#[test]
fn an_implausible_target_is_refused_by_name() {
    let (from, _) = equator_leg();
    let bad = Waypoint::new("BAD".into(), GeodeticPosition::new(f64::NAN, 0.0, 0.0));
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now());
    let refusal = guide_velocity(
        &own,
        LateralReference::track(from),
        &bad,
        now(),
        CLOCK,
        &VelocityGuidanceConfig::default(),
    )
    .expect_err("refuses");
    assert_eq!(
        refusal,
        GuidanceRefusal::ImplausibleTarget {
            ident: "BAD".into(),
        }
    );
}

#[test]
fn a_leg_with_coincident_endpoints_is_refused_as_implausible() {
    let anchor = deg(10.0, 20.0, 0.0);
    let to = Waypoint::new("ZERO".into(), anchor);
    let own = solution(SolutionQuality::Good, deg(10.5, 20.5, 0.0), now());
    let refusal = guide_velocity(
        &own,
        LateralReference::track(anchor),
        &to,
        now(),
        CLOCK,
        &VelocityGuidanceConfig::default(),
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
fn both_derivations_judge_one_solution_identically() {
    let (from, to) = equator_leg();
    let solved_at = MonotonicNanos::from_nanos(1_000);
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), solved_at);
    let admission = GuidanceConfig::default();
    let config = VelocityGuidanceConfig::default();
    assert_eq!(config.admission, admission, "one admission vocabulary");
    let past_bound = MonotonicNanos::from_nanos(
        solved_at.as_nanos() + admission.max_solution_age.as_nanos() + 1,
    );
    let deviation_refusal = guide(
        &own,
        LateralReference::track(from),
        &to,
        past_bound,
        CLOCK,
        &admission,
    )
    .expect_err("refuses");
    let velocity_refusal = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        past_bound,
        CLOCK,
        &config,
    )
    .expect_err("refuses");
    assert_eq!(
        deviation_refusal, velocity_refusal,
        "the shared admission helper cannot let the two derivations drift"
    );
}
