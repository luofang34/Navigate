//! The quality floor and the vertical limit are two independent
//! admission limits, and each refusal names its own limit.

use navigate_contract::{
    AltitudeConstraint, LateralReference, NavigationSolution, SolutionQuality,
};

use crate::config::{GuidanceConfig, VelocityGuidanceConfig};
use crate::derive::guide;
use crate::refusal::GuidanceRefusal;
use crate::scenario::{CLOCK, deg, equator_leg, now, solution};
use crate::velocity::guide_velocity;

/// An INS-only solution: quality `quality`, vertical 1-sigma `vertical_m`.
fn coasting(quality: SolutionQuality, vertical_m: f64) -> NavigationSolution {
    let mut own = solution(quality, deg(0.0, 0.5, 100.0), now());
    own.integrity.vertical_1sigma_m = vertical_m;
    own
}

/// Both derivations on an altitude-constrained leg.
fn both(own: &NavigationSolution, config: GuidanceConfig) -> [Result<(), GuidanceRefusal>; 2] {
    let (from, to) = equator_leg();
    let to = to.with_altitude(AltitudeConstraint::At(100.0));
    let reference = LateralReference::track(from);
    let velocity = VelocityGuidanceConfig {
        admission: config,
        ..VelocityGuidanceConfig::default()
    };
    [
        guide(own, reference, &to, now(), CLOCK, &config).map(|_| ()),
        guide_velocity(own, reference, &to, now(), CLOCK, &velocity).map(|_| ()),
    ]
}

fn unusable_floor() -> GuidanceConfig {
    GuidanceConfig {
        minimum_quality: SolutionQuality::Unusable,
        ..GuidanceConfig::default()
    }
}

#[test]
fn an_unusable_floor_does_not_lift_the_vertical_limit() {
    // The measured INS-only case: the caller lowers the quality floor
    // only, so the default 50 m vertical limit still applies.
    let own = coasting(SolutionQuality::Unusable, 800.0);
    for outcome in both(&own, unusable_floor()) {
        assert_eq!(
            outcome,
            Err(GuidanceRefusal::VerticalUncertaintyExceeded {
                vertical_1sigma_m: 800.0,
                bound_m: GuidanceConfig::DEFAULT_MAX_VERTICAL_1SIGMA_M,
            })
        );
    }
}

#[test]
fn an_infinite_vertical_limit_is_an_explicit_no_limit() {
    let own = coasting(SolutionQuality::Unusable, 800.0);
    let config = unusable_floor().with_max_vertical_1sigma_m(f64::INFINITY);
    for outcome in both(&own, config) {
        assert_eq!(outcome, Ok(()));
    }
}

#[test]
fn an_infinite_vertical_limit_still_refuses_a_non_finite_sigma() {
    let config = unusable_floor().with_max_vertical_1sigma_m(f64::INFINITY);
    for sigma in [f64::NAN, f64::INFINITY] {
        let own = coasting(SolutionQuality::Unusable, sigma);
        for outcome in both(&own, config) {
            assert!(
                matches!(
                    outcome,
                    Err(GuidanceRefusal::VerticalUncertaintyExceeded { .. })
                ),
                "sigma {sigma}: {outcome:?}"
            );
        }
    }
}

#[test]
fn a_nan_or_negative_vertical_limit_refuses_every_constrained_leg() {
    let own = coasting(SolutionQuality::Good, 1.0);
    for bound in [f64::NAN, -1.0, f64::NEG_INFINITY] {
        let config = GuidanceConfig::default().with_max_vertical_1sigma_m(bound);
        for outcome in both(&own, config) {
            assert!(
                matches!(
                    outcome,
                    Err(GuidanceRefusal::VerticalUncertaintyExceeded { .. })
                ),
                "bound {bound}: {outcome:?}"
            );
        }
    }
}

#[test]
fn the_quality_refusal_and_the_vertical_refusal_are_distinct() {
    // Below the default floor, the quality refusal comes first and does
    // not depend on the vertical limit, even when that limit is lifted.
    let own = coasting(SolutionQuality::Unusable, 800.0);
    let lifted = GuidanceConfig::default().with_max_vertical_1sigma_m(f64::INFINITY);
    for config in [GuidanceConfig::default(), lifted] {
        for outcome in both(&own, config) {
            assert_eq!(
                outcome,
                Err(GuidanceRefusal::IntegrityBelowFloor {
                    quality: SolutionQuality::Unusable,
                    floor: SolutionQuality::Degraded,
                })
            );
        }
    }
    // Above the floor, only the vertical limit refuses.
    let degraded = coasting(SolutionQuality::Degraded, 800.0);
    for outcome in both(&degraded, GuidanceConfig::default()) {
        assert!(matches!(
            outcome,
            Err(GuidanceRefusal::VerticalUncertaintyExceeded { bound_m, .. }) if bound_m == 50.0
        ));
    }
}
