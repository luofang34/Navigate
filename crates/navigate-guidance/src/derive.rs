//! The deviation-tracking derivation: solution plus leg geometry to one
//! flight-director setpoint.

use navigate_contract::{
    ClockDomainId, GeodeticPosition, GuidanceCommand, GuidanceSetpoint, MonotonicNanos,
    NavigationSolution, Waypoint,
};

use crate::admission::admit_leg;
use crate::config::GuidanceConfig;
use crate::refusal::GuidanceRefusal;
use crate::vertical;

/// Derives one deviation-tracking guidance command from a navigation
/// solution and the active leg, or refuses with a typed reason.
///
/// `now` must be read on the solution's clock domain
/// (`solution.stamp.clock`): readings from different domains are never
/// subtracted, so a mismatched `now_clock` is refused before any age
/// arithmetic.
///
/// Lateral: the reference track runs `leg_from` → `leg_to` when the leg
/// has an upstream fix. Direct-to (`leg_from` = `None`) anchors the
/// track at ownship, so the cross-track deviation is zero by
/// construction and the course is the live bearing to the waypoint.
/// `lateral_m` is positive right of course, matching
/// [`navigate_geodesy::cross_track_m`].
///
/// Vertical: deviation from `leg_to`'s altitude constraint, positive
/// above the target profile; the `OrAbove`/`OrBelow` forms report `0.0`
/// while satisfied (see [`GuidanceSetpoint::DeviationTracking`]). A
/// waypoint without a constraint yields `0.0`; profile interpolation
/// between constrained waypoints is a designed extension.
///
/// # Errors
///
/// - [`GuidanceRefusal::ClockDomainMismatch`] when `now_clock` differs
///   from the solution stamp's clock domain.
/// - [`GuidanceRefusal::IntegrityBelowFloor`] when solution quality is
///   below `config.minimum_quality`.
/// - [`GuidanceRefusal::ClockInversion`] when `now` is earlier than the
///   solution's `solved_at`.
/// - [`GuidanceRefusal::SolutionStale`] when the solution's age exceeds
///   `config.max_solution_age`.
/// - [`GuidanceRefusal::ImplausibleTarget`] when `leg_to`'s position
///   fails the geodetic plausibility screen, or when the leg's endpoints
///   coincide (a leg whose endpoints coincide cannot define a course).
pub fn guide(
    solution: &NavigationSolution,
    leg_from: Option<&GeodeticPosition>,
    leg_to: &Waypoint,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &GuidanceConfig,
) -> Result<GuidanceCommand, GuidanceRefusal> {
    let leg = admit_leg(solution, leg_from, leg_to, now, now_clock, config)?;
    let vertical_m = vertical::deviation_m(solution.position.altitude_m, leg_to.altitude.as_ref());
    Ok(GuidanceCommand::new(
        now,
        GuidanceSetpoint::DeviationTracking {
            lateral_m: leg.cross_track_m,
            vertical_m,
            course_rad: leg.course_rad,
        },
        solution.stamp,
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use navigate_contract::{
        AltitudeConstraint, ClockDomainId, DurationNanos, GeodeticPosition, GuidanceSetpoint,
        MonotonicNanos, SolutionQuality, Waypoint,
    };
    use navigate_geodesy::{cross_track_m, initial_bearing_rad, wgs84};

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
            Some(&from),
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
            Some(&from),
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
        let command =
            guide(&own, None, &to, now(), CLOCK, &GuidanceConfig::default()).expect("guides");
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
            Some(&from),
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
            Some(&from),
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
        let refusal = guide(&own, Some(&from), &to, now(), CLOCK, &tightened).expect_err("refuses");
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
        assert!(guide(&own, Some(&from), &to, at_bound, CLOCK, &config).is_ok());
        let past_bound = MonotonicNanos::from_nanos(at_bound.as_nanos() + 1);
        let refusal =
            guide(&own, Some(&from), &to, past_bound, CLOCK, &config).expect_err("refuses");
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
            Some(&from),
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
            Some(&from),
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
                Some(&from),
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
            Some(&anchor),
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
    fn implausible_target_is_refused_by_name() {
        let (from, _) = equator_leg();
        let bad = Waypoint::new("BAD".into(), GeodeticPosition::new(f64::NAN, 0.0, 0.0));
        let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now());
        let refusal = guide(
            &own,
            Some(&from),
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
}
