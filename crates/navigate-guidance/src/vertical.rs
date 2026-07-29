//! Vertical deviation from a waypoint's altitude constraint.

use navigate_contract::AltitudeConstraint;

/// Signed vertical deviation in meters, positive above the target
/// profile — the sign convention of
/// [`navigate_contract::GuidanceSetpoint::DeviationTracking`].
///
/// - [`AltitudeConstraint::At`] always reports the signed deviation
///   from the target altitude.
/// - [`AltitudeConstraint::AtOrAbove`] / [`AltitudeConstraint::AtOrBelow`]
///   report `0.0` while satisfied; a violation reports only its
///   direction — below an `AtOrAbove` floor is negative, above an
///   `AtOrBelow` ceiling is positive.
/// - No constraint reports `0.0`; profile interpolation between
///   constrained waypoints is a designed extension.
pub(crate) fn deviation_m(ownship_altitude_m: f64, constraint: Option<&AltitudeConstraint>) -> f64 {
    match constraint {
        None => 0.0,
        Some(AltitudeConstraint::At(target_m)) => ownship_altitude_m - target_m,
        Some(AltitudeConstraint::AtOrAbove(floor_m)) => {
            if ownship_altitude_m >= *floor_m {
                0.0
            } else {
                ownship_altitude_m - floor_m
            }
        }
        Some(AltitudeConstraint::AtOrBelow(ceiling_m)) => {
            if ownship_altitude_m <= *ceiling_m {
                0.0
            } else {
                ownship_altitude_m - ceiling_m
            }
        }
        // The constraint vocabulary grows in the contract crate first; a
        // variant this crate cannot interpret reports no deviation
        // rather than an invented one.
        Some(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use navigate_contract::AltitudeConstraint;

    use super::deviation_m;

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    #[test]
    fn at_reports_signed_deviation_both_sides() {
        let at = AltitudeConstraint::At(100.0);
        assert!(
            close(deviation_m(120.0, Some(&at)), 20.0),
            "above is positive"
        );
        assert!(
            close(deviation_m(80.0, Some(&at)), -20.0),
            "below is negative"
        );
        assert!(close(deviation_m(100.0, Some(&at)), 0.0));
    }

    #[test]
    fn at_or_above_reports_only_the_violation_direction() {
        let floor = AltitudeConstraint::AtOrAbove(100.0);
        assert!(
            close(deviation_m(150.0, Some(&floor)), 0.0),
            "satisfied is zero"
        );
        assert!(
            close(deviation_m(100.0, Some(&floor)), 0.0),
            "boundary is satisfied"
        );
        assert!(
            close(deviation_m(80.0, Some(&floor)), -20.0),
            "violation is negative: below profile"
        );
    }

    #[test]
    fn at_or_below_reports_only_the_violation_direction() {
        let ceiling = AltitudeConstraint::AtOrBelow(100.0);
        assert!(
            close(deviation_m(50.0, Some(&ceiling)), 0.0),
            "satisfied is zero"
        );
        assert!(
            close(deviation_m(100.0, Some(&ceiling)), 0.0),
            "boundary is satisfied"
        );
        assert!(
            close(deviation_m(120.0, Some(&ceiling)), 20.0),
            "violation is positive: above profile"
        );
    }

    #[test]
    fn no_constraint_is_zero() {
        assert!(close(deviation_m(1234.5, None), 0.0));
    }
}
