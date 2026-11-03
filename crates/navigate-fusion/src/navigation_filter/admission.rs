//! Admission checks that do not depend on filter state.

use navigate_contract::{DurationNanos, SourceComposition};

use crate::filter;
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

pub(super) fn check_composition(composition: SourceComposition) -> Result<(), RejectionReason> {
    if composition.is_empty() {
        return Err(RejectionReason::EmptyComposition);
    }
    if composition.is_estimator_derived() {
        return Err(RejectionReason::EstimatorDerived);
    }
    Ok(())
}

pub(super) fn check_value(value: &ObservationValue) -> Result<(), RejectionReason> {
    if !value.is_supported() {
        return Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() });
    }
    let covariance = match value {
        ObservationValue::PositionFix {
            position,
            covariance,
        } => {
            if !position.is_plausible() {
                return Err(RejectionReason::NonFiniteValue);
            }
            covariance
        }
        ObservationValue::VelocityFix {
            velocity,
            covariance,
        } => {
            if !velocity.is_finite() {
                return Err(RejectionReason::NonFiniteValue);
            }
            covariance
        }
        ObservationValue::Range {
            station,
            range_m,
            variance_m2,
        } => {
            if !station.is_plausible() || !(range_m.is_finite() && *range_m >= 0.0) {
                return Err(RejectionReason::NonFiniteValue);
            }
            return check_scalar_variance(*variance_m2);
        }
        ObservationValue::Altitude {
            altitude_m,
            variance_m2,
        } => {
            if !altitude_m.is_finite() {
                return Err(RejectionReason::NonFiniteValue);
            }
            return check_scalar_variance(*variance_m2);
        }
        ObservationValue::MagneticAnomaly {
            measured_t,
            map_t_at_lin,
            gradient_ned_t_per_m,
            linearization_point,
            valid_radius_m,
            variance_t2,
        } => {
            let values_finite = linearization_point.is_plausible()
                && measured_t.is_finite()
                && map_t_at_lin.is_finite()
                && gradient_ned_t_per_m.iter().all(|g| g.is_finite())
                && valid_radius_m.is_finite()
                && *valid_radius_m > 0.0;
            if !values_finite {
                return Err(RejectionReason::NonFiniteValue);
            }
            return check_scalar_variance(*variance_t2);
        }
        ObservationValue::Pseudorange { .. } | ObservationValue::VisualPose { .. } => {
            return Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() });
        }
    };
    if !covariance.is_plausible()
        || !filter::is_positive_definite(&filter::cov3_to_matrix(covariance))
    {
        return Err(RejectionReason::ImplausibleCovariance);
    }
    Ok(())
}

/// A scalar variance must be finite and strictly positive. The check has
/// no lower scale limit: a magnetic variance of 1e-18 T² is valid.
fn check_scalar_variance(variance: f64) -> Result<(), RejectionReason> {
    if variance.is_finite() && variance > 0.0 {
        Ok(())
    } else {
        Err(RejectionReason::ImplausibleCovariance)
    }
}

pub(super) fn seconds(duration: DurationNanos) -> f64 {
    duration.as_nanos() as f64 * 1e-9
}
