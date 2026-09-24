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
            if !(variance_m2.is_finite() && *variance_m2 > 0.0) {
                return Err(RejectionReason::ImplausibleCovariance);
            }
            return Ok(());
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

pub(super) fn seconds(duration: DurationNanos) -> f64 {
    duration.as_nanos() as f64 * 1e-9
}
