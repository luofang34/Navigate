//! Admission gates that do not depend on the filter state (ADR-0003,
//! ADR-0013). [`crate::NavigationFilter`] and
//! [`crate::InertialNavigationFilter`] run the same gates.

use navigate_contract::{
    AttitudeQuaternion, ClockDomainId, DurationNanos, FrameId, MonotonicNanos, ObservationStamp,
    SourceComposition, SymmetricCov3,
};

use crate::float_math::sqrt;
use crate::kalman;
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

/// Largest distance of a quaternion norm from 1 that admission accepts.
const UNIT_NORM_TOLERANCE: f64 = 1e-3;

/// Largest distance of a beam direction norm from 1 that admission
/// accepts.
const BEAM_NORM_TOLERANCE: f64 = 1e-3;

/// The stamp must come from the clock domain of the filter.
pub(crate) fn check_clock(
    expected: ClockDomainId,
    stamp: &ObservationStamp,
) -> Result<(), RejectionReason> {
    if stamp.clock != expected {
        return Err(RejectionReason::ClockDomainMismatch {
            expected,
            got: stamp.clock,
        });
    }
    Ok(())
}

/// A declared frame must be `expected`. A missing frame passes only when
/// `require_explicit` is `false`.
pub(crate) fn check_frame(
    expected: FrameId,
    declared: Option<FrameId>,
    require_explicit: bool,
) -> Result<(), RejectionReason> {
    match declared {
        Some(got) if got != expected => Err(RejectionReason::FrameMismatch { expected, got }),
        None if require_explicit => Err(RejectionReason::MissingFrame { expected }),
        _ => Ok(()),
    }
}

/// An acquisition time ahead of `now` is the same time-discipline
/// violation as a per-source regression. An age above `bound` is stale.
pub(crate) fn check_freshness(
    stamp: &ObservationStamp,
    now: MonotonicNanos,
    bound: DurationNanos,
) -> Result<(), RejectionReason> {
    let Some(age) = now.elapsed_since(stamp.acquired_at) else {
        return Err(RejectionReason::AcquisitionTimeRegression {
            last: now,
            got: stamp.acquired_at,
        });
    };
    if age > bound {
        return Err(RejectionReason::Stale { age, bound });
    }
    Ok(())
}

pub(crate) fn check_composition(composition: SourceComposition) -> Result<(), RejectionReason> {
    if composition.is_empty() {
        return Err(RejectionReason::EmptyComposition);
    }
    if composition.is_estimator_derived() {
        return Err(RejectionReason::EstimatorDerived);
    }
    Ok(())
}

/// Physical plausibility of every value. A filter checks that it has the
/// model of the value before this check.
pub(crate) fn check_value(value: &ObservationValue) -> Result<(), RejectionReason> {
    match value {
        ObservationValue::PositionFix {
            position,
            covariance,
        } => {
            require(position.is_plausible())?;
            check_cov3(covariance)
        }
        ObservationValue::VelocityFix {
            velocity,
            covariance,
        } => {
            require(velocity.is_finite())?;
            check_cov3(covariance)
        }
        ObservationValue::Range {
            station,
            range_m,
            variance_m2,
        } => {
            require(station.is_plausible() && range_m.is_finite() && *range_m >= 0.0)?;
            check_scalar_variance(*variance_m2)
        }
        ObservationValue::Altitude {
            altitude_m,
            variance_m2,
        } => {
            require(altitude_m.is_finite())?;
            check_scalar_variance(*variance_m2)
        }
        ObservationValue::MagneticAnomaly {
            measured_t,
            map_t_at_lin,
            gradient_ned_t_per_m,
            linearization_point,
            valid_radius_m,
            variance_t2,
        } => {
            require(
                linearization_point.is_plausible()
                    && measured_t.is_finite()
                    && map_t_at_lin.is_finite()
                    && gradient_ned_t_per_m.iter().all(|g| g.is_finite())
                    && valid_radius_m.is_finite()
                    && *valid_radius_m > 0.0,
            )?;
            check_scalar_variance(*variance_t2)
        }
        ObservationValue::Pseudorange {
            satellite_ecef_m,
            pseudorange_m,
            variance_m2,
        } => {
            require(satellite_ecef_m.iter().all(|c| c.is_finite()) && pseudorange_m.is_finite())?;
            check_scalar_variance(*variance_m2)
        }
        _ => check_attitude_or_cartesian(value),
    }
}

/// The plausibility of the values that the inertial filter adds
/// (ADR-0014).
fn check_attitude_or_cartesian(value: &ObservationValue) -> Result<(), RejectionReason> {
    match value {
        ObservationValue::VisualPose {
            position,
            position_covariance,
            attitude,
            attitude_covariance,
        } => {
            require(position.is_plausible() && is_unit_quaternion(attitude))?;
            check_cov3(position_covariance)?;
            check_cov3(attitude_covariance)
        }
        ObservationValue::TerrainClearance {
            clearance_m,
            terrain_height_m,
            terrain_slope_north_east,
            beam_body_unit,
            variance_m2,
        } => {
            let beam_norm = sqrt(beam_body_unit.iter().map(|c| c * c).sum::<f64>());
            require(
                clearance_m.is_finite()
                    && *clearance_m > 0.0
                    && terrain_height_m.is_finite()
                    && terrain_slope_north_east.iter().all(|s| s.is_finite())
                    && beam_norm.is_finite()
                    && (beam_norm - 1.0).abs() <= BEAM_NORM_TOLERANCE,
            )?;
            check_scalar_variance(*variance_m2)
        }
        ObservationValue::AttitudeFix {
            attitude,
            covariance,
        } => {
            require(is_unit_quaternion(attitude))?;
            check_cov3(covariance)
        }
        ObservationValue::CartesianPositionFix {
            position,
            covariance,
        } => {
            require(position.is_finite())?;
            check_cov3(covariance)
        }
        ObservationValue::CartesianVelocityFix {
            velocity,
            covariance,
        } => {
            require(velocity.is_finite())?;
            check_cov3(covariance)
        }
        ObservationValue::CartesianRange {
            beacon,
            range_m,
            variance_m2,
        } => {
            require(beacon.is_finite() && range_m.is_finite() && *range_m >= 0.0)?;
            check_scalar_variance(*variance_m2)
        }
        _ => Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() }),
    }
}

const fn require(plausible: bool) -> Result<(), RejectionReason> {
    if plausible {
        Ok(())
    } else {
        Err(RejectionReason::NonFiniteValue)
    }
}

/// Whether every component is finite and the norm is 1 within
/// [`UNIT_NORM_TOLERANCE`].
pub(crate) fn is_unit_quaternion(q: &AttitudeQuaternion) -> bool {
    let norm = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    norm.is_finite() && (norm - 1.0).abs() <= UNIT_NORM_TOLERANCE
}

/// A 3×3 covariance must be finite and positive definite.
pub(crate) fn check_cov3(covariance: &SymmetricCov3) -> Result<(), RejectionReason> {
    if covariance.is_plausible()
        && kalman::is_positive_definite(&kalman::cov3_to_matrix(covariance))
    {
        Ok(())
    } else {
        Err(RejectionReason::ImplausibleCovariance)
    }
}

/// A scalar variance must be finite and strictly positive. The check has
/// no lower scale limit: a magnetic variance of 1e-18 T² is valid.
pub(crate) fn check_scalar_variance(variance: f64) -> Result<(), RejectionReason> {
    if variance.is_finite() && variance > 0.0 {
        Ok(())
    } else {
        Err(RejectionReason::ImplausibleCovariance)
    }
}

pub(crate) fn seconds(duration: DurationNanos) -> f64 {
    duration.as_nanos() as f64 * 1e-9
}

#[cfg(test)]
mod tests;
