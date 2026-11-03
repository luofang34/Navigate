//! One measurement update of an initialized core.
//!
//! The in-order path, the delayed path, and the replay of the history
//! use this module. Each call builds the measurement model against the
//! core that it updates. Thus a replay after a re-anchor uses the plane
//! of the replayed core.

use nalgebra::Vector3;
use navigate_contract::MonotonicNanos;
use navigate_geodesy::NedOffset;

use super::{Core, reanchor};
use crate::admission::seconds;
use crate::config::FusionConfig;
use crate::filter::{
    self, AltitudeModel, LinearBlock, MagneticAnomalyModel, MapLinearization, MeasurementModel,
    RangeModel,
};
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

/// How much of the admission an update runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    /// A new observation: `admissible` and the chi-square gate run.
    Gated,
    /// A replay of an observation that the filter admitted before. The
    /// checks do not run again, so the replay is deterministic.
    Replay,
}

/// Updates `core` with `value` at `acquired_at` and returns the
/// chi-square statistic of the innovation. On an error, `core` is not
/// changed.
pub(super) fn apply_value(
    core: &mut Core,
    value: &ObservationValue,
    acquired_at: MonotonicNanos,
    config: &FusionConfig,
    mode: Mode,
) -> Result<f64, RejectionReason> {
    let vector_gate = config.innovation_gate_chi2;
    let scalar_gate = config.scalar_gate_chi2;
    match value {
        ObservationValue::PositionFix {
            position,
            covariance,
        } => {
            let offset = core.plane.to_ned(position);
            let z = Vector3::new(offset.north_m, offset.east_m, offset.down_m);
            let model = LinearBlock::position(z, filter::cov3_to_matrix(covariance));
            step(core, config, &model, vector_gate, acquired_at, mode)
        }
        ObservationValue::VelocityFix {
            velocity,
            covariance,
        } => {
            // The velocity arrives in the NED axes at the vehicle. The
            // vehicle position comes from the state at the measurement
            // time.
            let prop = core
                .state
                .propagated(dt_s(core, acquired_at), config.process_noise_accel_psd);
            let vehicle = core
                .plane
                .from_ned(&NedOffset::new(prop.x[0], prop.x[1], prop.x[2]));
            let (z, r_plane) = reanchor::vehicle_to_plane(
                &core.plane,
                vehicle,
                Vector3::new(velocity.north_mps, velocity.east_mps, velocity.down_mps),
                filter::cov3_to_matrix(covariance),
            );
            let model = LinearBlock::velocity(z, r_plane);
            step(core, config, &model, vector_gate, acquired_at, mode)
        }
        ObservationValue::Range {
            station,
            range_m,
            variance_m2,
        } => {
            let offset = core.plane.to_ned(station);
            let station_ned = Vector3::new(offset.north_m, offset.east_m, offset.down_m);
            let model = RangeModel::new(station_ned, *range_m, *variance_m2);
            step(core, config, &model, scalar_gate, acquired_at, mode)
        }
        ObservationValue::Altitude {
            altitude_m,
            variance_m2,
        } => {
            let model = AltitudeModel::new(core.plane, *altitude_m, *variance_m2);
            step(core, config, &model, scalar_gate, acquired_at, mode)
        }
        ObservationValue::MagneticAnomaly {
            measured_t,
            map_t_at_lin,
            gradient_ned_t_per_m,
            linearization_point,
            valid_radius_m,
            variance_t2,
        } => {
            let map = MapLinearization {
                linearization_point: *linearization_point,
                map_t_at_lin: *map_t_at_lin,
                gradient_ned_t_per_m: *gradient_ned_t_per_m,
                valid_radius_m: *valid_radius_m,
            };
            let variance_t2 = inflated_magnetic_variance(*variance_t2, config)?;
            let model = MagneticAnomalyModel::new(&core.plane, &map, *measured_t, variance_t2)
                .ok_or(RejectionReason::NonFiniteValue)?;
            step(core, config, &model, scalar_gate, acquired_at, mode)
        }
        _ => Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() }),
    }
}

/// Adds the error budget of the unmodeled map bias to the magnetic
/// variance. A negative or non-finite budget fails closed.
fn inflated_magnetic_variance(
    variance_t2: f64,
    config: &FusionConfig,
) -> Result<f64, RejectionReason> {
    let budget = config.unmodeled_magnetic_bias_variance_t2;
    if !(budget.is_finite() && budget >= 0.0) {
        return Err(RejectionReason::ImplausibleCovariance);
    }
    Ok(variance_t2 + budget)
}

fn dt_s(core: &Core, acquired_at: MonotonicNanos) -> f64 {
    acquired_at
        .elapsed_since(core.state_at)
        .map_or(0.0, seconds)
}

/// Propagates to the measurement time, runs the checks of `mode`, and
/// changes `core` only after every check passes (ADR-0003). The order is
/// `admissible`, then the factorization, then the chi-square gate. A
/// measurement at the state time applies with `dt = 0`.
fn step<const M: usize>(
    core: &mut Core,
    config: &FusionConfig,
    model: &impl MeasurementModel<M>,
    gate_chi2: f64,
    acquired_at: MonotonicNanos,
    mode: Mode,
) -> Result<f64, RejectionReason> {
    let candidate = core
        .state
        .propagated(dt_s(core, acquired_at), config.process_noise_accel_psd);
    if mode == Mode::Gated {
        model.admissible(&candidate.x, &candidate.p, config)?;
    }
    let prepared = candidate
        .prepare_update(model)
        .ok_or(RejectionReason::ImplausibleCovariance)?;
    if mode == Mode::Gated && (prepared.chi2.is_nan() || prepared.chi2 > gate_chi2) {
        return Err(RejectionReason::InnovationGate {
            chi2: prepared.chi2,
            threshold: gate_chi2,
        });
    }
    core.state = candidate.apply_update(&prepared);
    if acquired_at > core.state_at {
        core.state_at = acquired_at;
    }
    reanchor::reanchor_if_needed(core, config);
    Ok(prepared.chi2)
}
