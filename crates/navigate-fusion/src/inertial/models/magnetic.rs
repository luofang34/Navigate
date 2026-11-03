//! Scalar magnetic anomaly with a map bias state (ADR-0014).
//!
//! The model is `h(x) = map_t_at_lin + g·(p − p_lin) + b_mag`. The bias
//! `b_mag` belongs to one map: one source and one map revision. When the
//! source or the revision changes, the filter starts a new bias at zero
//! with the stationary variance and no correlation. The reset is part of
//! the candidate state, so a refused observation does not reset the
//! bias.

use nalgebra::SMatrix;

use super::scalar;
use crate::inertial::config::InertialConfig;
use crate::inertial::estimate::{InertialCore, Mode};
use crate::inertial::nominal::{MAG_BIAS, N, POS};
use crate::kalman::{
    MagneticAnomalyModel, MapLinearization, check_gradient_information, check_linearization_region,
    horizontal_1sigma_m,
};
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;
use navigate_astrodynamics::SurfaceFrame;

pub(super) fn apply(
    core: &mut InertialCore,
    surface: &SurfaceFrame,
    obs: &Observation,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason> {
    let ObservationValue::MagneticAnomaly {
        measured_t,
        map_t_at_lin,
        gradient_ned_t_per_m,
        linearization_point,
        valid_radius_m,
        variance_t2,
    } = obs.value
    else {
        return Err(RejectionReason::UnsupportedMeasurement {
            kind: obs.value.kind(),
        });
    };
    let map = MapLinearization {
        linearization_point,
        map_t_at_lin,
        gradient_ned_t_per_m,
        valid_radius_m,
    };
    let model = MagneticAnomalyModel::new(surface.plane(), &map, measured_t, variance_t2)
        .ok_or(RejectionReason::NonFiniteValue)?;
    let key = (obs.stamp.source, obs.map_revision);
    let mut candidate = *core;
    if candidate.mag_key.is_some_and(|active| active != key) {
        start_new_bias(&mut candidate, config);
    }
    let g = model.g_plane();
    let offset = candidate.nominal.p - model.p_lin_plane();
    if mode == Mode::Gated {
        check_linearization_region(
            offset.norm(),
            horizontal_1sigma_m(&candidate.p),
            model.valid_radius_m(),
            &config.fusion,
        )?;
        let p_pos = candidate.p.fixed_view::<3, 3>(POS, POS).into_owned();
        check_gradient_information(&g, &p_pos, variance_t2, &config.fusion)?;
    }
    let predicted = model.map_t_at_lin() + g.dot(&offset) + candidate.nominal.b_mag;
    let mut h = SMatrix::<f64, 1, N>::zeros();
    h.fixed_view_mut::<1, 3>(0, POS).copy_from(&g.transpose());
    h[(0, MAG_BIAS)] = 1.0;
    let linearized = scalar(measured_t - predicted, h, variance_t2);
    let chi2 = candidate.update(&linearized, config.fusion.scalar_gate_chi2, mode)?;
    candidate.mag_key = Some(key);
    *core = candidate;
    Ok(chi2)
}

/// Starts a new bias: zero mean, the stationary variance, and no
/// correlation with the other states.
fn start_new_bias(core: &mut InertialCore, config: &InertialConfig) {
    core.nominal.b_mag = 0.0;
    core.p.row_mut(MAG_BIAS).fill(0.0);
    core.p.column_mut(MAG_BIAS).fill(0.0);
    core.p[(MAG_BIAS, MAG_BIAS)] = config.mag_bias.stationary_variance();
}
