//! The error dynamics of the bias, empirical acceleration, and clock
//! states (ADR-0014, ADR-0018).
//!
//! Each propagation mode uses these blocks. A state whose model is not
//! configured has a zero row and a zero column in `F`, in `Q_c`, and in
//! the initial covariance. Thus its variance stays zero, its Kalman gain
//! stays zero, and it does not couple into the other states.

use nalgebra::{Matrix3, Vector3};

use super::config::InertialConfig;
use super::nominal::{ACCEL_BIAS, ErrorMat, GYRO_BIAS, InertialNominal};
use crate::kalman::FogmBias;

/// Adds the first-order Gauss–Markov decay of each bias and of the
/// empirical acceleration, and the clock drift into the clock bias, to
/// `f`.
pub(crate) fn add_dynamics<const N: usize>(f: &mut ErrorMat<N>, config: &InertialConfig) {
    let layout = config.layout;
    let identity = Matrix3::<f64>::identity();
    f.fixed_view_mut::<3, 3>(ACCEL_BIAS, ACCEL_BIAS)
        .copy_from(&(identity * config.accel_bias.decay_rate()));
    f.fixed_view_mut::<3, 3>(GYRO_BIAS, GYRO_BIAS)
        .copy_from(&(identity * config.gyro_bias.decay_rate()));
    if let Some(i) = layout.alt_bias() {
        f[(i, i)] = config.baro_bias.decay_rate();
    }
    if let Some(i) = layout.mag_bias() {
        f[(i, i)] = config.mag_bias.decay_rate();
    }
    if let (Some(dmc), Some(i)) = (&config.dmc, layout.dmc()) {
        f.fixed_view_mut::<3, 3>(i, i)
            .copy_from(&(identity * dmc.decay_rate()));
    }
    if let (Some(_), Some(i)) = (&config.clock_model, layout.clock()) {
        f[(i, i + 1)] = 1.0;
    }
}

/// Adds the noise density of each bias, of the empirical acceleration,
/// and of the clock to `q`. A first-order Gauss–Markov state has the
/// density `2·σ²/τ`. The clock has `diag(S_f, S_g)`.
pub(crate) fn add_noise<const N: usize>(q: &mut ErrorMat<N>, config: &InertialConfig) {
    let layout = config.layout;
    for axis in 0..3 {
        q[(ACCEL_BIAS + axis, ACCEL_BIAS + axis)] = config.accel_bias.continuous_psd();
        q[(GYRO_BIAS + axis, GYRO_BIAS + axis)] = config.gyro_bias.continuous_psd();
        if let (Some(dmc), Some(i)) = (&config.dmc, layout.dmc()) {
            q[(i + axis, i + axis)] = dmc.continuous_psd();
        }
    }
    if let Some(i) = layout.alt_bias() {
        q[(i, i)] = config.baro_bias.continuous_psd();
    }
    if let Some(i) = layout.mag_bias() {
        q[(i, i)] = config.mag_bias.continuous_psd();
    }
    if let (Some(clock), Some(i)) = (&config.clock_model, layout.clock()) {
        q[(i, i)] = clock.white_frequency_psd_m2_per_s;
        q[(i + 1, i + 1)] = clock.random_walk_frequency_psd_m2_per_s3;
    }
}

/// Sets the initial variance of each bias and of the empirical
/// acceleration to its stationary variance, and the clock variances to
/// the configured initial values.
pub(crate) fn set_initial_variances<const N: usize>(p: &mut ErrorMat<N>, config: &InertialConfig) {
    let layout = config.layout;
    for axis in 0..3 {
        p[(ACCEL_BIAS + axis, ACCEL_BIAS + axis)] = config.accel_bias.stationary_variance();
        p[(GYRO_BIAS + axis, GYRO_BIAS + axis)] = config.gyro_bias.stationary_variance();
        if let (Some(dmc), Some(i)) = (&config.dmc, layout.dmc()) {
            p[(i + axis, i + axis)] = dmc.stationary_variance();
        }
    }
    if let Some(i) = layout.alt_bias() {
        p[(i, i)] = config.baro_bias.stationary_variance();
    }
    if let Some(i) = layout.mag_bias() {
        p[(i, i)] = config.mag_bias.stationary_variance();
    }
    if let (Some(clock), Some(i)) = (&config.clock_model, layout.clock()) {
        p[(i, i)] = clock.initial_bias_sigma_m * clock.initial_bias_sigma_m;
        p[(i + 1, i + 1)] = clock.initial_drift_sigma_mps * clock.initial_drift_sigma_mps;
    }
}

/// `nominal` after `dt_s` seconds of the bias, empirical acceleration,
/// and clock models: each first-order Gauss–Markov mean decays, and the
/// clock bias integrates the constant drift. The position, the velocity,
/// and the attitude do not change.
pub(crate) fn advance(
    nominal: &InertialNominal,
    dt_s: f64,
    config: &InertialConfig,
) -> InertialNominal {
    let decay = |bias: &FogmBias, value: f64| bias.propagate(value, dt_s).0;
    let a_dmc = config
        .dmc
        .as_ref()
        .map_or(Vector3::zeros(), |dmc| nominal.a_dmc.map(|a| decay(dmc, a)));
    let clock = config.clock_model.is_some();
    InertialNominal {
        b_a: nominal.b_a.map(|b| decay(&config.accel_bias, b)),
        b_g: nominal.b_g.map(|b| decay(&config.gyro_bias, b)),
        b_alt: decay(&config.baro_bias, nominal.b_alt),
        b_mag: decay(&config.mag_bias, nominal.b_mag),
        a_dmc,
        clock_bias_m: if clock {
            nominal.clock_bias_m + nominal.clock_drift_mps * dt_s
        } else {
            0.0
        },
        clock_drift_mps: if clock { nominal.clock_drift_mps } else { 0.0 },
        ..*nominal
    }
}
