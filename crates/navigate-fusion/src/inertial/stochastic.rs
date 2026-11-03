//! The error dynamics of the bias, empirical acceleration, and clock
//! states (ADR-0014, ADR-0018).
//!
//! Each propagation mode uses these blocks. A state whose model is not
//! configured has a zero row and a zero column in `F`, in `Q_c`, and in
//! the initial covariance. Thus its variance stays zero, its Kalman gain
//! stays zero, and it does not couple into the other states.

use nalgebra::{Matrix3, Vector3};

use super::config::InertialConfig;
use super::nominal::{
    ACCEL_BIAS, ALT_BIAS, CLOCK_BIAS, CLOCK_DRIFT, DMC, ErrorMat, GYRO_BIAS, InertialNominal,
    MAG_BIAS,
};
use crate::kalman::FogmBias;

/// Adds the first-order Gauss–Markov decay of each bias and of the
/// empirical acceleration, and the clock drift into the clock bias, to
/// `f`.
pub(crate) fn add_dynamics(f: &mut ErrorMat, config: &InertialConfig) {
    let identity = Matrix3::<f64>::identity();
    f.fixed_view_mut::<3, 3>(ACCEL_BIAS, ACCEL_BIAS)
        .copy_from(&(identity * config.accel_bias.decay_rate()));
    f.fixed_view_mut::<3, 3>(GYRO_BIAS, GYRO_BIAS)
        .copy_from(&(identity * config.gyro_bias.decay_rate()));
    f[(ALT_BIAS, ALT_BIAS)] = config.baro_bias.decay_rate();
    f[(MAG_BIAS, MAG_BIAS)] = config.mag_bias.decay_rate();
    if let Some(dmc) = &config.dmc {
        f.fixed_view_mut::<3, 3>(DMC, DMC)
            .copy_from(&(identity * dmc.decay_rate()));
    }
    if config.clock_model.is_some() {
        f[(CLOCK_BIAS, CLOCK_DRIFT)] = 1.0;
    }
}

/// Adds the noise density of each bias, of the empirical acceleration,
/// and of the clock to `q`. A first-order Gauss–Markov state has the
/// density `2·σ²/τ`. The clock has `diag(S_f, S_g)`.
pub(crate) fn add_noise(q: &mut ErrorMat, config: &InertialConfig) {
    for axis in 0..3 {
        q[(ACCEL_BIAS + axis, ACCEL_BIAS + axis)] = config.accel_bias.continuous_psd();
        q[(GYRO_BIAS + axis, GYRO_BIAS + axis)] = config.gyro_bias.continuous_psd();
        if let Some(dmc) = &config.dmc {
            q[(DMC + axis, DMC + axis)] = dmc.continuous_psd();
        }
    }
    q[(ALT_BIAS, ALT_BIAS)] = config.baro_bias.continuous_psd();
    q[(MAG_BIAS, MAG_BIAS)] = config.mag_bias.continuous_psd();
    if let Some(clock) = &config.clock_model {
        q[(CLOCK_BIAS, CLOCK_BIAS)] = clock.white_frequency_psd_m2_per_s;
        q[(CLOCK_DRIFT, CLOCK_DRIFT)] = clock.random_walk_frequency_psd_m2_per_s3;
    }
}

/// Sets the initial variance of each bias and of the empirical
/// acceleration to its stationary variance, and the clock variances to
/// the configured initial values.
pub(crate) fn set_initial_variances(p: &mut ErrorMat, config: &InertialConfig) {
    for axis in 0..3 {
        p[(ACCEL_BIAS + axis, ACCEL_BIAS + axis)] = config.accel_bias.stationary_variance();
        p[(GYRO_BIAS + axis, GYRO_BIAS + axis)] = config.gyro_bias.stationary_variance();
        if let Some(dmc) = &config.dmc {
            p[(DMC + axis, DMC + axis)] = dmc.stationary_variance();
        }
    }
    p[(ALT_BIAS, ALT_BIAS)] = config.baro_bias.stationary_variance();
    p[(MAG_BIAS, MAG_BIAS)] = config.mag_bias.stationary_variance();
    if let Some(clock) = &config.clock_model {
        p[(CLOCK_BIAS, CLOCK_BIAS)] = clock.initial_bias_sigma_m * clock.initial_bias_sigma_m;
        p[(CLOCK_DRIFT, CLOCK_DRIFT)] =
            clock.initial_drift_sigma_mps * clock.initial_drift_sigma_mps;
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
