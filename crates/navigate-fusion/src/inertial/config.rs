//! Configuration of the inertial navigation filter (ADR-0014,
//! ADR-0018).

use super::clock::ClockNoiseModel;
use super::history::MAX_INERTIAL_HISTORY;
use super::layout::StateLayout;
use crate::config::FusionConfig;
use crate::kalman::FogmBias;

/// Number of history events that the default configuration keeps: the
/// inline bound. One event is one IMU sample, one IMU increment, or one
/// observation, so a delayed observation must arrive within this number
/// of events.
const DEFAULT_HISTORY_CAPACITY: usize = MAX_INERTIAL_HISTORY;

/// Configuration of a [`crate::InertialNavigationFilter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialConfig {
    /// The admission, gate, quality, and history settings.
    /// `history_capacity` counts IMU events and observations, up to
    /// [`super::MAX_INERTIAL_HISTORY`].
    pub fusion: FusionConfig,
    /// White noise density of the accelerometer in (m/s²)/√Hz.
    pub accel_noise_sigma_mps2_sqrt_hz: f64,
    /// White noise density of the gyroscope in (rad/s)/√Hz.
    pub gyro_noise_sigma_rad_s_sqrt_hz: f64,
    /// First-order Gauss–Markov model of each accelerometer bias in m/s².
    pub accel_bias: FogmBias,
    /// First-order Gauss–Markov model of each gyroscope bias in rad/s.
    pub gyro_bias: FogmBias,
    /// First-order Gauss–Markov model of the barometric altitude bias in
    /// meters.
    pub baro_bias: FogmBias,
    /// First-order Gauss–Markov model of the scalar magnetic map bias in
    /// tesla.
    pub mag_bias: FogmBias,
    /// Lowest initial attitude 1-sigma about each body axis in radians.
    /// The initializer raises a smaller diagonal variance to this floor,
    /// so an overconfident attitude prior cannot freeze the attitude
    /// states.
    pub initial_attitude_sigma_rad: [f64; 3],
    /// First-order Gauss–Markov model of each axis of the empirical
    /// acceleration of the dynamic model compensation in m/s², in the
    /// frame axes. The force-model propagation adds this acceleration to
    /// the modeled forces. `None` keeps the three states at zero without
    /// variance.
    pub dmc: Option<FogmBias>,
    /// Model of the receiver clock range bias and range-rate drift.
    /// The one-way radiometric models need it. `None` keeps the two
    /// states at zero without variance.
    pub clock_model: Option<ClockNoiseModel>,
    /// The blocks of the error state and their offsets. Its dimension is
    /// the dimension of the filter; the constructor refuses a layout that
    /// does not fit it.
    pub layout: StateLayout,
}

impl Default for InertialConfig {
    /// A navigation-grade IMU and the magnetic map bias of a crustal
    /// anomaly map. The accelerometer has 1e-4 (m/s²)/√Hz noise and a
    /// 2.5e-4 m/s² bias with a correlation time of one hour. The
    /// gyroscope has 5.8e-7 (rad/s)/√Hz noise and a 4.85e-8 rad/s bias
    /// with a correlation time of one hour. The barometric bias is 3 m
    /// with a correlation time of one hour. The magnetic bias is 3 nT
    /// with a correlation time of 600 s. The configuration has no
    /// empirical acceleration and no clock states.
    fn default() -> Self {
        Self {
            fusion: FusionConfig {
                history_capacity: DEFAULT_HISTORY_CAPACITY,
                ..FusionConfig::default()
            },
            accel_noise_sigma_mps2_sqrt_hz: 1e-4,
            gyro_noise_sigma_rad_s_sqrt_hz: 5.8e-7,
            accel_bias: FogmBias::new(3_600.0, 2.5e-4),
            gyro_bias: FogmBias::new(3_600.0, 4.85e-8),
            baro_bias: FogmBias::new(3_600.0, 3.0),
            mag_bias: FogmBias::new(600.0, 3e-9),
            initial_attitude_sigma_rad: [1e-6; 3],
            dmc: None,
            clock_model: None,
            layout: StateLayout::FULL,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InertialConfig, MAX_INERTIAL_HISTORY};

    #[test]
    fn documented_defaults_hold() {
        let config = InertialConfig::default();
        assert_eq!(config.fusion.history_capacity, MAX_INERTIAL_HISTORY);
        assert!((config.mag_bias.tau_s() - 600.0).abs() < 1e-12);
        assert!((config.mag_bias.sigma() - 3e-9).abs() < 1e-24);
        assert!((config.baro_bias.sigma() - 3.0).abs() < 1e-12);
        assert!((config.accel_noise_sigma_mps2_sqrt_hz - 1e-4).abs() < 1e-18);
        assert!((config.gyro_noise_sigma_rad_s_sqrt_hz - 5.8e-7).abs() < 1e-20);
        assert_eq!(config.initial_attitude_sigma_rad, [1e-6; 3]);
        assert_eq!(config.dmc, None);
        assert_eq!(config.clock_model, None);
    }
}
