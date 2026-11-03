//! Configuration of the inertial navigation filter (ADR-0014).

use crate::config::FusionConfig;
use crate::filter::FogmBias;

/// Number of history events that the default configuration keeps. One
/// event is one IMU sample or one observation, so the history must hold
/// more events than the IMU rate times the staleness bound.
const DEFAULT_HISTORY_CAPACITY: usize = 512;

/// Configuration of a [`crate::InertialNavigationFilter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialConfig {
    /// The admission, gate, quality, and history settings.
    /// `history_capacity` counts IMU samples and observations.
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
}

impl Default for InertialConfig {
    /// A navigation-grade IMU and the magnetic map bias of a crustal
    /// anomaly map. The accelerometer has 1e-4 (m/s²)/√Hz noise and a
    /// 2.5e-4 m/s² bias with a correlation time of one hour. The
    /// gyroscope has 5.8e-7 (rad/s)/√Hz noise and a 4.85e-8 rad/s bias
    /// with a correlation time of one hour. The barometric bias is 3 m
    /// with a correlation time of one hour. The magnetic bias is 3 nT
    /// with a correlation time of 600 s.
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InertialConfig;

    #[test]
    fn documented_defaults_hold() {
        let config = InertialConfig::default();
        assert_eq!(config.fusion.history_capacity, 512);
        assert!((config.mag_bias.tau_s() - 600.0).abs() < 1e-12);
        assert!((config.mag_bias.sigma() - 3e-9).abs() < 1e-24);
        assert!((config.baro_bias.sigma() - 3.0).abs() < 1e-12);
        assert!((config.accel_noise_sigma_mps2_sqrt_hz - 1e-4).abs() < 1e-18);
        assert!((config.gyro_noise_sigma_rad_s_sqrt_hz - 5.8e-7).abs() < 1e-20);
        assert_eq!(config.initial_attitude_sigma_rad, [1e-6; 3]);
    }
}
