//! Filter configuration with documented defaults.
//!
//! Every bound the admission gates and the integrity assessment apply is
//! configuration, not code: staleness, the assessment window, the
//! innovation gate, process noise, initial uncertainty, and the quality
//! horizons (ADR-0003, ADR-0004).

use navigate_contract::DurationNanos;

/// Configuration of a [`crate::NavigationFilter`].
///
/// The observation clock domain is not configuration: it is fixed at
/// [`crate::NavigationFilter::new`] and every stamp is checked against it
/// at admission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionConfig {
    /// Maximum observation age at admission; anything older is rejected
    /// as stale. Default: 500 ms.
    pub staleness_bound: DurationNanos,
    /// How far back an admitted observation counts toward the integrity
    /// assessment's contributing set. Default: 5 s.
    pub assessment_window: DurationNanos,
    /// Innovation gate threshold on the chi-square statistic with 3
    /// degrees of freedom. Default: 16.27, the 99.9% point of the
    /// chi-square distribution with 3 degrees of freedom.
    pub innovation_gate_chi2: f64,
    /// Power spectral density of the white-acceleration process noise in
    /// m²/s³. Default: 1.0.
    pub process_noise_accel_psd: f64,
    /// Diffuse prior velocity variance per NED axis in m²/s², held until
    /// velocity measurements arrive; position needs no prior because the
    /// first admitted fix's covariance anchors it directly.
    /// Default: 100.0 (10 m/s 1-sigma).
    pub initial_velocity_variance_m2_per_s2: f64,
    /// Horizontal 1-sigma at or below which quality is `Good`.
    /// Default: 10.0 m.
    pub good_horizontal_1sigma_m: f64,
    /// Horizontal 1-sigma at or below which quality is at least
    /// `Degraded`; beyond it the solution is `Unusable`. Default: 50.0 m.
    pub degraded_horizontal_1sigma_m: f64,
    /// Observation silence at or beyond which quality degrades to at
    /// least `Degraded`. Default: 2 s.
    pub quality_silence_degraded: DurationNanos,
    /// Observation silence at or beyond which quality is `Unusable`.
    /// Default: 5 s.
    pub quality_silence_unusable: DurationNanos,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            staleness_bound: DurationNanos::from_millis(500),
            assessment_window: DurationNanos::from_millis(5_000),
            innovation_gate_chi2: 16.27,
            process_noise_accel_psd: 1.0,
            initial_velocity_variance_m2_per_s2: 100.0,
            good_horizontal_1sigma_m: 10.0,
            degraded_horizontal_1sigma_m: 50.0,
            quality_silence_degraded: DurationNanos::from_millis(2_000),
            quality_silence_unusable: DurationNanos::from_millis(5_000),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use navigate_contract::DurationNanos;

    use super::FusionConfig;

    #[test]
    fn documented_defaults_hold() {
        let config = FusionConfig::default();
        assert_eq!(config.staleness_bound, DurationNanos::from_millis(500));
        assert_eq!(config.assessment_window, DurationNanos::from_millis(5_000));
        assert!((config.innovation_gate_chi2 - 16.27).abs() < 1e-12);
        assert!((config.process_noise_accel_psd - 1.0).abs() < 1e-12);
        assert!((config.initial_velocity_variance_m2_per_s2 - 100.0).abs() < 1e-12);
        assert!((config.good_horizontal_1sigma_m - 10.0).abs() < 1e-12);
        assert!((config.degraded_horizontal_1sigma_m - 50.0).abs() < 1e-12);
        assert_eq!(
            config.quality_silence_degraded,
            DurationNanos::from_millis(2_000)
        );
        assert_eq!(
            config.quality_silence_unusable,
            DurationNanos::from_millis(5_000)
        );
    }
}
