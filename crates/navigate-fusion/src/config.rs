//! Filter configuration with documented defaults.
//!
//! Every bound the admission gates and the integrity assessment apply is
//! configuration, not code: staleness, the assessment window, the
//! innovation gate, process noise, initial uncertainty, and the quality
//! horizons (ADR-0003, ADR-0004). The scalar map-model limits and the
//! depth of the delayed-measurement history are also configuration
//! (ADR-0012). The frame discipline is configuration (ADR-0013).

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
    /// Innovation gate threshold on the chi-square statistic with 1 degree
    /// of freedom. The filter applies it to scalar measurements: range,
    /// altitude, and magnetic anomaly. Default: 10.83, the 99.9% point.
    /// This is the same confidence as `innovation_gate_chi2`.
    pub scalar_gate_chi2: f64,
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
    /// Vertical 1-sigma limit in meters for `Good` solution quality.
    /// Default: 15.0 m.
    pub good_vertical_1sigma_m: f64,
    /// Vertical 1-sigma limit in meters for `Degraded` solution quality.
    /// Values above this limit produce `Unusable` quality. Default: 75.0 m.
    pub degraded_vertical_1sigma_m: f64,
    /// Horizontal offset threshold in meters that triggers tangent-plane
    /// re-anchoring. A non-positive value disables re-anchoring.
    /// Default: 50_000.0 m.
    pub reanchor_distance_m: f64,
    /// Observation silence at or beyond which quality degrades to at
    /// least `Degraded`. Default: 2 s.
    pub quality_silence_degraded: DurationNanos,
    /// Observation silence at or beyond which quality is `Unusable`.
    /// Default: 5 s.
    pub quality_silence_unusable: DurationNanos,
    /// Number of horizontal 1-sigma values that the filter adds to the
    /// offset from a map linearization point. The sum must not exceed the
    /// valid radius of the map model. A negative or non-finite value
    /// makes every map-model observation fail this check. Default: 2.0.
    pub linearization_sigma_multiplier: f64,
    /// Lowest information ratio that a map-model gradient must supply.
    /// The ratio is `gᵀ P_pos g / R`, which has no unit. A negative or
    /// non-finite value makes every map-model observation fail this
    /// check. Default: 1e-3.
    pub min_information_ratio: f64,
    /// Number of admitted updates that the delayed-measurement history
    /// keeps. A value of 0 disables delayed-measurement replay.
    /// Default: 64.
    pub history_capacity: usize,
    /// Whether every observation must declare its frame (ADR-0013). When
    /// it is `true`, the filter refuses an observation without a frame
    /// with `RejectionReason::MissingFrame`. Default: `false`.
    /// [`crate::NavigationFilter::new_for_ellipsoid`] sets it to `true`.
    pub require_explicit_frame: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            staleness_bound: DurationNanos::from_millis(500),
            assessment_window: DurationNanos::from_millis(5_000),
            innovation_gate_chi2: 16.27,
            scalar_gate_chi2: 10.83,
            process_noise_accel_psd: 1.0,
            initial_velocity_variance_m2_per_s2: 100.0,
            good_horizontal_1sigma_m: 10.0,
            degraded_horizontal_1sigma_m: 50.0,
            good_vertical_1sigma_m: 15.0,
            degraded_vertical_1sigma_m: 75.0,
            reanchor_distance_m: 50_000.0,
            quality_silence_degraded: DurationNanos::from_millis(2_000),
            quality_silence_unusable: DurationNanos::from_millis(5_000),
            linearization_sigma_multiplier: 2.0,
            min_information_ratio: 1e-3,
            history_capacity: 64,
            require_explicit_frame: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use navigate_contract::DurationNanos;

    use super::FusionConfig;

    #[test]
    fn documented_defaults_hold() {
        let config = FusionConfig::default();
        assert_eq!(config.staleness_bound, DurationNanos::from_millis(500));
        assert_eq!(config.assessment_window, DurationNanos::from_millis(5_000));
        assert!((config.innovation_gate_chi2 - 16.27).abs() < 1e-12);
        assert!((config.scalar_gate_chi2 - 10.83).abs() < 1e-12);
        assert!((config.process_noise_accel_psd - 1.0).abs() < 1e-12);
        assert!((config.initial_velocity_variance_m2_per_s2 - 100.0).abs() < 1e-12);
        assert!((config.good_horizontal_1sigma_m - 10.0).abs() < 1e-12);
        assert!((config.degraded_horizontal_1sigma_m - 50.0).abs() < 1e-12);
        assert!((config.good_vertical_1sigma_m - 15.0).abs() < 1e-12);
        assert!((config.degraded_vertical_1sigma_m - 75.0).abs() < 1e-12);
        assert!((config.reanchor_distance_m - 50_000.0).abs() < 1e-12);
        assert_eq!(
            config.quality_silence_degraded,
            DurationNanos::from_millis(2_000)
        );
        assert_eq!(
            config.quality_silence_unusable,
            DurationNanos::from_millis(5_000)
        );
        assert!((config.linearization_sigma_multiplier - 2.0).abs() < 1e-12);
        assert!((config.min_information_ratio - 1e-3).abs() < 1e-15);
        assert_eq!(config.history_capacity, 64);
        assert!(!config.require_explicit_frame);
    }
}
