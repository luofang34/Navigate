//! The two-state receiver clock model (ADR-0018).
//!
//! The clock states are the range bias `c·δt` in meters and the
//! range-rate drift `c·δṫ` in m/s. White frequency noise drives the bias
//! and random-walk frequency noise drives the drift. Over `Δt` the
//! transition and the discrete noise are
//!
//! - `Φ = [[1, Δt], [0, 1]]`
//! - `Q = [[S_f·Δt + S_g·Δt³/3, S_g·Δt²/2], [S_g·Δt²/2, S_g·Δt]]`
//!
//! The filter gets the same `Φ` and `Q` from its Van Loan step, because
//! the clock block of the continuous dynamics is `[[0, 1], [0, 0]]` with
//! the noise density `diag(S_f, S_g)`.

use core::f64::consts::PI;

/// Speed of light in vacuum in m/s.
pub(crate) const SPEED_OF_LIGHT_MPS: f64 = 299_792_458.0;

/// Noise model and initial uncertainty of the receiver clock states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockNoiseModel {
    /// White frequency noise density `S_f` in m²/s. It drives the range
    /// bias.
    pub white_frequency_psd_m2_per_s: f64,
    /// Random-walk frequency noise density `S_g` in m²/s³. It drives the
    /// range-rate drift.
    pub random_walk_frequency_psd_m2_per_s3: f64,
    /// Initial 1-sigma of the range bias `c·δt` in meters.
    pub initial_bias_sigma_m: f64,
    /// Initial 1-sigma of the range-rate drift `c·δṫ` in m/s.
    pub initial_drift_sigma_mps: f64,
}

impl ClockNoiseModel {
    /// The model from the power-law coefficients `h₀` and `h₋₂` of the
    /// fractional frequency. They have no unit. The densities are
    /// `S_f = c²·h₀/2` and `S_g = 2π²·c²·h₋₂`.
    #[must_use]
    pub fn from_allan_coefficients(
        h0: f64,
        h_minus_2: f64,
        initial_bias_sigma_m: f64,
        initial_drift_sigma_mps: f64,
    ) -> Self {
        let c2 = SPEED_OF_LIGHT_MPS * SPEED_OF_LIGHT_MPS;
        Self {
            white_frequency_psd_m2_per_s: c2 * h0 * 0.5,
            random_walk_frequency_psd_m2_per_s3: 2.0 * PI * PI * c2 * h_minus_2,
            initial_bias_sigma_m,
            initial_drift_sigma_mps,
        }
    }

    /// Whether each density is finite and not negative, and each initial
    /// 1-sigma is finite and positive.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let density = |value: f64| value.is_finite() && value >= 0.0;
        let sigma = |value: f64| value.is_finite() && value > 0.0;
        density(self.white_frequency_psd_m2_per_s)
            && density(self.random_walk_frequency_psd_m2_per_s3)
            && sigma(self.initial_bias_sigma_m)
            && sigma(self.initial_drift_sigma_mps)
    }

    /// The transition `Φ` and the discrete noise `Q` of the two clock
    /// states over `dt_s` seconds, as row-major 2×2 arrays.
    #[must_use]
    pub fn discrete(&self, dt_s: f64) -> ([[f64; 2]; 2], [[f64; 2]; 2]) {
        let s_f = self.white_frequency_psd_m2_per_s;
        let s_g = self.random_walk_frequency_psd_m2_per_s3;
        let dt2 = dt_s * dt_s;
        let cross = 0.5 * s_g * dt2;
        (
            [[1.0, dt_s], [0.0, 1.0]],
            [
                [s_f * dt_s + s_g * dt2 * dt_s / 3.0, cross],
                [cross, s_g * dt_s],
            ],
        )
    }
}
