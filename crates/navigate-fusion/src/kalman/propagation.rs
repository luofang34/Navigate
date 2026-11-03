//! Propagators of the Kalman core and the first-order Gauss–Markov bias
//! seam (ADR-0008, ADR-0012).

use super::{KalmanState, Mat6};
use crate::float_math::exp;

/// Advances a state of dimension `N` through time (ADR-0008).
///
/// A propagator owns the dynamics of its state layout. The measurement
/// models do not change when a propagator adds states.
pub(crate) trait Propagator<const N: usize> {
    /// The state advanced by `dt_s` seconds. A non-positive `dt_s`
    /// returns the state unchanged.
    fn propagate(&self, state: &KalmanState<N>, dt_s: f64) -> KalmanState<N>;
}

/// Constant-velocity dynamics over the six-state layout with
/// white-acceleration process noise.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConstantVelocityPropagator {
    /// Power spectral density of the acceleration noise in m²/s³.
    pub(crate) accel_psd: f64,
}

impl Propagator<6> for ConstantVelocityPropagator {
    /// Applies `Q_pp = q·dt³/3`, `Q_pv = q·dt²/2`, and `Q_vv = q·dt` per
    /// axis.
    fn propagate(&self, state: &KalmanState<6>, dt_s: f64) -> KalmanState<6> {
        if dt_s <= 0.0 {
            return *state;
        }
        let q_pp = self.accel_psd * dt_s * dt_s * dt_s / 3.0;
        let q_pv = self.accel_psd * dt_s * dt_s / 2.0;
        let q_vv = self.accel_psd * dt_s;
        let mut f = Mat6::identity();
        let mut q = Mat6::zeros();
        for i in 0..3 {
            f[(i, i + 3)] = dt_s;
            q[(i, i)] = q_pp;
            q[(i, i + 3)] = q_pv;
            q[(i + 3, i)] = q_pv;
            q[(i + 3, i + 3)] = q_vv;
        }
        KalmanState {
            x: f * state.x,
            p: f * state.p * f.transpose() + q,
        }
    }
}

/// A first-order Gauss–Markov (FOGM) bias: a scalar error that decays to
/// zero with the correlation time `tau_s` (ADR-0012).
///
/// A bias-carrying propagator adds one bias state for each FOGM error,
/// such as the error of one magnetic map release. The discrete form keeps
/// the steady-state variance at `sigma²` for every step size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogmBias {
    /// Correlation time in seconds.
    pub(crate) tau_s: f64,
    /// Steady-state 1-sigma of the bias, in the unit of the bias.
    pub(crate) sigma: f64,
}

impl FogmBias {
    /// Makes a bias with the correlation time `tau_s` in seconds and the
    /// steady-state 1-sigma `sigma`.
    #[must_use]
    pub const fn new(tau_s: f64, sigma: f64) -> Self {
        Self { tau_s, sigma }
    }

    /// Correlation time in seconds.
    #[must_use]
    pub const fn tau_s(&self) -> f64 {
        self.tau_s
    }

    /// Steady-state 1-sigma, in the unit of the bias.
    #[must_use]
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Steady-state variance `σ²`.
    #[must_use]
    pub const fn stationary_variance(&self) -> f64 {
        self.sigma * self.sigma
    }

    /// Power spectral density `2·σ²/τ` of the continuous white noise
    /// that drives the bias. A non-positive or non-finite `tau_s` gives
    /// 0.0, which holds the bias constant.
    #[must_use]
    pub fn continuous_psd(&self) -> f64 {
        if self.tau_s > 0.0 && self.tau_s.is_finite() {
            2.0 * self.stationary_variance() / self.tau_s
        } else {
            0.0
        }
    }

    /// Continuous decay rate `−1/τ` of the bias. A non-positive or
    /// non-finite `tau_s` gives 0.0.
    #[must_use]
    pub fn decay_rate(&self) -> f64 {
        if self.tau_s > 0.0 && self.tau_s.is_finite() {
            -1.0 / self.tau_s
        } else {
            0.0
        }
    }

    /// Advances the bias mean `x` by `dt_s` seconds. The result is
    /// `(next_mean, process_variance)`. The next mean is `φ·x` with
    /// `φ = exp(−dt/τ)`. The process variance is `σ²·(1 − φ²)`. A
    /// non-positive `dt_s` or a non-positive `tau_s` returns `(x, 0.0)`.
    #[must_use]
    pub fn propagate(&self, x: f64, dt_s: f64) -> (f64, f64) {
        if !(dt_s > 0.0 && self.tau_s > 0.0) {
            return (x, 0.0);
        }
        let phi = exp(-dt_s / self.tau_s);
        let process_variance = self.sigma * self.sigma * (1.0 - exp(-2.0 * dt_s / self.tau_s));
        (phi * x, process_variance)
    }
}

#[cfg(test)]
mod tests {
    use super::FogmBias;

    /// Runs the variance recursion `P' = φ²·P + Q` for `steps` steps.
    fn variance_after(bias: &FogmBias, p0: f64, dt_s: f64, steps: usize) -> f64 {
        let (phi, q) = bias.propagate(1.0, dt_s);
        (0..steps).fold(p0, |p, _| phi * phi * p + q)
    }

    #[test]
    fn mean_decays_with_the_correlation_time() {
        let bias = FogmBias::new(100.0, 3e-9);
        let (mean, q) = bias.propagate(10e-9, 100.0);
        assert!((mean - 10e-9 * (-1.0_f64).exp()).abs() < 1e-24);
        assert!((q - 9e-18 * (1.0 - (-2.0_f64).exp())).abs() < 1e-30);
    }

    #[test]
    fn variance_converges_to_sigma_squared_from_zero_and_from_above() {
        let bias = FogmBias::new(60.0, 3e-9);
        let steady = 9e-18;
        let from_zero = variance_after(&bias, 0.0, 1.0, 2_000);
        let from_above = variance_after(&bias, 100.0 * steady, 1.0, 2_000);
        assert!((from_zero - steady).abs() < 1e-6 * steady, "{from_zero}");
        assert!((from_above - steady).abs() < 1e-6 * steady, "{from_above}");
    }

    #[test]
    fn steady_state_is_a_fixed_point_for_every_step_size() {
        let bias = FogmBias::new(60.0, 2.0);
        for dt_s in [0.01, 0.5, 7.0, 600.0] {
            let next = variance_after(&bias, 4.0, dt_s, 1);
            assert!((next - 4.0).abs() < 1e-12, "dt {dt_s}: {next}");
        }
    }

    #[test]
    fn non_positive_time_or_correlation_time_holds_the_mean() {
        let bias = FogmBias::new(60.0, 2.0);
        assert_eq!(bias.propagate(1.5, 0.0), (1.5, 0.0));
        assert_eq!(bias.propagate(1.5, -1.0), (1.5, 0.0));
        assert_eq!(FogmBias::new(0.0, 2.0).propagate(1.5, 1.0), (1.5, 0.0));
        assert_eq!(FogmBias::new(-5.0, 2.0).propagate(1.5, 1.0), (1.5, 0.0));
    }
}
