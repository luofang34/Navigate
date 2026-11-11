//! The declaration of an error that is correlated in time (ADR-0023).

/// The time-correlated part of the error of one source: a first-order
/// Gauss–Markov process with the correlation time `tau_s` and the
/// stationary 1-sigma `stationary_sigma` about three axes. The
/// covariance of the observation holds the white part of the error
/// only.
///
/// The source declares it because only the source knows its error
/// model: a GNSS receiver for its ionosphere, orbit, and clock
/// residuals, a star camera for its mount error. The filter keys the
/// process by the source and the epoch of the stamp, so a new source
/// epoch starts a new, independent process.
///
/// The axes of `stationary_sigma` are the axes of the observed
/// quantity:
///
/// - a geodetic position fix: the local north, east, and down axes;
/// - a Cartesian position fix: the frame axes;
/// - an attitude fix or a line of sight: the body axes. The process is
///   a rotation of the reference of the sensor, such as a mount error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrelatedError {
    /// The correlation time in seconds. `f64::INFINITY` declares a
    /// constant error, such as a mount error or the error of one map
    /// cell.
    pub tau_s: f64,
    /// The stationary 1-sigma about each axis.
    pub stationary_sigma: [f64; 3],
}

impl CorrelatedError {
    /// A process with the correlation time `tau_s` and the stationary
    /// 1-sigma `stationary_sigma`.
    #[must_use]
    pub const fn new(tau_s: f64, stationary_sigma: [f64; 3]) -> Self {
        Self {
            tau_s,
            stationary_sigma,
        }
    }

    /// Whether the correlation time is positive and each 1-sigma is
    /// finite and not negative.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.tau_s > 0.0
            && !self.tau_s.is_nan()
            && self
                .stationary_sigma
                .iter()
                .all(|sigma| sigma.is_finite() && *sigma >= 0.0)
    }

    /// The correlation coefficient `ρ = e^(−Δt/τ)` of two errors
    /// `dt_s` apart. It is one for a constant error.
    #[must_use]
    pub fn correlation(&self, dt_s: f64) -> f64 {
        if self.tau_s.is_infinite() {
            return 1.0;
        }
        crate::float_math::exp(-dt_s / self.tau_s)
    }
}
