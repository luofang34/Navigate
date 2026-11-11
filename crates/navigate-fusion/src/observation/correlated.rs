//! The declaration of an error that is correlated in time (ADR-0023).

/// The time-correlated part of the error of one source: a first-order
/// Gauss–Markov process with the correlation time `tau_s` and the
/// stationary 1-sigma `stationary_sigma` about three axes. The
/// covariance of the observation holds the white part of the error
/// only.
///
/// The source declares it because only the source knows its error
/// model: a GNSS receiver for its ionosphere, orbit, and clock
/// residuals, a star camera for its mount error. Without a key, the
/// filter keys the process by the source and the epoch of the stamp, so
/// a new source epoch starts a new, independent process.
///
/// With a key ([`Self::with_key`]), the key alone names the process, so
/// fixes of several sources can share it: the anchors of two cameras on
/// one map cell share the registration error of that cell. A constant
/// keyed process (`tau_s = ∞`) can take over the block of an older
/// constant keyed process of the same kind when no block is free. The
/// filter then refuses the older key until the next takeover of that
/// block, and the caller must not offer a key that lost its block before
/// that: the filter does not hold the error of a key that it no longer
/// binds.
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
    /// The key of the process, or `None` for the process of the source
    /// and its epoch.
    pub key: Option<u64>,
}

impl CorrelatedError {
    /// A process with the correlation time `tau_s` and the stationary
    /// 1-sigma `stationary_sigma`.
    #[must_use]
    pub const fn new(tau_s: f64, stationary_sigma: [f64; 3]) -> Self {
        Self {
            tau_s,
            stationary_sigma,
            key: None,
        }
    }

    /// This process with the key `key`.
    #[must_use]
    pub const fn with_key(mut self, key: u64) -> Self {
        self.key = Some(key);
        self
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
