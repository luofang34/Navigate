//! The integrity of GNSS against the other sensors of a suite: the
//! configuration of the check and the status that a display reads.
//!
//! A receiver that an emitter deceives gives positions that leave the
//! truth slowly. Each position agrees with the position before it, so a
//! filter that takes them stays close to them and its innovation gate
//! stays open. The check compares each GNSS position with a reference
//! solution that never takes a GNSS value.

use navigate_contract::{MonotonicNanos, SourceId};

/// How a navigator checks the GNSS positions.
///
/// The statistic of one fix is `dᵀ·S⁻¹·d`. `d` is the horizontal
/// separation of the fix from the reference solution at the acquisition
/// time. `S` is the sum of the horizontal covariances of the reference
/// position, of the white error of the fix, and of the declared
/// time-correlated error of the receiver. With an honest receiver the
/// statistic has the chi-square distribution with 2 degrees of freedom,
/// whose upper tail is `e^(−x/2)`: a probability `p` gives the threshold
/// `−2·ln(p)`.
///
/// Successive fixes meet almost the same reference error, so their
/// statistics are not independent. Each probability is that of one fix,
/// not a rate for each hour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnssIntegrityConfig {
    /// Probability that the statistic of one honest fix is above the
    /// caution threshold. Default: `1e-3`, the threshold 13.8, a
    /// separation of 3.7 sigma on the axis of the largest variance.
    pub caution_probability: f64,
    /// Probability that the statistic of one honest fix is above the
    /// exclusion threshold. Default: `1e-6`, the threshold 27.6, a
    /// separation of 5.3 sigma.
    pub exclusion_probability: f64,
    /// Fixes in a row above a threshold that change the state for the
    /// worse: above the caution threshold to `Disagreeing`, above the
    /// exclusion threshold to `Excluded`. The white error of a receiver
    /// is independent from fix to fix, so one fix of an honest receiver
    /// is above the caution threshold about once in a thousand, and the
    /// innovation gate of the filter refuses a wild one; a deceived
    /// receiver is above the threshold on each fix. Default: 3.
    pub confirming_fixes: u32,
    /// Time of agreement before the state gets better, seconds: from
    /// `Disagreeing` to `Consistent`, and from `Excluded` to
    /// `Consistent`. The agreement ends with `confirming_fixes` fixes in a
    /// row above the caution threshold. Default: 30 s. In that time a
    /// receiver that still leaves the truth at 0.5 m/s moves 15 m, which
    /// is of the order of the caution bound of a reference with terrain
    /// fixes, so it does not pass. The navigator takes an excluded
    /// receiver again only when the reference also took a position fix of
    /// another sensor in that time. A reference on inertial data alone
    /// has a bound that grows until it holds any offset, so it cannot say
    /// that a receiver is honest again: the host then aligns the
    /// navigator again.
    pub readmission_s: f64,
    /// Largest exclusion bound at which the check has a use, metres.
    /// Above it the reference says too little: a navigator that still
    /// takes GNSS starts the reference again from its solution, and a
    /// navigator that excluded GNSS does not take GNSS again. Default:
    /// 556 m, 0.3 nautical miles, the lateral limit of an RNP 0.3
    /// approach.
    pub alert_limit_m: f64,
    /// When a reference that holds its position on inertial data alone
    /// starts again from the solution of the navigator. Its exclusion
    /// bound over the time since its start is the slowest receiver drift
    /// that it finds; that rate falls, has a lowest value, and rises with
    /// the inertial drift. The reference starts again when the rate is
    /// this many times its lowest value. Default: 2. At 1 a drift that
    /// begins in the middle of the time has too little of it left; a
    /// large value leaves a reference whose bound is far above the drift
    /// that it is to find.
    pub restart_rate_ratio: f64,
}

impl Default for GnssIntegrityConfig {
    fn default() -> Self {
        Self {
            caution_probability: 1e-3,
            exclusion_probability: 1e-6,
            confirming_fixes: 3,
            readmission_s: 30.0,
            alert_limit_m: 556.0,
            restart_rate_ratio: 2.0,
        }
    }
}

impl GnssIntegrityConfig {
    /// Whether each probability is in `(0, 1)`, the exclusion is not more
    /// probable than the caution, the count is not zero, the time and the
    /// limit are finite and positive, and the ratio is finite and not
    /// below 1.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        let probability = |p: f64| p > 0.0 && p < 1.0;
        probability(self.caution_probability)
            && probability(self.exclusion_probability)
            && self.exclusion_probability <= self.caution_probability
            && self.confirming_fixes > 0
            && self.readmission_s.is_finite()
            && self.readmission_s > 0.0
            && self.alert_limit_m.is_finite()
            && self.alert_limit_m > 0.0
            && self.restart_rate_ratio.is_finite()
            && self.restart_rate_ratio >= 1.0
    }

    /// The caution threshold of the statistic.
    #[must_use]
    pub fn caution_chi2(&self) -> f64 {
        -2.0 * self.caution_probability.ln()
    }

    /// The exclusion threshold of the statistic.
    #[must_use]
    pub fn exclusion_chi2(&self) -> f64 {
        -2.0 * self.exclusion_probability.ln()
    }
}

/// What the navigator does with GNSS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GnssIntegrityState {
    /// The fixes are below the caution threshold. The navigator takes
    /// them, and the check changes nothing in the solution.
    Consistent,
    /// The fixes are above the caution threshold. The navigator still
    /// takes GNSS. The solution states the separation in its covariance, and
    /// its quality is not `Good`.
    Disagreeing,
    /// The navigator does not take GNSS. The solution continues from the
    /// reference, on the other sensors.
    Excluded,
}

/// Why GNSS is not [`GnssIntegrityState::Consistent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GnssFault {
    /// The horizontal position of the receiver is too far from the
    /// reference solution.
    HorizontalSeparation,
}

/// One GNSS position against the reference solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnssSeparation {
    /// The receiver.
    pub source: SourceId,
    /// Acquisition time of the fix.
    pub at: MonotonicNanos,
    /// Horizontal distance of the fix from the reference solution,
    /// metres.
    pub separation_m: f64,
    /// The statistic of the fix.
    pub chi2: f64,
    /// Separation along the axis of the largest variance at which the
    /// statistic is at the caution threshold, metres.
    pub caution_bound_m: f64,
    /// Separation along the axis of the largest variance at which the
    /// statistic is at the exclusion threshold, metres. A receiver error
    /// below it is not found.
    pub exclusion_bound_m: f64,
    /// Horizontal one-sigma of the reference solution on the axis of its
    /// largest variance, metres.
    pub reference_1sigma_m: f64,
}

/// The state of the GNSS check, for a display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnssIntegrityStatus {
    /// What the navigator does with GNSS.
    pub state: GnssIntegrityState,
    /// The acquisition time of the fix that started the state, or `None`
    /// for the state from the start of the check.
    pub since: Option<MonotonicNanos>,
    /// Why the state is not `Consistent`, or `None` when it is.
    pub fault: Option<GnssFault>,
    /// The last fix that the navigator compared, with its receiver, its
    /// separation, and the bounds. `None` before the first one.
    pub last: Option<GnssSeparation>,
    /// Whether the exclusion bound of the last fix is at or below the
    /// alert limit. Without it the check says nothing about the
    /// receiver.
    pub within_alert_limit: bool,
    /// When the reference last took its state from the solution that
    /// includes GNSS: at the alignment, at the start of the check, or
    /// when it said too little. `None` before the first state.
    pub reference_anchored_at: Option<MonotonicNanos>,
    /// Acquisition time of the last position fix of another sensor that
    /// the reference took after that, for example a terrain fix. With
    /// `None` the reference holds its position on inertial data alone,
    /// and the check finds only an error that grows faster than the
    /// inertial drift.
    pub reference_position_fix_at: Option<MonotonicNanos>,
    /// For a reference on inertial data alone: the slowest receiver drift
    /// that it finds, m/s, the lowest exclusion bound over the time since
    /// its start. A slower drift moves the solution and is not found.
    /// `None` with a position fix of another sensor, and before the first
    /// compared fix.
    pub inertial_detectable_rate_m_s: Option<f64>,
    /// While the state is not `Consistent`: the acquisition time of the
    /// first fix of the present time of agreement.
    pub agreeing_since: Option<MonotonicNanos>,
}
