//! Admission floors guidance enforces before deriving a setpoint.

use navigate_contract::{DurationNanos, SolutionQuality};

/// Floors a solution must clear before [`crate::guide`] derives a
/// setpoint from it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GuidanceConfig {
    /// Minimum solution quality guidance accepts, under the ordering
    /// `Good` > `Degraded` > `Unusable`.
    ///
    /// Defaults to [`SolutionQuality::Degraded`]: `Unusable` solutions
    /// are refused while degraded ones still guide. Requiring `Good`
    /// is a per-deployment tightening for maneuvers that need it, not
    /// the baseline.
    pub minimum_quality: SolutionQuality,
    /// Maximum solution age at guidance time (`now` minus the stamp's
    /// `solved_at`). Defaults to one second.
    pub max_solution_age: DurationNanos,
}

impl GuidanceConfig {
    /// Builds a config from its parts.
    #[must_use]
    pub const fn new(minimum_quality: SolutionQuality, max_solution_age: DurationNanos) -> Self {
        Self {
            minimum_quality,
            max_solution_age,
        }
    }
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self::new(SolutionQuality::Degraded, DurationNanos::from_millis(1_000))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use navigate_contract::{DurationNanos, SolutionQuality};

    use super::GuidanceConfig;

    #[test]
    fn default_floor_is_degraded_and_one_second() {
        let config = GuidanceConfig::default();
        assert_eq!(config.minimum_quality, SolutionQuality::Degraded);
        assert_eq!(config.max_solution_age, DurationNanos::from_millis(1_000));
    }
}
