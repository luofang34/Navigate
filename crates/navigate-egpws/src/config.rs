//! Alert floors and the admission floors for terrain assessment.

use navigate_contract::{DurationNanos, SolutionQuality};

/// Configuration for [`crate::assess`].
///
/// Both alert floors are heights above terrain in meters; a smaller
/// height is closer to terrain, and each floor is crossed downward. The
/// warning floor is expected to sit below the caution floor so
/// severities escalate as terrain closes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EgpwsConfig {
    /// Height above terrain in meters below which the assessment raises
    /// [`crate::AlertSeverity::Caution`].
    pub caution_floor_m: f64,
    /// Height above terrain in meters below which the assessment raises
    /// [`crate::AlertSeverity::Warning`], superseding a caution.
    pub warning_floor_m: f64,
    /// Lowest [`SolutionQuality`] the assessment accepts; a solution
    /// below this floor is refused with
    /// [`crate::EgpwsUnavailable::SolutionIntegrity`] rather than
    /// scored against terrain (ADR-0004).
    pub minimum_quality: SolutionQuality,
    /// Maximum solution age at assessment time (`now` minus the stamp's
    /// `solved_at`); an older solution is refused with
    /// [`crate::EgpwsUnavailable::SolutionStale`] rather than scored
    /// against terrain. Defaults to one second.
    pub max_solution_age: DurationNanos,
}

impl Default for EgpwsConfig {
    /// Caution below 150 m above terrain, warning below 60 m, `Degraded`
    /// solutions admitted — terrain proximity is worth stating even from
    /// a degraded solution, while `Unusable` is not a position at all —
    /// and solutions older than one second refused as stale.
    fn default() -> Self {
        Self {
            caution_floor_m: 150.0,
            warning_floor_m: 60.0,
            minimum_quality: SolutionQuality::Degraded,
            max_solution_age: DurationNanos::from_millis(1_000),
        }
    }
}
