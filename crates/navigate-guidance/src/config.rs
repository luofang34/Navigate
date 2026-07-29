//! Admission floors guidance enforces before deriving a setpoint, and
//! the limits each derivation shapes its setpoint within.

use navigate_contract::{DurationNanos, SolutionQuality};

/// Floors a solution must clear before a derivation reads it. Every
/// derivation enforces these, so a solution refused by [`crate::guide`]
/// is refused identically by [`crate::guide_velocity`].
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

/// Admission floors and the speed limits [`crate::guide_velocity`]
/// shapes its velocity setpoint within.
///
/// Gains are inverse seconds: meters of deviation to meters per second
/// of correction. The defaults suit a small multirotor operating in a
/// confined volume; a deployment flying larger legs raises them
/// deliberately.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VelocityGuidanceConfig {
    /// Solution admission, shared verbatim with [`crate::guide`]: one
    /// vocabulary of floors, judged once per derivation.
    pub admission: GuidanceConfig,
    /// Along-track speed commanded outside the arrival slowdown radius,
    /// meters per second. Defaults to 2.0.
    pub cruise_mps: f64,
    /// Ceiling on the composed horizontal speed, meters per second.
    /// Defaults to 2.5. A vector over the ceiling is scaled, never
    /// clipped per axis, so the cap cannot rotate the commanded
    /// direction.
    pub max_horizontal_mps: f64,
    /// Ceiling on the magnitude of the commanded vertical rate, meters
    /// per second. Defaults to 1.0.
    pub max_vertical_mps: f64,
    /// Cross-track correction gain, inverse seconds. Defaults to 0.3:
    /// one meter of lateral deviation asks for 0.3 m/s of correction.
    pub cross_track_gain_per_s: f64,
    /// Vertical correction gain, inverse seconds. Defaults to 0.5.
    pub vertical_gain_per_s: f64,
    /// Distance to the target waypoint inside which along-track speed
    /// scales linearly with the distance remaining, meters. Defaults to
    /// 30.0; a non-positive radius disables the slowdown.
    ///
    /// The slowdown makes an arrival flyable, it does not end a leg:
    /// speed never decays below [`Self::MIN_APPROACH_SPEED_MPS`], and
    /// deciding that a waypoint is captured — or that the plan is
    /// finished — stays `navigate-fpl`'s job.
    pub arrival_slowdown_radius_m: f64,
}

impl VelocityGuidanceConfig {
    /// Speed floor the arrival slowdown never commands below, meters per
    /// second.
    ///
    /// A setpoint decaying toward zero would leave the vehicle drifting
    /// short of the waypoint with guidance still claiming to fly the
    /// leg; holding a floor keeps the command honest about the intent to
    /// arrive. A `cruise_mps` below this floor caps it — guidance never
    /// commands more than the configured cruise speed.
    pub const MIN_APPROACH_SPEED_MPS: f64 = 0.3;

    /// Builds a config from its parts.
    #[must_use]
    pub const fn new(
        admission: GuidanceConfig,
        cruise_mps: f64,
        max_horizontal_mps: f64,
        max_vertical_mps: f64,
        cross_track_gain_per_s: f64,
        vertical_gain_per_s: f64,
        arrival_slowdown_radius_m: f64,
    ) -> Self {
        Self {
            admission,
            cruise_mps,
            max_horizontal_mps,
            max_vertical_mps,
            cross_track_gain_per_s,
            vertical_gain_per_s,
            arrival_slowdown_radius_m,
        }
    }
}

impl Default for VelocityGuidanceConfig {
    /// The per-field defaults documented on [`VelocityGuidanceConfig`],
    /// over [`GuidanceConfig::default`] admission.
    fn default() -> Self {
        Self::new(GuidanceConfig::default(), 2.0, 2.5, 1.0, 0.3, 0.5, 30.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use navigate_contract::{DurationNanos, SolutionQuality};

    use super::{GuidanceConfig, VelocityGuidanceConfig};

    #[test]
    fn default_floor_is_degraded_and_one_second() {
        let config = GuidanceConfig::default();
        assert_eq!(config.minimum_quality, SolutionQuality::Degraded);
        assert_eq!(config.max_solution_age, DurationNanos::from_millis(1_000));
    }

    #[test]
    fn velocity_defaults_are_the_documented_values() {
        let config = VelocityGuidanceConfig::default();
        assert_eq!(config.admission, GuidanceConfig::default());
        assert!((config.cruise_mps - 2.0).abs() < f64::EPSILON);
        assert!((config.max_horizontal_mps - 2.5).abs() < f64::EPSILON);
        assert!((config.max_vertical_mps - 1.0).abs() < f64::EPSILON);
        assert!((config.cross_track_gain_per_s - 0.3).abs() < f64::EPSILON);
        assert!((config.vertical_gain_per_s - 0.5).abs() < f64::EPSILON);
        assert!((config.arrival_slowdown_radius_m - 30.0).abs() < f64::EPSILON);
        assert!((VelocityGuidanceConfig::MIN_APPROACH_SPEED_MPS - 0.3).abs() < f64::EPSILON);
    }
}
