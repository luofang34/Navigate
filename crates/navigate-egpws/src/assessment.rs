//! Terrain-clearance assessment with typed refusals.

use navigate_contract::{
    ClockDomainId, DurationNanos, MonotonicNanos, NavigationSolution, SolutionQuality,
};

use crate::config::EgpwsConfig;
use crate::terrain::TerrainDatabase;

/// Severity of a terrain-proximity alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertSeverity {
    /// Height above terrain is below the configured caution floor.
    Caution,
    /// Height above terrain is below the configured warning floor.
    Warning,
}

/// The outcome of one terrain-clearance assessment.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EgpwsAssessment {
    /// Ownship height above the terrain elevation at the solution
    /// position, in meters. Negative means the solution places ownship
    /// below the terrain surface.
    pub height_above_terrain_m: f64,
    /// The alert the configured floors raise, or `None` when clear.
    pub severity: Option<AlertSeverity>,
}

impl EgpwsAssessment {
    /// Builds an assessment from its parts.
    #[must_use]
    pub const fn new(height_above_terrain_m: f64, severity: Option<AlertSeverity>) -> Self {
        Self {
            height_above_terrain_m,
            severity,
        }
    }
}

/// Why a terrain assessment is refused instead of scored (ADR-0004: a
/// missing input is stated, never substituted).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum EgpwsUnavailable {
    /// No terrain database is bound; the host side has not supplied a
    /// terrain package.
    #[error("terrain assessment unavailable: no terrain database is bound")]
    NoDatabase,
    /// `now` was read on a different clock domain than the solution's
    /// stamp, so the two readings are not comparable and solution age
    /// cannot be judged: readings from different domains are never
    /// subtracted. Refused before any age arithmetic runs.
    #[error(
        "terrain assessment unavailable: now was read on clock domain {}, \
         the solution's stamp is on domain {}",
        got.get(),
        expected.get()
    )]
    ClockDomainMismatch {
        /// Clock domain the solution's stamp carries.
        expected: ClockDomainId,
        /// Clock domain the caller read `now` from.
        got: ClockDomainId,
    },
    /// `now` is earlier than the solution's `solved_at`, so age cannot
    /// be judged: the caller reused an old reading. Refusing surfaces
    /// the misuse instead of saturating it into a fake zero age.
    #[error(
        "terrain assessment unavailable: now {} ns is earlier than solved_at {} ns",
        now.as_nanos(),
        solved_at.as_nanos()
    )]
    ClockInversion {
        /// Caller-supplied assessment time.
        now: MonotonicNanos,
        /// Solution reference time from the stamp.
        solved_at: MonotonicNanos,
    },
    /// The solution is older than the configured bound; terrain
    /// clearance from a stale position would be a false assurance.
    #[error(
        "terrain assessment unavailable: solution age {} ns exceeds the configured bound {} ns",
        age.as_nanos(),
        bound.as_nanos()
    )]
    SolutionStale {
        /// Solution age at assessment time.
        age: DurationNanos,
        /// Configured maximum age.
        bound: DurationNanos,
    },
    /// The solution's quality classification is below the configured
    /// minimum; terrain clearance from a position that weak would be a
    /// false assurance.
    #[error(
        "terrain assessment unavailable: solution quality {quality:?} is below \
         the configured minimum {minimum:?}"
    )]
    SolutionIntegrity {
        /// Quality the refused solution carries.
        quality: SolutionQuality,
        /// Configured minimum quality the solution failed to meet.
        minimum: SolutionQuality,
    },
    /// The bound database has no coverage at the solution position.
    #[error("terrain assessment unavailable: no terrain coverage at the solution position")]
    NoCoverage,
}

/// Assesses ownship height above terrain against the configured floors.
///
/// `now` must be read on the solution's clock domain
/// (`solution.stamp.clock`): readings from different domains are never
/// subtracted, so a mismatched `now_clock` is refused before any age
/// arithmetic — the same admission discipline guidance applies.
///
/// The height is the solution altitude minus the database elevation at
/// the solution position; see the crate-level docs for the open vertical
/// datum question that subtraction carries. Severity is `Warning` below
/// `warning_floor_m`, `Caution` below `caution_floor_m`, `None`
/// otherwise; each floor is an open bound, so a height exactly at a
/// floor does not raise that floor's alert.
///
/// # Errors
///
/// Judged in this order:
///
/// - [`EgpwsUnavailable::NoDatabase`] when `database` is `None`;
/// - [`EgpwsUnavailable::ClockDomainMismatch`] when `now_clock` differs
///   from the solution stamp's clock domain;
/// - [`EgpwsUnavailable::ClockInversion`] when `now` is earlier than the
///   solution's `solved_at`;
/// - [`EgpwsUnavailable::SolutionStale`] when the solution's age exceeds
///   `config.max_solution_age`;
/// - [`EgpwsUnavailable::SolutionIntegrity`] when the solution's quality
///   is below `config.minimum_quality`;
/// - [`EgpwsUnavailable::NoCoverage`] when the database has no elevation
///   at the solution position.
pub fn assess(
    solution: &NavigationSolution,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    database: Option<&dyn TerrainDatabase>,
    config: &EgpwsConfig,
) -> Result<EgpwsAssessment, EgpwsUnavailable> {
    let Some(database) = database else {
        return Err(EgpwsUnavailable::NoDatabase);
    };
    let expected = solution.stamp.clock;
    if now_clock != expected {
        return Err(EgpwsUnavailable::ClockDomainMismatch {
            expected,
            got: now_clock,
        });
    }
    let solved_at = solution.stamp.solved_at;
    let age = now
        .elapsed_since(solved_at)
        .ok_or(EgpwsUnavailable::ClockInversion { now, solved_at })?;
    if age > config.max_solution_age {
        return Err(EgpwsUnavailable::SolutionStale {
            age,
            bound: config.max_solution_age,
        });
    }
    let quality = solution.integrity.quality;
    if quality_rank(quality) < quality_rank(config.minimum_quality) {
        return Err(EgpwsUnavailable::SolutionIntegrity {
            quality,
            minimum: config.minimum_quality,
        });
    }
    let Some(elevation_m) = database.elevation_m(&solution.position) else {
        return Err(EgpwsUnavailable::NoCoverage);
    };
    let height_above_terrain_m = solution.position.altitude_m - elevation_m;
    let severity = if height_above_terrain_m < config.warning_floor_m {
        Some(AlertSeverity::Warning)
    } else if height_above_terrain_m < config.caution_floor_m {
        Some(AlertSeverity::Caution)
    } else {
        None
    };
    Ok(EgpwsAssessment::new(height_above_terrain_m, severity))
}

/// Orders quality for the floor comparison. The enum is non-exhaustive
/// upstream, so an unrecognized classification ranks lowest and fails
/// closed as insufficient.
const fn quality_rank(quality: SolutionQuality) -> u8 {
    match quality {
        SolutionQuality::Good => 2,
        SolutionQuality::Degraded => 1,
        SolutionQuality::Unusable => 0,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use navigate_contract::{
        ClockDomainId, DurationNanos, FaultDetection, GeodeticPosition, IntegrityAssessment,
        MonotonicNanos, NavigationSolution, NedVelocity, Redundancy, SensorClass, SolutionQuality,
        SolutionStamp, SourceComposition, SourceEpoch, SymmetricCov3, WrappingSequence,
    };

    use super::{AlertSeverity, EgpwsUnavailable, assess};
    use crate::config::EgpwsConfig;
    use crate::terrain::TerrainDatabase;

    /// The single clock domain of these tests: solutions are stamped on
    /// it and `now` is read from it unless a test probes the mismatch.
    const CLOCK: ClockDomainId = ClockDomainId::new(0);

    /// Flat terrain at one elevation with global coverage.
    struct FlatTerrain {
        elevation_m: f64,
    }

    impl TerrainDatabase for FlatTerrain {
        fn elevation_m(&self, _position: &GeodeticPosition) -> Option<f64> {
            Some(self.elevation_m)
        }
    }

    /// A database with no coverage anywhere.
    struct NoCoverageTerrain;

    impl TerrainDatabase for NoCoverageTerrain {
        fn elevation_m(&self, _position: &GeodeticPosition) -> Option<f64> {
            None
        }
    }

    fn solution(
        quality: SolutionQuality,
        altitude_m: f64,
        solved_at: MonotonicNanos,
    ) -> NavigationSolution {
        let composition = SourceComposition::of(SensorClass::Gnss);
        NavigationSolution::new(
            SolutionStamp::new(
                SourceEpoch::new(0),
                WrappingSequence::new(0),
                solved_at,
                CLOCK,
            ),
            GeodeticPosition::new(0.0, 0.0, altitude_m),
            NedVelocity::new(0.0, 0.0, 0.0),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(0.1, 0.1, 0.1),
            IntegrityAssessment::new(
                quality,
                composition,
                Redundancy::None,
                1.0,
                1.5,
                FaultDetection::Unavailable,
            ),
            composition,
        )
    }

    fn t0() -> MonotonicNanos {
        MonotonicNanos::from_nanos(0)
    }

    fn flat_assess(
        quality: SolutionQuality,
        altitude_m: f64,
        terrain_m: f64,
    ) -> Result<super::EgpwsAssessment, EgpwsUnavailable> {
        let db = FlatTerrain {
            elevation_m: terrain_m,
        };
        assess(
            &solution(quality, altitude_m, t0()),
            t0(),
            CLOCK,
            Some(&db),
            &EgpwsConfig::default(),
        )
    }

    #[test]
    fn no_database_is_refused_before_anything_else() {
        // Even a mismatched clock domain reports NoDatabase: the seam's
        // first question is whether terrain data exists at all.
        let result = assess(
            &solution(SolutionQuality::Good, 500.0, t0()),
            t0(),
            ClockDomainId::new(9),
            None,
            &EgpwsConfig::default(),
        );
        assert_eq!(result, Err(EgpwsUnavailable::NoDatabase));
    }

    #[test]
    fn now_read_on_another_clock_domain_is_refused_before_age_arithmetic() {
        let db = FlatTerrain { elevation_m: 0.0 };
        let solved_at = MonotonicNanos::from_nanos(1_000);
        let own = solution(SolutionQuality::Good, 500.0, solved_at);
        let foreign = ClockDomainId::new(9);
        // `now` precedes `solved_at`, so if age arithmetic ran first this
        // would be ClockInversion; the mismatch refusal proves the domain
        // is judged before any age arithmetic.
        let earlier = MonotonicNanos::from_nanos(0);
        let result = assess(&own, earlier, foreign, Some(&db), &EgpwsConfig::default());
        assert_eq!(
            result,
            Err(EgpwsUnavailable::ClockDomainMismatch {
                expected: CLOCK,
                got: foreign,
            })
        );
    }

    #[test]
    fn clock_inversion_is_refused_not_zeroed() {
        let db = FlatTerrain { elevation_m: 0.0 };
        let solved_at = MonotonicNanos::from_nanos(10);
        let earlier = MonotonicNanos::from_nanos(5);
        let own = solution(SolutionQuality::Good, 500.0, solved_at);
        let result = assess(&own, earlier, CLOCK, Some(&db), &EgpwsConfig::default());
        assert_eq!(
            result,
            Err(EgpwsUnavailable::ClockInversion {
                now: earlier,
                solved_at,
            })
        );
    }

    #[test]
    fn stale_solution_is_refused_and_the_bound_is_inclusive() {
        let db = FlatTerrain { elevation_m: 0.0 };
        let config = EgpwsConfig::default();
        let solved_at = MonotonicNanos::from_nanos(1_000);
        let own = solution(SolutionQuality::Good, 500.0, solved_at);
        let at_bound = MonotonicNanos::from_nanos(1_000 + config.max_solution_age.as_nanos());
        assert!(assess(&own, at_bound, CLOCK, Some(&db), &config).is_ok());
        let past_bound = MonotonicNanos::from_nanos(at_bound.as_nanos() + 1);
        let result = assess(&own, past_bound, CLOCK, Some(&db), &config);
        assert_eq!(
            result,
            Err(EgpwsUnavailable::SolutionStale {
                age: DurationNanos::from_nanos(config.max_solution_age.as_nanos() + 1),
                bound: config.max_solution_age,
            })
        );
        // Staleness is judged before integrity: a stale Unusable
        // solution reports staleness, not the quality floor.
        let unusable = solution(SolutionQuality::Unusable, 500.0, solved_at);
        let result = assess(&unusable, past_bound, CLOCK, Some(&db), &config);
        assert!(matches!(
            result,
            Err(EgpwsUnavailable::SolutionStale { .. })
        ));
    }

    #[test]
    fn unusable_quality_is_refused_and_carries_quality_and_floor() {
        let result = flat_assess(SolutionQuality::Unusable, 500.0, 0.0);
        assert_eq!(
            result,
            Err(EgpwsUnavailable::SolutionIntegrity {
                quality: SolutionQuality::Unusable,
                minimum: SolutionQuality::Degraded,
            })
        );
    }

    #[test]
    fn raised_minimum_quality_refuses_degraded() {
        let db = FlatTerrain { elevation_m: 0.0 };
        let config = EgpwsConfig {
            minimum_quality: SolutionQuality::Good,
            ..EgpwsConfig::default()
        };
        let result = assess(
            &solution(SolutionQuality::Degraded, 500.0, t0()),
            t0(),
            CLOCK,
            Some(&db),
            &config,
        );
        assert_eq!(
            result,
            Err(EgpwsUnavailable::SolutionIntegrity {
                quality: SolutionQuality::Degraded,
                minimum: SolutionQuality::Good,
            })
        );
    }

    #[test]
    fn degraded_meets_the_default_floor() {
        let result = flat_assess(SolutionQuality::Degraded, 500.0, 100.0);
        let assessment = result.unwrap_or_else(|e| panic!("expected assessment, got {e}"));
        assert_eq!(assessment.height_above_terrain_m, 400.0);
        assert_eq!(assessment.severity, None);
    }

    #[test]
    fn no_coverage_is_refused() {
        let result = assess(
            &solution(SolutionQuality::Good, 500.0, t0()),
            t0(),
            CLOCK,
            Some(&NoCoverageTerrain),
            &EgpwsConfig::default(),
        );
        assert_eq!(result, Err(EgpwsUnavailable::NoCoverage));
    }

    #[test]
    fn caution_band_raises_caution() {
        let result = flat_assess(SolutionQuality::Good, 210.0, 100.0);
        let assessment = result.unwrap_or_else(|e| panic!("expected assessment, got {e}"));
        assert_eq!(assessment.height_above_terrain_m, 110.0);
        assert_eq!(assessment.severity, Some(AlertSeverity::Caution));
    }

    #[test]
    fn warning_band_raises_warning() {
        let result = flat_assess(SolutionQuality::Good, 150.0, 100.0);
        let assessment = result.unwrap_or_else(|e| panic!("expected assessment, got {e}"));
        assert_eq!(assessment.height_above_terrain_m, 50.0);
        assert_eq!(assessment.severity, Some(AlertSeverity::Warning));
    }

    #[test]
    fn height_exactly_at_the_caution_floor_is_clear() {
        let result = flat_assess(SolutionQuality::Good, 250.0, 100.0);
        let assessment = result.unwrap_or_else(|e| panic!("expected assessment, got {e}"));
        assert_eq!(assessment.height_above_terrain_m, 150.0);
        assert_eq!(assessment.severity, None);
    }

    #[test]
    fn height_exactly_at_the_warning_floor_is_caution() {
        let result = flat_assess(SolutionQuality::Good, 160.0, 100.0);
        let assessment = result.unwrap_or_else(|e| panic!("expected assessment, got {e}"));
        assert_eq!(assessment.height_above_terrain_m, 60.0);
        assert_eq!(assessment.severity, Some(AlertSeverity::Caution));
    }

    #[test]
    fn below_terrain_is_a_warning_with_negative_height() {
        let result = flat_assess(SolutionQuality::Good, 50.0, 100.0);
        let assessment = result.unwrap_or_else(|e| panic!("expected assessment, got {e}"));
        assert_eq!(assessment.height_above_terrain_m, -50.0);
        assert_eq!(assessment.severity, Some(AlertSeverity::Warning));
    }
}
