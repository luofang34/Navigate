use navigate_contract::{
    ClockDomainId, DurationNanos, FaultDetection, GeodeticPosition, IntegrityAssessment,
    MonotonicNanos, NavigationSolution, NedVelocity, Redundancy, SensorClass, SolutionQuality,
    SolutionStamp, SourceComposition, SourceEpoch, SymmetricCov3, WrappingSequence,
};

use super::{AlertSeverity, EgpwsAssessment, EgpwsUnavailable, assess};
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

fn solution_with_vertical_sigma(
    quality: SolutionQuality,
    altitude_m: f64,
    vertical_1sigma_m: f64,
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
        SymmetricCov3::from_diagonal(1.0, 1.0, vertical_1sigma_m * vertical_1sigma_m),
        SymmetricCov3::from_diagonal(0.1, 0.1, 0.1),
        IntegrityAssessment::new(
            quality,
            composition,
            Redundancy::None,
            1.0,
            vertical_1sigma_m,
            FaultDetection::Unavailable,
        ),
        composition,
    )
}

fn solution(
    quality: SolutionQuality,
    altitude_m: f64,
    solved_at: MonotonicNanos,
) -> NavigationSolution {
    solution_with_vertical_sigma(quality, altitude_m, 1.5, solved_at)
}

fn t0() -> MonotonicNanos {
    MonotonicNanos::from_nanos(0)
}

fn flat_assess(
    quality: SolutionQuality,
    altitude_m: f64,
    terrain_m: f64,
) -> Result<EgpwsAssessment, EgpwsUnavailable> {
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
fn vertical_uncertainty_above_bound_is_refused() {
    let db = FlatTerrain { elevation_m: 0.0 };
    let config = EgpwsConfig::default();
    let at_bound = solution_with_vertical_sigma(SolutionQuality::Degraded, 500.0, 30.0, t0());
    assert!(assess(&at_bound, t0(), CLOCK, Some(&db), &config).is_ok());

    let past_bound = solution_with_vertical_sigma(SolutionQuality::Degraded, 500.0, 45.0, t0());
    let err = assess(&past_bound, t0(), CLOCK, Some(&db), &config)
        .expect_err("weak vertical sigma must refuse terrain clearance");
    assert_eq!(
        err,
        EgpwsUnavailable::VerticalUncertaintyExceeded {
            vertical_1sigma_m: 45.0,
            bound_m: 30.0,
        }
    );
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
