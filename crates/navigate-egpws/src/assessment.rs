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
    /// Vertical 1-sigma uncertainty exceeds the configured bound.
    #[error(
        "terrain assessment unavailable: vertical 1-sigma {vertical_1sigma_m} m \
         exceeds the configured bound {bound_m} m"
    )]
    VerticalUncertaintyExceeded {
        /// Vertical 1-sigma uncertainty of the refused solution in meters.
        vertical_1sigma_m: f64,
        /// Configured vertical 1-sigma bound in meters.
        bound_m: f64,
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
/// - [`EgpwsUnavailable::VerticalUncertaintyExceeded`] when
///   `solution.integrity.vertical_1sigma_m` exceeds
///   `config.max_vertical_1sigma_m`;
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
    let vertical_1sigma_m = solution.integrity.vertical_1sigma_m;
    if !vertical_1sigma_m.is_finite()
        || !config.max_vertical_1sigma_m.is_finite()
        || vertical_1sigma_m > config.max_vertical_1sigma_m
    {
        return Err(EgpwsUnavailable::VerticalUncertaintyExceeded {
            vertical_1sigma_m,
            bound_m: config.max_vertical_1sigma_m,
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
mod tests;
