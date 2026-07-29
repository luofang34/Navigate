//! Typed refusals: why guidance declined to issue a setpoint.

use navigate_contract::{ClockDomainId, DurationNanos, MonotonicNanos, SolutionQuality};
use thiserror::Error;

/// Why [`crate::guide`] refused to derive a setpoint.
///
/// Guidance consumes integrity fail-closed (ADR-0004): every refusal
/// names the floor or bound it enforced so the caller can log, display,
/// or escalate without re-deriving the judgment.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum GuidanceRefusal {
    /// Solution quality is below the configured floor.
    #[error("solution quality {quality:?} is below the configured floor {floor:?}")]
    IntegrityBelowFloor {
        /// Quality the solution actually carries.
        quality: SolutionQuality,
        /// Configured minimum quality.
        floor: SolutionQuality,
    },
    /// The solution is older than the configured bound.
    #[error(
        "solution age {} ns exceeds the configured bound {} ns",
        age.as_nanos(),
        bound.as_nanos()
    )]
    SolutionStale {
        /// Solution age at guidance time.
        age: DurationNanos,
        /// Configured maximum age.
        bound: DurationNanos,
    },
    /// `now` was read on a different clock domain than the solution's
    /// stamp, so the two readings are not comparable and age cannot be
    /// judged: readings from different domains are never subtracted.
    /// Refused before any age arithmetic runs.
    #[error(
        "clock domain mismatch: now was read on domain {}, the solution's stamp is on domain {}",
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
        "clock inversion: now {} ns is earlier than solved_at {} ns",
        now.as_nanos(),
        solved_at.as_nanos()
    )]
    ClockInversion {
        /// Caller-supplied guidance time.
        now: MonotonicNanos,
        /// Solution reference time from the stamp.
        solved_at: MonotonicNanos,
    },
    /// The target waypoint cannot anchor a leg: its position fails the
    /// geodetic plausibility screen (non-finite component or
    /// latitude/longitude out of range), or the leg's endpoints coincide
    /// so the leg cannot define a course.
    #[error("target waypoint {ident} has an implausible position or defines a degenerate leg")]
    ImplausibleTarget {
        /// Identifier of the refused waypoint.
        ident: String,
    },
}
