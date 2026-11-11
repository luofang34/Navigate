//! Typed refusals: why guidance declined to issue a setpoint.

use navigate_contract::{
    ClockDomainId, DurationNanos, Ellipsoid, FrameId, Ident, MonotonicNanos, SolutionQuality,
};
use navigate_trajectory::TrajectoryError;
use thiserror::Error;

use crate::mission::MissionPhase;

/// Why a guidance derivation refused to issue a setpoint. The surface
/// derivations call one shared admission helper, and the space
/// derivations call [`crate::admit_space_solution`]. Thus one solution
/// gives the same refusal in each derivation of one kind.
///
/// Guidance checks integrity with fail-closed logic (ADR-0004). Each
/// refusal records the bound that failed so the caller can log, display,
/// or escalate the condition directly.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
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
    /// The solution ellipsoid cannot define geodetic coordinates
    /// (ADR-0013): the semi-major axis is not finite and positive, the
    /// flattening is not in `[0, 1)`, or the frame is inertial. Guidance
    /// geometry on such an ellipsoid has no meaning.
    #[error(
        "solution ellipsoid is not plausible: semi-major axis {semi_major_axis_m} m, \
         flattening {flattening}"
    )]
    ImplausibleEllipsoid {
        /// Semi-major axis of the refused ellipsoid in meters.
        semi_major_axis_m: f64,
        /// Flattening of the refused ellipsoid.
        flattening: f64,
    },
    /// The solution ellipsoid does not match the expected plan ellipsoid
    /// (ADR-0015).
    #[error("ellipsoid mismatch: expected {expected:?}, got {got:?}")]
    EllipsoidMismatch {
        /// Reference ellipsoid required by the active plan or config.
        expected: Ellipsoid,
        /// Reference ellipsoid carried by the navigation solution.
        got: Ellipsoid,
    },
    /// Vertical 1-sigma uncertainty exceeds the configured bound on an
    /// altitude-constrained leg, or the uncertainty is not finite, or
    /// the bound is NaN. The bound is
    /// `GuidanceConfig::max_vertical_1sigma_m`, which is independent of
    /// the quality floor of [`Self::IntegrityBelowFloor`].
    #[error(
        "vertical 1-sigma uncertainty {vertical_1sigma_m} m exceeds the configured bound {bound_m} m"
    )]
    VerticalUncertaintyExceeded {
        /// Vertical 1-sigma uncertainty of the refused solution in meters.
        vertical_1sigma_m: f64,
        /// Configured vertical 1-sigma bound in meters.
        bound_m: f64,
    },
    /// The target waypoint cannot anchor a leg: its position fails the
    /// geodetic plausibility screen (non-finite component or
    /// latitude/longitude out of range), or the leg's endpoints coincide
    /// so the leg cannot define a course.
    #[error("target waypoint {ident} has an implausible position or defines a degenerate leg")]
    ImplausibleTarget {
        /// Identifier of the refused waypoint.
        ident: Ident,
    },
    /// The frame of a space solution is not the frame of the target
    /// (ADR-0019).
    #[error("frame mismatch: expected {expected:?}, got {got:?}")]
    FrameMismatch {
        /// The frame of the target.
        expected: FrameId,
        /// The frame of the solution.
        got: FrameId,
    },
    /// The one-sigma position uncertainty along the largest principal
    /// axis of a space solution is not finite or exceeds the configured
    /// bound.
    #[error("position 1-sigma uncertainty {max_axis_1sigma_m} m exceeds the bound {bound_m} m")]
    UncertaintyExceeded {
        /// The one-sigma uncertainty of the refused solution in meters.
        max_axis_1sigma_m: f64,
        /// The configured bound in meters.
        bound_m: f64,
    },
    /// The vehicle is outside a keep-out corridor or below a glide
    /// slope, or its scaled uncertainty does not fit in the corridor
    /// margin.
    #[error("corridor violated: {violation}")]
    CorridorViolated {
        /// The violation that the trajectory law found.
        #[source]
        violation: TrajectoryError,
    },
    /// The trajectory law cannot give a command for this target and
    /// state.
    #[error("trajectory invalid: {error}")]
    TrajectoryInvalid {
        /// The error of the trajectory law.
        #[source]
        error: TrajectoryError,
    },
    /// The mission input is not the kind that the phase reads: a space
    /// phase reads a Cartesian solution, and the atmospheric route reads
    /// a surface solution.
    #[error("phase {phase:?} cannot read this kind of solution")]
    PhaseInputMismatch {
        /// The phase of the mission.
        phase: MissionPhase,
    },
    /// The phase is not part of the mission profile.
    #[error("phase {phase:?} is not part of the mission profile")]
    PhaseNotInProfile {
        /// The phase of the mission.
        phase: MissionPhase,
    },
}

impl GuidanceRefusal {
    /// The refusal for an error of a trajectory law: a corridor
    /// violation is [`Self::CorridorViolated`], and every other error is
    /// [`Self::TrajectoryInvalid`].
    #[must_use]
    pub const fn from_trajectory(error: TrajectoryError) -> Self {
        if error.is_corridor_violation() {
            Self::CorridorViolated { violation: error }
        } else {
            Self::TrajectoryInvalid { error }
        }
    }
}
