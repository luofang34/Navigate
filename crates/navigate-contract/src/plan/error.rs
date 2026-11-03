//! Structural defects of a flight plan. Every defect names the plan and
//! the waypoint with an [`Ident`], so the error is `Copy` and does not
//! allocate.

use core::fmt;

use super::Ident;

/// Which declared quantity a validation refusal names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConstraintField {
    /// A bound of the altitude constraint.
    Altitude,
    /// The maximum-speed constraint.
    MaxSpeed,
    /// The vertical gradient.
    Gradient,
}

impl fmt::Display for ConstraintField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Altitude => "altitude constraint",
            Self::MaxSpeed => "maximum-speed constraint",
            Self::Gradient => "gradient",
        };
        formatter.write_str(name)
    }
}

/// Structural defects a plan can carry.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PlanValidationError {
    /// The plan ellipsoid cannot define geodetic coordinates
    /// ([`crate::Ellipsoid::is_plausible`] refuses it).
    #[error("plan {plan} has an implausible reference ellipsoid")]
    ImplausibleEllipsoid {
        /// Offending plan id.
        plan: Ident,
    },
    /// The plan holds no waypoints.
    #[error("plan {plan} has no waypoints")]
    Empty {
        /// Offending plan id.
        plan: Ident,
    },
    /// A waypoint coordinate is non-finite or out of geodetic range.
    #[error("plan {plan} waypoint {index} ({ident}) has an implausible position")]
    ImplausiblePosition {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: Ident,
    },
    /// A waypoint's altitude window does not bracket a band: its lower
    /// bound is at or above its upper bound.
    #[error(
        "plan {plan} waypoint {index} ({ident}) has an altitude window whose lower bound is not below its upper bound"
    )]
    InvertedAltitudeWindow {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: Ident,
    },
    /// A declared constraint on a waypoint is not a finite number.
    #[error("plan {plan} waypoint {index} ({ident}) has a non-finite {field}")]
    NonFiniteConstraint {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: Ident,
        /// Which declared quantity is non-finite.
        field: ConstraintField,
    },
    /// A waypoint demands a speed no vehicle can fly a leg at.
    #[error(
        "plan {plan} waypoint {index} ({ident}) demands a non-positive speed {max_speed_mps} m/s"
    )]
    NonPositiveSpeedConstraint {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: Ident,
        /// The speed that cannot be flown.
        max_speed_mps: f64,
    },
    /// A waypoint declares a zero gradient, which cannot fly any profile
    /// change; absence expresses "no gradient limit".
    #[error("plan {plan} waypoint {index} ({ident}) declares a zero gradient")]
    ZeroGradient {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: Ident,
    },
    /// The first waypoint declares a track-to-fix leg, which needs a fix
    /// before it to run from (NAV-LG-004).
    #[error("plan {plan} waypoint 0 ({ident}) declares a TF leg with no fix before it")]
    TrackWithoutUpstreamFix {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint identifier.
        ident: Ident,
    },
    /// A waypoint after the first declares an initial-fix leg, which
    /// starts a procedure and cannot sit inside one (NAV-LG-004).
    #[error("plan {plan} waypoint {index} ({ident}) declares an IF leg after the first waypoint")]
    InitialFixNotFirst {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: Ident,
    },
    /// A course-to-fix leg declares a course that is not a finite angle
    /// in `[0, 2π)`, so no vehicle can fly it (NAV-LG-004).
    #[error(
        "plan {plan} waypoint {index} ({ident}) declares a course {course_rad} outside [0, 2π)"
    )]
    CourseOutOfRange {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: Ident,
        /// The course that cannot be flown.
        course_rad: f64,
    },
}
