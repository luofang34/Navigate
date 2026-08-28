//! Guidance vocabulary: the setpoints Navigate issues toward the flight
//! controller's declared command surface.
//!
//! Commands enter the host platform's fenced authority path; nothing here
//! names a transport or an authority concept (ADR-0005).

use crate::kinematics::{GeodeticPosition, NedVelocity};
use crate::stamp::SolutionStamp;
use crate::time::MonotonicNanos;

/// The lateral geometry the active leg gives a guidance derivation
/// (NAV-LG-014).
///
/// Each form is the reference a flown leg type defines, so a caller
/// cannot state a geometry the leg does not have: a track-to-fix leg
/// yields [`Self::Track`], an initial-fix or direct-to-fix leg yields
/// [`Self::PresentPosition`], and a course-to-fix leg yields
/// [`Self::Course`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LateralReference {
    /// A great-circle track from an upstream fix to the leg's fix.
    Track {
        /// Position of the upstream fix the track runs from.
        from: GeodeticPosition,
    },
    /// No inbound track: the path runs from the present position to the
    /// leg's fix, so the cross-track deviation is zero by construction
    /// and the reference course is the live bearing to the fix.
    PresentPosition,
    /// A published course terminating at the leg's fix. The reference
    /// path is the great circle through the fix whose true bearing at
    /// the fix, in the direction of flight, is `course_rad`.
    Course {
        /// The published course in radians, true, in `[0, 2π)`.
        course_rad: f64,
    },
}

impl LateralReference {
    /// A track running from the fix at `from`.
    #[must_use]
    pub const fn track(from: GeodeticPosition) -> Self {
        Self::Track { from }
    }

    /// A published course, radians true, in `[0, 2π)`.
    #[must_use]
    pub const fn course(course_rad: f64) -> Self {
        Self::Course { course_rad }
    }
}

/// One guidance setpoint, mirroring the FC's declared command surface.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum GuidanceSetpoint {
    /// Hold or capture a position.
    Position {
        /// Target position, WGS84.
        target: GeodeticPosition,
    },
    /// Fly a velocity vector.
    Velocity {
        /// Commanded velocity, local-level NED.
        velocity: NedVelocity,
    },
    /// Track out deviations from a reference path, flight-director
    /// style.
    DeviationTracking {
        /// Lateral deviation from the reference path in meters,
        /// positive right of course.
        lateral_m: f64,
        /// Vertical deviation from the reference profile in meters,
        /// positive above profile.
        vertical_m: f64,
        /// Reference course in radians, true, `[0, 2π)`.
        course_rad: f64,
    },
}

/// One guidance command: a setpoint plus the solution it was derived
/// from, for audit and staleness judgment.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GuidanceCommand {
    /// Monotonic time the command was issued. `issued_at` is read on the
    /// same clock domain as `basis.clock`; consumers judge command age
    /// only within that domain.
    pub issued_at: MonotonicNanos,
    /// The setpoint.
    pub setpoint: GuidanceSetpoint,
    /// Stamp of the navigation solution this command derives from.
    pub basis: SolutionStamp,
}

impl GuidanceCommand {
    /// Builds a command from its parts.
    #[must_use]
    pub const fn new(
        issued_at: MonotonicNanos,
        setpoint: GuidanceSetpoint,
        basis: SolutionStamp,
    ) -> Self {
        Self {
            issued_at,
            setpoint,
            basis,
        }
    }
}
