//! Guidance vocabulary: the setpoints Navigate issues toward the flight
//! controller's declared command surface.
//!
//! Commands enter the host platform's fenced authority path; nothing here
//! names a transport or an authority concept (ADR-0005).

use crate::kinematics::{GeodeticPosition, NedVelocity};
use crate::stamp::SolutionStamp;
use crate::time::MonotonicNanos;

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
