//! Typed reasons why a trajectory law gives no command.

use navigate_contract::{FrameId, MonotonicNanos};
use thiserror::Error;

/// Why a trajectory law gives no command.
///
/// Each variant records the value that failed, so that the caller can
/// log or show the condition directly. A law never replaces a failed
/// value with a default.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum TrajectoryError {
    /// A component of the position or the velocity is not finite.
    #[error("a component of the state vector is not finite")]
    NonFiniteState,
    /// A parameter of a target, a corridor, or a law is not finite or is
    /// out of its range.
    #[error("parameter {name} is not finite or is out of its range")]
    InvalidParameter {
        /// The name of the parameter.
        name: &'static str,
    },
    /// The position or the angular momentum is too small to define an
    /// orbit plane or a local frame.
    #[error("the state is too close to the center or to a radial line to define an orbit plane")]
    DegenerateOrbit,
    /// The radius is outside the radii of the target conic.
    #[error(
        "radius {radius_m} m is outside the target conic from {periapsis_radius_m} m to \
         {apoapsis_radius_m} m"
    )]
    ApsidesUnreachable {
        /// The current radius in meters.
        radius_m: f64,
        /// The periapsis radius of the target conic in meters.
        periapsis_radius_m: f64,
        /// The apoapsis radius of the target conic in meters.
        apoapsis_radius_m: f64,
    },
    /// A relative-motion law needs an inertial frame.
    #[error("frame {frame:?} is not inertial")]
    NonInertialFrame {
        /// The frame that is not inertial.
        frame: FrameId,
    },
    /// The Clohessy–Wiltshire transfer has no solution for this time of
    /// flight, because the transfer time is near a whole number of
    /// orbits.
    #[error("the transfer is singular for a time of flight of {time_of_flight_s} s")]
    SingularTransfer {
        /// The time of flight in seconds.
        time_of_flight_s: f64,
    },
    /// The chaser is in the keep-out sphere and outside the approach
    /// cone.
    #[error("the chaser is {margin_m} m inside the boundary of the approach corridor")]
    OutsideApproachCorridor {
        /// The signed distance to the cone boundary in meters. It is
        /// negative outside the cone.
        margin_m: f64,
    },
    /// The scaled position uncertainty is larger than the distance to the
    /// boundary of the approach corridor.
    #[error(
        "scaled position uncertainty {scaled_1sigma_m} m exceeds the corridor margin {margin_m} m"
    )]
    UncertaintyExceedsCorridorMargin {
        /// The one-sigma position uncertainty times the corridor sigma
        /// multiplier, in meters.
        scaled_1sigma_m: f64,
        /// The distance to the cone boundary in meters.
        margin_m: f64,
    },
    /// The vehicle is below the glide-slope cone of the landing target.
    #[error("elevation {elevation_rad} rad is below the glide slope {minimum_rad} rad")]
    GlideSlopeViolated {
        /// The elevation of the vehicle above the target horizon in
        /// radians.
        elevation_rad: f64,
        /// The minimum elevation in radians.
        minimum_rad: f64,
    },
    /// The entry corridor has fewer than two knots, or the knot energies
    /// do not decrease strictly.
    #[error("the entry corridor needs two or more knots in strictly decreasing energy")]
    InvalidCorridor,
    /// The heading is not defined: the vehicle is on the polar axis, or
    /// its horizontal velocity is zero.
    #[error("the heading is not defined at this position and velocity")]
    UndefinedHeading,
    /// The ignition window closed before the burn started.
    #[error("the ignition window closed at {} ns", close.as_nanos())]
    IgnitionWindowClosed {
        /// The end of the ignition window.
        close: MonotonicNanos,
    },
}

impl TrajectoryError {
    /// Whether the error is a safety refusal of a keep-out corridor or of
    /// a glide slope, as opposed to an invalid input. An uncertainty that
    /// does not fit in the corridor margin is a corridor refusal.
    #[must_use]
    pub const fn is_corridor_violation(&self) -> bool {
        matches!(
            self,
            Self::OutsideApproachCorridor { .. }
                | Self::UncertaintyExceedsCorridorMargin { .. }
                | Self::GlideSlopeViolated { .. }
        )
    }
}
