//! The deviation-tracking derivation: solution plus leg geometry to one
//! flight-director setpoint.

use navigate_contract::{
    ClockDomainId, GuidanceCommand, GuidanceSetpoint, LateralReference, MonotonicNanos,
    NavigationSolution, Waypoint,
};

use crate::admission::admit_leg;
use crate::config::GuidanceConfig;
use crate::refusal::GuidanceRefusal;
use crate::vertical;

/// Derives one deviation-tracking guidance command from a navigation
/// solution and the active leg, or refuses with a typed reason.
///
/// `now` must be read on the solution's clock domain
/// (`solution.stamp.clock`): readings from different domains are never
/// subtracted, so a mismatched `now_clock` is refused before any age
/// arithmetic.
///
/// Lateral: `reference` is the geometry the active leg defines
/// (NAV-LG-014). A track-to-fix leg runs the reference track from its
/// upstream fix to `leg_to`. An initial-fix or direct-to-fix leg anchors
/// the track at ownship, so the cross-track deviation is zero by
/// construction and the course is the live bearing to the waypoint. A
/// course-to-fix leg measures against the published course line through
/// `leg_to` and reports the published course. `lateral_m` is positive
/// right of course, matching [`navigate_geodesy::cross_track_m`].
///
/// Vertical: deviation from `leg_to`'s altitude constraint, positive
/// above the target profile; the `OrAbove`/`OrBelow` forms report `0.0`
/// while satisfied (see [`GuidanceSetpoint::DeviationTracking`]). A
/// waypoint without a constraint yields `0.0`; profile interpolation
/// between constrained waypoints is a designed extension.
///
/// # Errors
///
/// - [`GuidanceRefusal::ClockDomainMismatch`] when `now_clock` differs
///   from the solution stamp's clock domain.
/// - [`GuidanceRefusal::IntegrityBelowFloor`] when solution quality is
///   below `config.minimum_quality`.
/// - [`GuidanceRefusal::ClockInversion`] when `now` is earlier than the
///   solution's `solved_at`.
/// - [`GuidanceRefusal::SolutionStale`] when the solution's age exceeds
///   `config.max_solution_age`.
/// - [`GuidanceRefusal::ImplausibleTarget`] when `leg_to`'s position
///   fails the geodetic plausibility screen, or when a track reference's
///   endpoints coincide (such a track cannot define a course). A course
///   reference has no such case: a fix and a course define the line.
pub fn guide(
    solution: &NavigationSolution,
    reference: LateralReference,
    leg_to: &Waypoint,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &GuidanceConfig,
) -> Result<GuidanceCommand, GuidanceRefusal> {
    let leg = admit_leg(solution, reference, leg_to, now, now_clock, config)?;
    let vertical_m = vertical::deviation_m(solution.position.altitude_m, leg_to.altitude.as_ref());
    Ok(GuidanceCommand::new(
        now,
        GuidanceSetpoint::DeviationTracking {
            lateral_m: leg.cross_track_m,
            vertical_m,
            course_rad: leg.course_rad,
        },
        solution.stamp,
    ))
}

#[cfg(test)]
mod tests;
