//! The velocity derivation: solution plus leg geometry to one NED
//! velocity setpoint.

use navigate_contract::{
    ClockDomainId, GeodeticPosition, GuidanceCommand, GuidanceSetpoint, MonotonicNanos,
    NavigationSolution, NedVelocity, Waypoint,
};
use navigate_geodesy::distance_m;

use crate::admission::admit_leg;
use crate::config::VelocityGuidanceConfig;
use crate::refusal::GuidanceRefusal;
use crate::vertical;

mod compose;

#[cfg(test)]
mod constraint_tests;
#[cfg(test)]
mod tests;

/// Derives one local-level NED velocity setpoint from a navigation
/// solution and the active leg, or refuses with a typed reason.
///
/// Admission is [`crate::guide`]'s, enforced by the same helper against
/// `config.admission`: a solution one derivation refuses, the other
/// refuses identically. `now` must be read on the solution's clock
/// domain (`solution.stamp.clock`).
///
/// Lateral: the commanded horizontal velocity is along-track progress
/// plus a cross-track correction.
///
/// - The reference track runs `leg_from` → `leg_to`, or ownship →
///   `leg_to` for a direct-to leg, whose cross-track deviation is
///   therefore zero and whose velocity is pure bearing-aligned progress.
/// - Along-track speed is `config.cruise_mps` — bounded by the target
///   waypoint's speed constraint when it carries one (NAV-VC-003) —
///   scaled linearly by the distance remaining inside
///   `config.arrival_slowdown_radius_m` and floored at
///   [`VelocityGuidanceConfig::MIN_APPROACH_SPEED_MPS`]. Capturing the
///   waypoint and terminating the leg stay `navigate-fpl`'s decisions;
///   the slowdown only shapes the speed guidance asks for.
/// - The correction is `config.cross_track_gain_per_s` times the
///   deviation, limited to `config.max_horizontal_mps`, directed to
///   *reduce* the deviation: right of course (positive cross-track,
///   matching [`navigate_geodesy::cross_track_m`]) corrects leftward.
/// - The composed horizontal vector is capped at
///   `config.max_horizontal_mps` by scaling, so the cap trades speed for
///   nothing else — the commanded direction survives it.
///
/// Vertical: `config.vertical_gain_per_s` times the deviation from
/// `leg_to`'s altitude constraint (positive above the profile, the
/// convention of [`GuidanceSetpoint::DeviationTracking`]), limited to
/// `config.max_vertical_mps` and, when the leg declares a gradient, to
/// `|gradient|` times the commanded along-track speed (NAV-VC-002).
/// Above the profile commands a positive down component — descent. A
/// waypoint without a constraint commands `0.0`.
///
/// # Errors
///
/// - [`GuidanceRefusal::ClockDomainMismatch`] when `now_clock` differs
///   from the solution stamp's clock domain.
/// - [`GuidanceRefusal::IntegrityBelowFloor`] when solution quality is
///   below `config.admission.minimum_quality`.
/// - [`GuidanceRefusal::ClockInversion`] when `now` is earlier than the
///   solution's `solved_at`.
/// - [`GuidanceRefusal::SolutionStale`] when the solution's age exceeds
///   `config.admission.max_solution_age`.
/// - [`GuidanceRefusal::ImplausibleTarget`] when `leg_to`'s position
///   fails the geodetic plausibility screen, or when the leg's endpoints
///   coincide (a leg whose endpoints coincide cannot define a course).
pub fn guide_velocity(
    solution: &NavigationSolution,
    leg_from: Option<&GeodeticPosition>,
    leg_to: &Waypoint,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &VelocityGuidanceConfig,
) -> Result<GuidanceCommand, GuidanceRefusal> {
    let leg = admit_leg(
        solution,
        leg_from,
        leg_to,
        now,
        now_clock,
        &config.admission,
    )?;
    let remaining_m = distance_m(&solution.position, &leg_to.position);
    // A waypoint speed constraint bounds the leg toward it (NAV-VC-003);
    // constraints below a flyable floor are refused at plan activation,
    // so a plain min is honest here.
    let cruise_mps = leg_to
        .max_speed_mps
        .map_or(config.cruise_mps, |constraint| {
            config.cruise_mps.min(constraint)
        });
    let (horizontal, along_speed_mps) =
        compose::horizontal_mps(&leg, remaining_m, cruise_mps, config);
    let deviation_m = vertical::deviation_m(solution.position.altitude_m, leg_to.altitude.as_ref());
    Ok(GuidanceCommand::new(
        now,
        GuidanceSetpoint::Velocity {
            velocity: NedVelocity::new(
                horizontal.north_mps,
                horizontal.east_mps,
                compose::down_mps(deviation_m, leg_to.gradient, along_speed_mps, config),
            ),
        },
        solution.stamp,
    ))
}
