//! Fail-closed admission and the leg geometry it yields.
//!
//! Every derivation in this crate enters through [`admit_leg`], so the
//! floors two derivations enforce cannot drift apart: one solution
//! judged twice yields the same refusal, and one leg yields one geometry.

use navigate_contract::{
    ClockDomainId, LateralReference, MonotonicNanos, NavigationSolution, SolutionQuality, Waypoint,
};
use navigate_geodesy::{cross_track_from_course_m, cross_track_m, initial_bearing_rad};

use crate::config::GuidanceConfig;
use crate::refusal::GuidanceRefusal;

/// The lateral geometry an admitted solution and leg define.
pub(crate) struct AdmittedLeg {
    /// Signed cross-track deviation in meters, positive right of course,
    /// matching [`navigate_geodesy::cross_track_m`]. A direct-to leg
    /// anchors the track at ownship, making this zero by construction.
    pub(crate) cross_track_m: f64,
    /// Reference course in radians, true, `[0, 2π)`.
    pub(crate) course_rad: f64,
}

/// Rank under the quality ordering `Good` > `Degraded` > `Unusable`.
/// A classification this crate cannot interpret ranks with `Unusable`:
/// fail closed, never optimistic (ADR-0004).
const fn quality_rank(quality: SolutionQuality) -> u8 {
    match quality {
        SolutionQuality::Good => 2,
        SolutionQuality::Degraded => 1,
        _ => 0,
    }
}

/// Admits the solution and the leg, returning the lateral geometry a
/// derivation reads.
///
/// Each form of `reference` is the geometry one flown leg type defines
/// (NAV-LG-014). [`LateralReference::Track`] runs the reference track
/// from the upstream fix to `leg_to`.
/// [`LateralReference::PresentPosition`] anchors the track at ownship,
/// so the cross-track deviation is zero by construction and the course
/// is the live bearing to the waypoint.
/// [`LateralReference::Course`] measures against the published course
/// line through `leg_to` and reports the published course.
///
/// # Errors
///
/// The refusal set of [`crate::guide`], in the order enforced:
/// [`GuidanceRefusal::ClockDomainMismatch`],
/// [`GuidanceRefusal::IntegrityBelowFloor`],
/// [`GuidanceRefusal::ClockInversion`], [`GuidanceRefusal::SolutionStale`],
/// then [`GuidanceRefusal::ImplausibleTarget`].
pub(crate) fn admit_leg(
    solution: &NavigationSolution,
    reference: LateralReference,
    leg_to: &Waypoint,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &GuidanceConfig,
) -> Result<AdmittedLeg, GuidanceRefusal> {
    admit_solution(solution, now, now_clock, config)?;
    if !leg_to.position.is_plausible() {
        return Err(GuidanceRefusal::ImplausibleTarget {
            ident: leg_to.ident.clone(),
        });
    }
    // A course line needs one position and one direction, so it has no
    // two-endpoint degenerate case to refuse (NAV-LG-012).
    if let LateralReference::Course { course_rad } = reference {
        return Ok(AdmittedLeg {
            cross_track_m: cross_track_from_course_m(
                &solution.position,
                &leg_to.position,
                course_rad,
            ),
            course_rad,
        });
    }
    let track_start = match reference {
        LateralReference::Track { ref from } => from,
        _ => &solution.position,
    };
    // Any track the geodesy layer refuses (endpoints below its
    // degenerate-track floor) cannot define a course, so the leg's
    // target is implausible as a guidance target.
    let lateral_m =
        cross_track_m(&solution.position, track_start, &leg_to.position).map_err(|_| {
            GuidanceRefusal::ImplausibleTarget {
                ident: leg_to.ident.clone(),
            }
        })?;
    Ok(AdmittedLeg {
        cross_track_m: lateral_m,
        course_rad: initial_bearing_rad(track_start, &leg_to.position),
    })
}

/// Fail-closed solution admission: clock-domain agreement first (age
/// arithmetic across domains is meaningless), then quality floor, then
/// age against `now`.
fn admit_solution(
    solution: &NavigationSolution,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &GuidanceConfig,
) -> Result<(), GuidanceRefusal> {
    let expected = solution.stamp.clock;
    if now_clock != expected {
        return Err(GuidanceRefusal::ClockDomainMismatch {
            expected,
            got: now_clock,
        });
    }
    let quality = solution.integrity.quality;
    if quality_rank(quality) < quality_rank(config.minimum_quality) {
        return Err(GuidanceRefusal::IntegrityBelowFloor {
            quality,
            floor: config.minimum_quality,
        });
    }
    let solved_at = solution.stamp.solved_at;
    let age = now
        .elapsed_since(solved_at)
        .ok_or(GuidanceRefusal::ClockInversion { now, solved_at })?;
    if age > config.max_solution_age {
        return Err(GuidanceRefusal::SolutionStale {
            age,
            bound: config.max_solution_age,
        });
    }
    Ok(())
}
