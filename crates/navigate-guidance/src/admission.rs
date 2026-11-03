//! Fail-closed admission and the leg geometry it yields.
//!
//! Every surface derivation in this crate enters through [`admit_leg`],
//! so the floors two derivations enforce cannot drift apart: one
//! solution judged twice yields the same refusal, and one leg yields one
//! geometry. The geometry runs on the mean-radius sphere of the solution
//! ellipsoid (ADR-0013). Every space derivation enters through
//! [`admit_space_solution`]. Both paths share the checks of the stamp.

use navigate_contract::{
    CartesianSolution, ClockDomainId, DurationNanos, FrameId, LateralReference, MonotonicNanos,
    NavigationSolution, SolutionQuality, SolutionStamp, Waypoint,
};
use navigate_geodesy::{
    BodyEllipsoid, cross_track_from_course_m, cross_track_m, initial_bearing_rad,
};

use crate::config::{GuidanceConfig, SpaceGuidanceConfig};
use crate::refusal::GuidanceRefusal;

/// The lateral geometry an admitted solution and leg define.
pub(crate) struct AdmittedLeg {
    /// Signed cross-track deviation in meters, positive right of course,
    /// matching [`navigate_geodesy::cross_track_m`]. A direct-to leg
    /// anchors the track at ownship, making this zero by construction.
    pub(crate) cross_track_m: f64,
    /// Reference course in radians, true, `[0, 2π)`.
    pub(crate) course_rad: f64,
    /// The admitted ellipsoid of the solution. Every distance of the leg
    /// uses its mean-radius sphere.
    pub(crate) ellipsoid: BodyEllipsoid,
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
/// [`GuidanceRefusal::ImplausibleEllipsoid`],
/// [`GuidanceRefusal::EllipsoidMismatch`],
/// [`GuidanceRefusal::VerticalUncertaintyExceeded`],
/// then [`GuidanceRefusal::ImplausibleTarget`].
pub(crate) fn admit_leg(
    solution: &NavigationSolution,
    reference: LateralReference,
    leg_to: &Waypoint,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &GuidanceConfig,
) -> Result<AdmittedLeg, GuidanceRefusal> {
    let ellipsoid = admit_solution(solution, now, now_clock, config)?;
    let vertical_1sigma_m = solution.integrity.vertical_1sigma_m;
    if leg_to.altitude.is_some()
        && (!vertical_1sigma_m.is_finite()
            || !config.max_vertical_1sigma_m.is_finite()
            || vertical_1sigma_m > config.max_vertical_1sigma_m)
    {
        return Err(GuidanceRefusal::VerticalUncertaintyExceeded {
            vertical_1sigma_m,
            bound_m: config.max_vertical_1sigma_m,
        });
    }
    if !leg_to.position.is_plausible() {
        return Err(GuidanceRefusal::ImplausibleTarget {
            ident: leg_to.ident,
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
                ellipsoid,
            ),
            course_rad,
            ellipsoid,
        });
    }
    let track_start = match reference {
        LateralReference::Track { ref from } => from,
        _ => &solution.position,
    };
    // Any track the geodesy layer refuses (endpoints below its
    // degenerate-track floor) cannot define a course, so the leg's
    // target is implausible as a guidance target.
    let lateral_m = cross_track_m(&solution.position, track_start, &leg_to.position, ellipsoid)
        .map_err(|_| GuidanceRefusal::ImplausibleTarget {
            ident: leg_to.ident,
        })?;
    Ok(AdmittedLeg {
        cross_track_m: lateral_m,
        course_rad: initial_bearing_rad(track_start, &leg_to.position),
        ellipsoid,
    })
}

/// Fail-closed solution admission: clock-domain agreement first (age
/// arithmetic across domains is meaningless), then quality floor, then
/// age against `now`, then the solution ellipsoid and optional body check.
/// The result is the admitted ellipsoid.
fn admit_solution(
    solution: &NavigationSolution,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &GuidanceConfig,
) -> Result<BodyEllipsoid, GuidanceRefusal> {
    admit_stamp(
        &solution.stamp,
        solution.integrity.quality,
        now,
        now_clock,
        config.minimum_quality,
        config.max_solution_age,
    )?;
    let body_ellipsoid = BodyEllipsoid::try_from_contract(solution.ellipsoid).map_err(|_| {
        GuidanceRefusal::ImplausibleEllipsoid {
            semi_major_axis_m: solution.ellipsoid.semi_major_axis_m,
            flattening: solution.ellipsoid.flattening,
        }
    })?;
    if let Some(expected_ellipsoid) = config.expected_ellipsoid
        && solution.ellipsoid != expected_ellipsoid
    {
        return Err(GuidanceRefusal::EllipsoidMismatch {
            expected: expected_ellipsoid,
            got: solution.ellipsoid,
        });
    }
    Ok(body_ellipsoid)
}

/// Fail-closed admission of a solution in a Cartesian frame (ADR-0019).
///
/// Every space derivation calls this function, so that one solution
/// gives the same refusal in each of them. A caller that steps a
/// mission can also call it to check a solution before the step.
///
/// # Errors
///
/// In the order enforced: [`GuidanceRefusal::ClockDomainMismatch`] when
/// `now_clock` differs from the clock domain of the stamp,
/// [`GuidanceRefusal::IntegrityBelowFloor`] when the quality is below
/// `config.minimum_quality`, [`GuidanceRefusal::ClockInversion`] when
/// `now` is earlier than the stamp, [`GuidanceRefusal::SolutionStale`]
/// when the age exceeds `config.max_solution_age`,
/// [`GuidanceRefusal::FrameMismatch`] when `solution.frame` is not
/// `expected_frame`, and [`GuidanceRefusal::UncertaintyExceeded`] when
/// the one-sigma position uncertainty along the largest principal axis
/// is not finite or exceeds `config.max_position_1sigma_m`.
pub fn admit_space_solution(
    solution: &CartesianSolution,
    expected_frame: FrameId,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &SpaceGuidanceConfig,
) -> Result<(), GuidanceRefusal> {
    admit_stamp(
        &solution.stamp,
        solution.integrity.quality,
        now,
        now_clock,
        config.minimum_quality,
        config.max_solution_age,
    )?;
    if solution.frame != expected_frame {
        return Err(GuidanceRefusal::FrameMismatch {
            expected: expected_frame,
            got: solution.frame,
        });
    }
    let max_axis_1sigma_m = solution.integrity.uncertainty.max_axis_1sigma_m();
    let bound_m = config.max_position_1sigma_m;
    if !(max_axis_1sigma_m.is_finite() && bound_m.is_finite() && max_axis_1sigma_m <= bound_m) {
        return Err(GuidanceRefusal::UncertaintyExceeded {
            max_axis_1sigma_m,
            bound_m,
        });
    }
    Ok(())
}

/// The checks that a surface solution and a space solution share: the
/// clock domain, the quality floor, the clock order, and the age.
fn admit_stamp(
    stamp: &SolutionStamp,
    quality: SolutionQuality,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    minimum_quality: SolutionQuality,
    max_solution_age: DurationNanos,
) -> Result<(), GuidanceRefusal> {
    if now_clock != stamp.clock {
        return Err(GuidanceRefusal::ClockDomainMismatch {
            expected: stamp.clock,
            got: now_clock,
        });
    }
    if quality_rank(quality) < quality_rank(minimum_quality) {
        return Err(GuidanceRefusal::IntegrityBelowFloor {
            quality,
            floor: minimum_quality,
        });
    }
    let solved_at = stamp.solved_at;
    let age = now
        .elapsed_since(solved_at)
        .ok_or(GuidanceRefusal::ClockInversion { now, solved_at })?;
    if age > max_solution_age {
        return Err(GuidanceRefusal::SolutionStale {
            age,
            bound: max_solution_age,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
