//! Path-terminator resolution and the activation screen that refuses a
//! leg this build cannot fly.
//!
//! Resolution and refusal live together because they form one judgment.
//! The screen decides what the sequencer may see, and resolution gives
//! the terminator that the sequencer executes. A leg kind without an
//! implemented termination condition never reaches `advance` (NAV-LG-005).

use navigate_contract::{CourseReference, FlightPlan, LegPath, Waypoint};

use crate::plan_set::PlanActivationError;

/// Resolves the path terminator of the leg toward the waypoint at `index`
/// (NAV-LG-002).
///
/// When a waypoint declares no path, the first waypoint resolves to
/// [`LegPath::DirectToFix`] and every later waypoint resolves to
/// [`LegPath::TrackToFix`].
pub(crate) const fn resolve_path(index: usize, declared: Option<LegPath>) -> LegPath {
    match declared {
        Some(path) => path,
        None if index == 0 => LegPath::DirectToFix,
        None => LegPath::TrackToFix,
    }
}

/// Returns `true` when the leg toward `path` has no two-fix minimum-length
/// guard at activation (NAV-LG-012).
pub(crate) const fn skips_minimum_length_guard(path: LegPath) -> bool {
    matches!(
        path,
        LegPath::CourseToAltitude { .. }
            | LegPath::HoldToAltitude { .. }
            | LegPath::HoldToFix { .. }
            | LegPath::HoldManual { .. }
    )
}

/// Refuses every leg that this build cannot fly, by name (NAV-LG-005).
///
/// # Errors
///
/// Returns [`PlanActivationError::UnsupportedLegKind`] for a reserved path
/// terminator and [`PlanActivationError::MagneticCourseUnsupported`] for a
/// magnetic course reference.
pub(crate) fn screen_leg_paths(plan: &FlightPlan<'_>) -> Result<(), PlanActivationError> {
    for (index, waypoint) in plan.waypoints.iter().enumerate() {
        screen_one(plan, index, waypoint)?;
    }
    Ok(())
}

fn screen_one(
    plan: &FlightPlan<'_>,
    index: usize,
    waypoint: &Waypoint,
) -> Result<(), PlanActivationError> {
    let path = resolve_path(index, waypoint.path);
    match path {
        LegPath::InitialFix | LegPath::TrackToFix | LegPath::DirectToFix => Ok(()),
        LegPath::CourseToFix {
            course_rad,
            reference,
        }
        | LegPath::CourseToAltitude {
            course_rad,
            reference,
        } => screen_reference(plan, index, waypoint, course_rad, reference),
        LegPath::HoldToAltitude { hold }
        | LegPath::HoldToFix { hold }
        | LegPath::HoldManual { hold } => screen_reference(
            plan,
            index,
            waypoint,
            hold.inbound_course_rad,
            hold.reference,
        ),
        other => Err(PlanActivationError::UnsupportedLegKind {
            plan: plan.id,
            index,
            ident: waypoint.ident,
            path: other,
        }),
    }
}

fn screen_reference(
    plan: &FlightPlan<'_>,
    index: usize,
    waypoint: &Waypoint,
    course_rad: f64,
    reference: CourseReference,
) -> Result<(), PlanActivationError> {
    match reference {
        CourseReference::True => Ok(()),
        CourseReference::Magnetic | _ => Err(PlanActivationError::MagneticCourseUnsupported {
            plan: plan.id,
            index,
            ident: waypoint.ident,
            course_rad,
        }),
    }
}
