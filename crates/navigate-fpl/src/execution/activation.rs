//! Path-terminator resolution and the activation screen that refuses a
//! leg this build cannot fly.
//!
//! Resolution and refusal live together because they are one judgment:
//! the screen decides what the sequencer may see, and resolution is what
//! the sequencer sees. A leg kind with no implemented termination
//! condition never reaches `advance`, so no sequencing guard can
//! misapply to a geometry it was not written for (NAV-LG-005).

use navigate_contract::{CourseReference, FlightPlan, LegPath, Waypoint};

use crate::plan_set::PlanActivationError;

/// The path terminator of the leg toward the waypoint at `index`
/// (NAV-LG-002).
///
/// A waypoint that declares none gets one by position in the fly order:
/// the first waypoint runs from the present position, and every later
/// waypoint runs on the great-circle track from the fix before it. This
/// is the implicit model the executor already flies, now named.
pub(crate) const fn resolve_path(index: usize, declared: Option<LegPath>) -> LegPath {
    match declared {
        Some(path) => path,
        None if index == 0 => LegPath::DirectToFix,
        None => LegPath::TrackToFix,
    }
}

/// Refuses every leg this build cannot fly, by name (NAV-LG-005).
///
/// # Errors
///
/// [`PlanActivationError::UnsupportedLegKind`] for a reserved path
/// terminator, and [`PlanActivationError::MagneticCourseUnsupported`]
/// for a course declared against magnetic north, which needs a magnetic
/// variation model this workspace does not have.
pub(crate) fn screen_leg_paths(plan: &FlightPlan) -> Result<(), PlanActivationError> {
    for (index, waypoint) in plan.waypoints.iter().enumerate() {
        screen_one(plan, index, waypoint)?;
    }
    Ok(())
}

fn screen_one(
    plan: &FlightPlan,
    index: usize,
    waypoint: &Waypoint,
) -> Result<(), PlanActivationError> {
    let path = resolve_path(index, waypoint.path);
    match path {
        LegPath::InitialFix | LegPath::TrackToFix | LegPath::DirectToFix => Ok(()),
        LegPath::CourseToFix {
            reference: CourseReference::True,
            ..
        } => Ok(()),
        LegPath::CourseToFix {
            reference: CourseReference::Magnetic,
            course_rad,
        } => Err(PlanActivationError::MagneticCourseUnsupported {
            plan: plan.id.clone(),
            index,
            ident: waypoint.ident.clone(),
            course_rad,
        }),
        // Fail closed: the vocabulary is non-exhaustive, so a terminator
        // this crate has never seen is refused with the reserved ones
        // rather than admitted by an optimistic wildcard.
        other => Err(PlanActivationError::UnsupportedLegKind {
            plan: plan.id.clone(),
            index,
            ident: waypoint.ident.clone(),
            path: other,
        }),
    }
}
