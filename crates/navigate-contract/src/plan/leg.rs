//! Leg vocabulary: the path terminator of the leg that ends at a
//! waypoint (NAV-LG-001).
//!
//! One vocabulary carries the RNAV path terminators and the vertical
//! procedure legs, because one sequencer walks one leg order and a
//! departure interleaves the two families (ADR-0006). A terminator this
//! workspace cannot fly yet is still nameable here: a plan source states
//! the leg it holds, and the executor refuses it by name at activation
//! (NAV-LG-005). The behavioral requirements are in
//! `docs/leg-requirements.md`.

use core::f64::consts::TAU;
use core::fmt;

use super::{PlanValidationError, Waypoint};

/// North reference a published course is measured from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum CourseReference {
    /// True north. The canonical reference of this workspace, and this
    /// type's [`Default`].
    #[default]
    True,
    /// Magnetic north. Procedure sources publish courses this way. A
    /// consumer needs a magnetic variation model to fly one.
    Magnetic,
}

/// Path terminator of the leg that ends at a waypoint (NAV-LG-001).
///
/// The first four terminators are flown. The rest are reserved: they
/// name a leg a plan source may state, and the executor refuses them by
/// name until each one's sequencer lands (NAV-LG-005). A reserved
/// terminator carries no parameters, because the parameters follow from
/// the sequencer that flies it;
/// `docs/leg-requirements.md` lists what each one needs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LegPath {
    /// `IF` — the published start fix of a procedure. It declares no
    /// path, so the executor flies it from the present position. Valid
    /// at the first waypoint only (NAV-LG-008).
    InitialFix,
    /// `TF` — a great-circle track from the fix before it to this fix.
    /// The standard RNAV leg (NAV-LG-007).
    TrackToFix,
    /// `DF` — a direct path from the present position to this fix. No
    /// published inbound track exists (NAV-LG-008).
    DirectToFix,
    /// `CF` — a published course that terminates at this fix
    /// (NAV-LG-009).
    CourseToFix {
        /// The published course in radians, measured at the fix in the
        /// direction of flight, in `[0, 2π)`.
        course_rad: f64,
        /// North reference `course_rad` is measured from.
        reference: CourseReference,
    },
    /// `RF` — a constant-radius arc that terminates at this fix.
    RadiusToFix,
    /// `HA` — a hold that ends when the vehicle reaches an altitude.
    HoldToAltitude,
    /// `HF` — a hold that ends at the hold fix after one circuit.
    HoldToFix,
    /// `HM` — a hold that ends on an external command.
    HoldManual,
    /// A vertical climb from the departure point, ending at a declared
    /// altitude.
    VerticalTakeoff,
    /// A held position, ending on an external command or a declared
    /// condition.
    Hover,
    /// A change between rotor-borne and wing-borne flight, ending when
    /// the vehicle reports the target flight mode.
    Transition,
    /// A vertical descent, ending on reported ground contact. Ground
    /// contact and not a zero altitude: an altitude estimate is not a
    /// statement that the vehicle is on the ground.
    VerticalLanding,
}

impl LegPath {
    /// The terminator's code, as procedure coding writes it. The
    /// vertical legs have no published code, so they read as names.
    ///
    /// The match is deliberately exhaustive without a wildcard: this is
    /// the defining crate, so a terminator added to the vocabulary
    /// becomes a compile error here instead of reading as another leg's
    /// name in every refusal message.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InitialFix => "IF",
            Self::TrackToFix => "TF",
            Self::DirectToFix => "DF",
            Self::CourseToFix { .. } => "CF",
            Self::RadiusToFix => "RF",
            Self::HoldToAltitude => "HA",
            Self::HoldToFix => "HF",
            Self::HoldManual => "HM",
            Self::VerticalTakeoff => "vertical takeoff",
            Self::Hover => "hover",
            Self::Transition => "transition",
            Self::VerticalLanding => "vertical landing",
        }
    }
}

impl fmt::Display for LegPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Screens one waypoint's declared path terminator (NAV-LG-004).
///
/// Structural defects only: a terminator that contradicts its position
/// in the fly order, or a course no vehicle can fly. Whether a
/// terminator is implemented is a property of the executor, not of this
/// exchange type, so a reserved terminator validates here and is
/// refused at activation.
pub(super) fn validate_path(
    plan: &str,
    index: usize,
    waypoint: &Waypoint,
) -> Result<(), PlanValidationError> {
    let Some(path) = waypoint.path else {
        return Ok(());
    };
    match path {
        LegPath::TrackToFix if index == 0 => Err(PlanValidationError::TrackWithoutUpstreamFix {
            plan: plan.to_owned(),
            ident: waypoint.ident.clone(),
        }),
        LegPath::InitialFix if index != 0 => Err(PlanValidationError::InitialFixNotFirst {
            plan: plan.to_owned(),
            index,
            ident: waypoint.ident.clone(),
        }),
        LegPath::CourseToFix { course_rad, .. }
            if !course_rad.is_finite() || !(0.0..TAU).contains(&course_rad) =>
        {
            Err(PlanValidationError::CourseOutOfRange {
                plan: plan.to_owned(),
                index,
                ident: waypoint.ident.clone(),
                course_rad,
            })
        }
        _ => Ok(()),
    }
}
