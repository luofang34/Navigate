//! Leg vocabulary: the path terminator of the leg that ends at a
//! waypoint (NAV-LG-001).
//!
//! One vocabulary carries the RNAV path terminators and the vertical
//! procedure legs. One sequencer walks one leg order, and a departure
//! interleaves the two families (ADR-0006). A terminator that this
//! workspace cannot fly yet is still nameable here. A plan source states
//! the leg it holds, and the executor refuses it by name at activation
//! (NAV-LG-005). See `docs/leg-requirements.md` for the behavioral
//! requirements.

use core::f64::consts::TAU;
use core::fmt;

use super::{Ident, PlanValidationError, Waypoint};

/// North reference from which a published course is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum CourseReference {
    /// True north. This is the canonical reference of this workspace and
    /// the [`Default`] variant.
    #[default]
    True,
    /// Magnetic north. Procedure sources publish courses against magnetic
    /// north. A caller needs a magnetic variation model to fly one.
    Magnetic,
}

/// Direction of a turn, viewed from above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TurnDirection {
    /// Counterclockwise turn.
    Left,
    /// Clockwise turn.
    Right,
}

impl TurnDirection {
    /// Returns `-1.0` for [`Self::Left`] and `1.0` for [`Self::Right`].
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::Left => -1.0,
            Self::Right => 1.0,
        }
    }
}

/// Length of a hold's inbound leg (NAV-LG-015).
///
/// This value defines the size of the racetrack. It never ends the hold
/// (NAV-LG-011).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HoldLegLength {
    /// Inbound leg duration in seconds.
    Time {
        /// Duration in seconds, finite and positive.
        seconds: f64,
    },
    /// Inbound leg ground distance in meters.
    Distance {
        /// Ground distance in meters, finite and positive.
        meters: f64,
    },
}

/// Racetrack geometry of a hold at the waypoint that carries it
/// (NAV-LG-015).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoldGeometry {
    /// Inbound course in radians, measured at the fix in the direction of
    /// flight, in `[0, 2π)`.
    pub inbound_course_rad: f64,
    /// North reference of `inbound_course_rad`.
    pub reference: CourseReference,
    /// Direction of the turns in the hold.
    pub turn: TurnDirection,
    /// Length of the inbound leg.
    pub leg: HoldLegLength,
}

impl HoldGeometry {
    /// Builds a hold racetrack from its parts.
    #[must_use]
    pub const fn new(
        inbound_course_rad: f64,
        reference: CourseReference,
        turn: TurnDirection,
        leg: HoldLegLength,
    ) -> Self {
        Self {
            inbound_course_rad,
            reference,
            turn,
            leg,
        }
    }
}

/// Path terminator of the leg that ends at a waypoint (NAV-LG-001).
///
/// `InitialFix`, `TrackToFix`, `DirectToFix`, `CourseToFix`,
/// `CourseToAltitude`, `HoldToAltitude`, `HoldToFix`, and `HoldManual`
/// are flown. The remaining terminators are reserved so a plan source can
/// state them, and activation refuses them by name (NAV-LG-005).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LegPath {
    /// `IF` — published start fix of a procedure. Valid at the first
    /// waypoint only (NAV-LG-008).
    InitialFix,
    /// `TF` — great-circle track from the upstream fix to this fix
    /// (NAV-LG-007).
    TrackToFix,
    /// `DF` — direct path from the present position to this fix
    /// (NAV-LG-008).
    DirectToFix,
    /// `CF` — published course that terminates at this fix (NAV-LG-009).
    CourseToFix {
        /// Published course in radians, measured at the fix in the
        /// direction of flight, in `[0, 2π)`.
        course_rad: f64,
        /// North reference of `course_rad`.
        reference: CourseReference,
    },
    /// `CA` — published course that terminates when the vehicle reaches
    /// the waypoint's altitude constraint.
    CourseToAltitude {
        /// Published course in radians, measured at the waypoint position
        /// in the direction of flight, in `[0, 2π)`.
        course_rad: f64,
        /// North reference of `course_rad`.
        reference: CourseReference,
    },
    /// `RF` — constant-radius arc that terminates at this fix.
    RadiusToFix,
    /// `HA` — hold that ends when the vehicle reaches the waypoint's
    /// altitude constraint.
    HoldToAltitude {
        /// Racetrack geometry.
        hold: HoldGeometry,
    },
    /// `HF` — hold that ends at the hold fix after one circuit.
    HoldToFix {
        /// Racetrack geometry.
        hold: HoldGeometry,
    },
    /// `HM` — hold that ends on an external exit command.
    HoldManual {
        /// Racetrack geometry.
        hold: HoldGeometry,
    },
    /// Vertical climb from the departure point to a declared altitude.
    VerticalTakeoff,
    /// Held position that ends on an external command or condition.
    Hover,
    /// Change between rotor-borne and wing-borne flight.
    Transition,
    /// Vertical descent that ends on reported ground contact.
    VerticalLanding,
}

impl LegPath {
    /// Returns the procedure code for this terminator.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InitialFix => "IF",
            Self::TrackToFix => "TF",
            Self::DirectToFix => "DF",
            Self::CourseToFix { .. } => "CF",
            Self::CourseToAltitude { .. } => "CA",
            Self::RadiusToFix => "RF",
            Self::HoldToAltitude { .. } => "HA",
            Self::HoldToFix { .. } => "HF",
            Self::HoldManual { .. } => "HM",
            Self::VerticalTakeoff => "vertical takeoff",
            Self::Hover => "hover",
            Self::Transition => "transition",
            Self::VerticalLanding => "vertical landing",
        }
    }

    /// Returns the hold geometry when this terminator is `HA`, `HF`, or
    /// `HM`.
    #[must_use]
    pub const fn hold(&self) -> Option<HoldGeometry> {
        match *self {
            Self::HoldToAltitude { hold }
            | Self::HoldToFix { hold }
            | Self::HoldManual { hold } => Some(hold),
            _ => None,
        }
    }
}

impl fmt::Display for LegPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Validates one waypoint's declared path terminator (NAV-LG-004).
pub(super) fn validate_path(
    plan: Ident,
    index: usize,
    waypoint: &Waypoint,
) -> Result<(), PlanValidationError> {
    let Some(path) = waypoint.path else {
        return Ok(());
    };
    match path {
        LegPath::TrackToFix if index == 0 => Err(PlanValidationError::TrackWithoutUpstreamFix {
            plan,
            ident: waypoint.ident,
        }),
        LegPath::InitialFix if index != 0 => Err(PlanValidationError::InitialFixNotFirst {
            plan,
            index,
            ident: waypoint.ident,
        }),
        LegPath::CourseToFix { course_rad, .. } => {
            validate_course(plan, index, waypoint, course_rad)
        }
        LegPath::CourseToAltitude { course_rad, .. } => {
            validate_course(plan, index, waypoint, course_rad)?;
            if waypoint.altitude.is_none() {
                return Err(PlanValidationError::CourseToAltitudeWithoutAltitude {
                    plan,
                    index,
                    ident: waypoint.ident,
                });
            }
            Ok(())
        }
        LegPath::HoldToAltitude { hold } => {
            validate_hold(plan, index, waypoint, &hold)?;
            if waypoint.altitude.is_none() {
                return Err(PlanValidationError::HoldToAltitudeWithoutAltitude {
                    plan,
                    index,
                    ident: waypoint.ident,
                });
            }
            Ok(())
        }
        LegPath::HoldToFix { hold } | LegPath::HoldManual { hold } => {
            validate_hold(plan, index, waypoint, &hold)
        }
        _ => Ok(()),
    }
}

fn validate_course(
    plan: Ident,
    index: usize,
    waypoint: &Waypoint,
    course_rad: f64,
) -> Result<(), PlanValidationError> {
    if !course_rad.is_finite() || !(0.0..TAU).contains(&course_rad) {
        return Err(PlanValidationError::CourseOutOfRange {
            plan,
            index,
            ident: waypoint.ident,
            course_rad,
        });
    }
    Ok(())
}

fn validate_hold(
    plan: Ident,
    index: usize,
    waypoint: &Waypoint,
    hold: &HoldGeometry,
) -> Result<(), PlanValidationError> {
    validate_course(plan, index, waypoint, hold.inbound_course_rad)?;
    let length = match hold.leg {
        HoldLegLength::Time { seconds } => seconds,
        HoldLegLength::Distance { meters } => meters,
    };
    if !length.is_finite() || length <= 0.0 {
        return Err(PlanValidationError::HoldLegNotPositive {
            plan,
            index,
            ident: waypoint.ident,
            leg: hold.leg,
        });
    }
    if let Some(waypoint_turn) = waypoint.turn_direction
        && waypoint_turn != hold.turn
    {
        return Err(PlanValidationError::ConflictingTurnDirection {
            plan,
            index,
            ident: waypoint.ident,
            waypoint_turn,
            hold_turn: hold.turn,
        });
    }
    Ok(())
}
