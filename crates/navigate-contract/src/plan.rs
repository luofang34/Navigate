//! Flight-plan vocabulary: plans, waypoints, constraints, roles.
//!
//! The model here is the exchange shape; execution (sequencing, capture,
//! procedure selection) lives in `navigate-fpl`. Turn types, the window
//! altitude form, and the per-leg speed and gradient constraints follow
//! the requirements in `docs/procedure-requirements.md`.

use core::fmt;

use crate::kinematics::GeodeticPosition;

pub mod leg;

pub use leg::{CourseReference, LegPath};

#[cfg(test)]
mod tests;

/// How the turn at a waypoint is flown (NAV-TT-001).
///
/// The type comes from the plan source and is never inferred by an
/// executor. A plan's final waypoint is terminal: there is no outbound
/// course to anticipate, so its turn type has no effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum TurnType {
    /// Turn anticipation: the turn begins before the fix and rolls out
    /// on the next course. The RNAV default, and this type's [`Default`].
    #[default]
    FlyBy,
    /// The fix is crossed before the turn begins — the missed-approach
    /// and missed-approach-holding cases.
    FlyOver,
}

/// An altitude constraint on a waypoint, meters above the WGS84
/// ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AltitudeConstraint {
    /// Cross at exactly this altitude.
    At(f64),
    /// Cross at or above this altitude.
    AtOrAbove(f64),
    /// Cross at or below this altitude.
    AtOrBelow(f64),
    /// Cross between two altitudes (NAV-VC-001). `lower_m` below
    /// `upper_m`; a window that does not bracket a band is refused by
    /// [`FlightPlan::validate`].
    Window {
        /// Lower bound of the window, meters.
        lower_m: f64,
        /// Upper bound of the window, meters.
        upper_m: f64,
    },
}

/// One waypoint of a flight plan.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Waypoint {
    /// Human-meaningful identifier (fix name, generated id).
    pub ident: String,
    /// Waypoint position. The altitude component is advisory when an
    /// explicit constraint is present.
    pub position: GeodeticPosition,
    /// Altitude constraint at this waypoint, if any.
    pub altitude: Option<AltitudeConstraint>,
    /// How the turn at this waypoint is flown.
    pub turn: TurnType,
    /// Maximum along-track speed on the leg toward this waypoint, meters
    /// per second (NAV-VC-003). Must be finite and positive;
    /// [`FlightPlan::validate`] refuses anything else.
    pub max_speed_mps: Option<f64>,
    /// Vertical gradient of the leg toward this waypoint: height in
    /// meters per along-track meter (NAV-VC-002). Procedure sources
    /// express gradients in feet per nautical mile; the canonical form
    /// is dimensionless m/m. Descent gradients may be expressed
    /// negative — consumers limit a rate magnitude, so the sign carries
    /// intent, not the limit. Zero is refused by
    /// [`FlightPlan::validate`]: absence, not zero, expresses "no
    /// gradient limit".
    pub gradient: Option<f64>,
    /// Path terminator of the leg toward this waypoint (NAV-LG-001).
    /// `None` states that the plan source declared none; the executor
    /// resolves it by position in the fly order — the first waypoint to
    /// [`LegPath::DirectToFix`], every later one to
    /// [`LegPath::TrackToFix`] (NAV-LG-002).
    pub path: Option<LegPath>,
}

impl Waypoint {
    /// Builds a waypoint without constraints, flown as
    /// [`TurnType::FlyBy`]; the `with_*` builders add the rest.
    #[must_use]
    pub const fn new(ident: String, position: GeodeticPosition) -> Self {
        Self {
            ident,
            position,
            altitude: None,
            turn: TurnType::FlyBy,
            max_speed_mps: None,
            gradient: None,
            path: None,
        }
    }

    /// This waypoint with an altitude constraint.
    #[must_use]
    pub fn with_altitude(mut self, constraint: AltitudeConstraint) -> Self {
        self.altitude = Some(constraint);
        self
    }

    /// This waypoint with an explicit turn type.
    #[must_use]
    pub fn with_turn(mut self, turn: TurnType) -> Self {
        self.turn = turn;
        self
    }

    /// This waypoint with a maximum-speed constraint on the leg toward
    /// it, meters per second.
    #[must_use]
    pub fn with_max_speed(mut self, max_speed_mps: f64) -> Self {
        self.max_speed_mps = Some(max_speed_mps);
        self
    }

    /// This waypoint with a vertical gradient on the leg toward it,
    /// height in meters per along-track meter.
    #[must_use]
    pub fn with_gradient(mut self, gradient: f64) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// This waypoint with an explicit path terminator on the leg toward
    /// it (NAV-LG-001).
    #[must_use]
    pub fn with_path(mut self, path: LegPath) -> Self {
        self.path = Some(path);
        self
    }
}

/// Why a plan exists. A loss-of-communication procedure is a plan with a
/// different role, not a different machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PlanRole {
    /// The primary mission plan.
    Mission,
    /// The procedure flown on sustained loss of communication.
    LossOfComm,
    /// A contingency procedure selected by explicit command.
    Contingency,
}

/// A validated sequence of waypoints with a role.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FlightPlan {
    /// Plan identifier, unique within a deployment's plan store.
    pub id: String,
    /// Why this plan exists.
    pub role: PlanRole,
    /// Waypoints in fly order.
    pub waypoints: Vec<Waypoint>,
}

impl FlightPlan {
    /// Builds a plan from its parts. Call [`Self::validate`] before
    /// executing it.
    #[must_use]
    pub const fn new(id: String, role: PlanRole, waypoints: Vec<Waypoint>) -> Self {
        Self {
            id,
            role,
            waypoints,
        }
    }

    /// Structural validation: a plan must have at least one waypoint,
    /// every coordinate must be plausible, every declared constraint
    /// must be a number that can be flown — finite, and an altitude
    /// window that brackets a band — and every declared path terminator
    /// must agree with its position in the fly order (NAV-LG-004).
    ///
    /// Whether a terminator is implemented is the executor's property,
    /// not this type's: a plan carrying a reserved leg is a valid plan,
    /// and activation refuses it by name (NAV-LG-005).
    ///
    /// # Errors
    ///
    /// Returns the first structural defect found.
    pub fn validate(&self) -> Result<(), PlanValidationError> {
        if self.waypoints.is_empty() {
            return Err(PlanValidationError::Empty {
                plan: self.id.clone(),
            });
        }
        for (index, waypoint) in self.waypoints.iter().enumerate() {
            if !waypoint.position.is_plausible() {
                return Err(PlanValidationError::ImplausiblePosition {
                    plan: self.id.clone(),
                    index,
                    ident: waypoint.ident.clone(),
                });
            }
            validate_constraints(&self.id, index, waypoint)?;
            leg::validate_path(&self.id, index, waypoint)?;
        }
        Ok(())
    }
}

/// Screens one waypoint's declared constraints. A non-finite bound would
/// silently disable every comparison a consumer makes against it, so it
/// is refused here rather than clamped in flight.
fn validate_constraints(
    plan: &str,
    index: usize,
    waypoint: &Waypoint,
) -> Result<(), PlanValidationError> {
    let defect = |field| PlanValidationError::NonFiniteConstraint {
        plan: plan.to_owned(),
        index,
        ident: waypoint.ident.clone(),
        field,
    };
    // The match is deliberately exhaustive without a wildcard: this is
    // the defining crate, so a future AltitudeConstraint variant becomes
    // a compile error here instead of silently skipping every screen.
    match waypoint.altitude {
        None => {}
        Some(
            AltitudeConstraint::At(altitude_m)
            | AltitudeConstraint::AtOrAbove(altitude_m)
            | AltitudeConstraint::AtOrBelow(altitude_m),
        ) if !altitude_m.is_finite() => {
            return Err(defect(ConstraintField::Altitude));
        }
        Some(
            AltitudeConstraint::At(_)
            | AltitudeConstraint::AtOrAbove(_)
            | AltitudeConstraint::AtOrBelow(_),
        ) => {}
        Some(AltitudeConstraint::Window { lower_m, upper_m }) => {
            if !lower_m.is_finite() || !upper_m.is_finite() {
                return Err(defect(ConstraintField::Altitude));
            }
            if lower_m >= upper_m {
                return Err(PlanValidationError::InvertedAltitudeWindow {
                    plan: plan.to_owned(),
                    index,
                    ident: waypoint.ident.clone(),
                });
            }
        }
    }
    if let Some(max_speed_mps) = waypoint.max_speed_mps {
        if !max_speed_mps.is_finite() {
            return Err(defect(ConstraintField::MaxSpeed));
        }
        // A zero or negative demanded speed cannot fly a leg; guidance's
        // plain min over the constraint is honest only because this
        // refusal exists (NAV-VC-003).
        if max_speed_mps <= 0.0 {
            return Err(PlanValidationError::NonPositiveSpeedConstraint {
                plan: plan.to_owned(),
                index,
                ident: waypoint.ident.clone(),
                max_speed_mps,
            });
        }
    }
    if let Some(gradient) = waypoint.gradient {
        if !gradient.is_finite() {
            return Err(defect(ConstraintField::Gradient));
        }
        // A zero gradient cannot fly any profile change; absence — not
        // zero — expresses "no gradient limit" (NAV-VC-002).
        if gradient == 0.0 {
            return Err(PlanValidationError::ZeroGradient {
                plan: plan.to_owned(),
                index,
                ident: waypoint.ident.clone(),
            });
        }
    }
    Ok(())
}

/// Which declared quantity a validation refusal names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConstraintField {
    /// A bound of the altitude constraint.
    Altitude,
    /// The maximum-speed constraint.
    MaxSpeed,
    /// The vertical gradient.
    Gradient,
}

impl fmt::Display for ConstraintField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Altitude => "altitude constraint",
            Self::MaxSpeed => "maximum-speed constraint",
            Self::Gradient => "gradient",
        };
        formatter.write_str(name)
    }
}

/// Structural defects a plan can carry.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PlanValidationError {
    /// The plan holds no waypoints.
    #[error("plan {plan} has no waypoints")]
    Empty {
        /// Offending plan id.
        plan: String,
    },
    /// A waypoint coordinate is non-finite or out of geodetic range.
    #[error("plan {plan} waypoint {index} ({ident}) has an implausible position")]
    ImplausiblePosition {
        /// Offending plan id.
        plan: String,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: String,
    },
    /// A waypoint's altitude window does not bracket a band: its lower
    /// bound is at or above its upper bound.
    #[error(
        "plan {plan} waypoint {index} ({ident}) has an altitude window whose lower bound is not below its upper bound"
    )]
    InvertedAltitudeWindow {
        /// Offending plan id.
        plan: String,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: String,
    },
    /// A declared constraint on a waypoint is not a finite number.
    #[error("plan {plan} waypoint {index} ({ident}) has a non-finite {field}")]
    NonFiniteConstraint {
        /// Offending plan id.
        plan: String,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: String,
        /// Which declared quantity is non-finite.
        field: ConstraintField,
    },
    /// A waypoint demands a speed no vehicle can fly a leg at.
    #[error(
        "plan {plan} waypoint {index} ({ident}) demands a non-positive speed {max_speed_mps} m/s"
    )]
    NonPositiveSpeedConstraint {
        /// Offending plan id.
        plan: String,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: String,
        /// The speed that cannot be flown.
        max_speed_mps: f64,
    },
    /// A waypoint declares a zero gradient, which cannot fly any profile
    /// change; absence expresses "no gradient limit".
    #[error("plan {plan} waypoint {index} ({ident}) declares a zero gradient")]
    ZeroGradient {
        /// Offending plan id.
        plan: String,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: String,
    },
    /// The first waypoint declares a track-to-fix leg, which needs a fix
    /// before it to run from (NAV-LG-004).
    #[error("plan {plan} waypoint 0 ({ident}) declares a TF leg with no fix before it")]
    TrackWithoutUpstreamFix {
        /// Offending plan id.
        plan: String,
        /// Waypoint identifier.
        ident: String,
    },
    /// A waypoint after the first declares an initial-fix leg, which
    /// starts a procedure and cannot sit inside one (NAV-LG-004).
    #[error("plan {plan} waypoint {index} ({ident}) declares an IF leg after the first waypoint")]
    InitialFixNotFirst {
        /// Offending plan id.
        plan: String,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: String,
    },
    /// A course-to-fix leg declares a course that is not a finite angle
    /// in `[0, 2π)`, so no vehicle can fly it (NAV-LG-004).
    #[error(
        "plan {plan} waypoint {index} ({ident}) declares a course {course_rad} outside [0, 2π)"
    )]
    CourseOutOfRange {
        /// Offending plan id.
        plan: String,
        /// Waypoint index in fly order.
        index: usize,
        /// Waypoint identifier.
        ident: String,
        /// The course that cannot be flown.
        course_rad: f64,
    },
}
