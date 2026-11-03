//! Flight-plan vocabulary: plans, waypoints, constraints, roles.
//!
//! The model here is the exchange shape; execution (sequencing, capture,
//! procedure selection) lives in `navigate-fpl`. Turn types, the window
//! altitude form, and the per-leg speed and gradient constraints follow
//! the requirements in `docs/procedure-requirements.md`.
//!
//! The vocabulary does not allocate (ADR-0015). An [`Ident`] holds its
//! text in place, a [`FlightPlan`] borrows its waypoints, and every
//! [`PlanValidationError`] is `Copy`.

use crate::frame::Ellipsoid;
use crate::kinematics::GeodeticPosition;

mod error;
pub mod ident;
pub mod leg;

pub use error::{ConstraintField, PlanValidationError};
pub use ident::{Ident, IdentError, MAX_IDENT_BYTES};
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

/// An altitude constraint on a waypoint, in meters above the reference
/// ellipsoid of the plan ([`FlightPlan::ellipsoid`]).
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
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Waypoint {
    /// Human-meaningful identifier (fix name, generated id).
    pub ident: Ident,
    /// Waypoint position on the plan ellipsoid. The altitude component
    /// is advisory when an explicit constraint is present.
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
    pub const fn new(ident: Ident, position: GeodeticPosition) -> Self {
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
    pub const fn with_altitude(mut self, constraint: AltitudeConstraint) -> Self {
        self.altitude = Some(constraint);
        self
    }

    /// This waypoint with an explicit turn type.
    #[must_use]
    pub const fn with_turn(mut self, turn: TurnType) -> Self {
        self.turn = turn;
        self
    }

    /// This waypoint with a maximum-speed constraint on the leg toward
    /// it, meters per second.
    #[must_use]
    pub const fn with_max_speed(mut self, max_speed_mps: f64) -> Self {
        self.max_speed_mps = Some(max_speed_mps);
        self
    }

    /// This waypoint with a vertical gradient on the leg toward it,
    /// height in meters per along-track meter.
    #[must_use]
    pub const fn with_gradient(mut self, gradient: f64) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// This waypoint with an explicit path terminator on the leg toward
    /// it (NAV-LG-001).
    #[must_use]
    pub const fn with_path(mut self, path: LegPath) -> Self {
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

/// A sequence of waypoints with a role, on one reference ellipsoid.
///
/// The plan borrows its waypoints, so it does not allocate and it is
/// `Copy`. The owner of the waypoint storage chooses where the storage
/// is: a static table, a stack array, or a buffer of the host.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FlightPlan<'a> {
    /// Plan identifier, unique within a deployment's plan store.
    pub id: Ident,
    /// Why this plan exists.
    pub role: PlanRole,
    /// The reference ellipsoid of every waypoint position and of every
    /// altitude constraint (ADR-0013).
    pub ellipsoid: Ellipsoid,
    /// Waypoints in fly order.
    pub waypoints: &'a [Waypoint],
}

impl<'a> FlightPlan<'a> {
    /// Builds a plan from its parts. Call [`Self::validate`] before
    /// executing it.
    #[must_use]
    pub const fn new(
        id: Ident,
        role: PlanRole,
        ellipsoid: Ellipsoid,
        waypoints: &'a [Waypoint],
    ) -> Self {
        Self {
            id,
            role,
            ellipsoid,
            waypoints,
        }
    }

    /// Structural validation: the plan ellipsoid must be plausible, a
    /// plan must have at least one waypoint, every coordinate must be
    /// plausible, every declared constraint must be a number that can be
    /// flown — finite, and an altitude window that brackets a band — and
    /// every declared path terminator must agree with its position in
    /// the fly order (NAV-LG-004).
    ///
    /// Whether a terminator is implemented is the executor's property,
    /// not this type's: a plan carrying a reserved leg is a valid plan,
    /// and activation refuses it by name (NAV-LG-005).
    ///
    /// # Errors
    ///
    /// Returns the first structural defect found.
    pub fn validate(&self) -> Result<(), PlanValidationError> {
        if !self.ellipsoid.is_plausible() {
            return Err(PlanValidationError::ImplausibleEllipsoid { plan: self.id });
        }
        if self.waypoints.is_empty() {
            return Err(PlanValidationError::Empty { plan: self.id });
        }
        for (index, waypoint) in self.waypoints.iter().enumerate() {
            if !waypoint.position.is_plausible() {
                return Err(PlanValidationError::ImplausiblePosition {
                    plan: self.id,
                    index,
                    ident: waypoint.ident,
                });
            }
            validate_constraints(self.id, index, waypoint)?;
            leg::validate_path(self.id, index, waypoint)?;
        }
        Ok(())
    }
}

/// Screens one waypoint's declared constraints. A non-finite bound would
/// silently disable every comparison a consumer makes against it, so it
/// is refused here rather than clamped in flight.
fn validate_constraints(
    plan: Ident,
    index: usize,
    waypoint: &Waypoint,
) -> Result<(), PlanValidationError> {
    let ident = waypoint.ident;
    let defect = |field| PlanValidationError::NonFiniteConstraint {
        plan,
        index,
        ident,
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
                return Err(PlanValidationError::InvertedAltitudeWindow { plan, index, ident });
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
                plan,
                index,
                ident,
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
            return Err(PlanValidationError::ZeroGradient { plan, index, ident });
        }
    }
    Ok(())
}
