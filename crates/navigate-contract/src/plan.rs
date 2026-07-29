//! Flight-plan vocabulary: plans, waypoints, constraints, roles.
//!
//! The model here is the exchange shape; execution (sequencing, capture,
//! procedure selection) lives in `navigate-fpl`.

use crate::kinematics::GeodeticPosition;

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
}

impl Waypoint {
    /// Builds a waypoint without constraints;
    /// [`Self::with_altitude`] adds one.
    #[must_use]
    pub const fn new(ident: String, position: GeodeticPosition) -> Self {
        Self {
            ident,
            position,
            altitude: None,
        }
    }

    /// This waypoint with an altitude constraint.
    #[must_use]
    pub fn with_altitude(mut self, constraint: AltitudeConstraint) -> Self {
        self.altitude = Some(constraint);
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

    /// Structural validation: a plan must have at least one waypoint and
    /// every coordinate must be plausible.
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
        }
        Ok(())
    }
}

/// Structural defects a plan can carry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
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
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::{FlightPlan, PlanRole, PlanValidationError, Waypoint};
    use crate::kinematics::GeodeticPosition;

    #[test]
    fn empty_plan_is_invalid() {
        let plan = FlightPlan::new("p1".into(), PlanRole::Mission, Vec::new());
        assert!(matches!(
            plan.validate(),
            Err(PlanValidationError::Empty { .. })
        ));
    }

    #[test]
    fn implausible_waypoint_is_named() {
        let bad = Waypoint::new("W1".into(), GeodeticPosition::new(f64::NAN, 0.0, 0.0));
        let plan = FlightPlan::new("p2".into(), PlanRole::Mission, vec![bad]);
        assert!(matches!(
            plan.validate(),
            Err(PlanValidationError::ImplausiblePosition { index: 0, .. })
        ));
    }
}
