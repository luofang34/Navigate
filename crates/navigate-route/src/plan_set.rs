//! Role-keyed procedure store and activation.
//!
//! A [`PlanSet`] carries the procedures loaded for one vehicle: the
//! mission plan plus optional loss-of-communication and contingency
//! procedures. Activation turns one of them into a
//! [`PlanExecution`]. Selecting which role applies when — normal
//! operation, sustained link loss, explicit contingency command — is the
//! host platform's decision; this crate only executes the selection.

use navigate_contract::{FlightPlan, Ident, LegPath, PlanRole, PlanValidationError};

use crate::execution::{ExecutionConfig, PlanExecution};

/// The procedures loaded for one vehicle, keyed by [`PlanRole`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PlanSet<'a> {
    /// The primary mission plan.
    pub mission: FlightPlan<'a>,
    /// The procedure flown on sustained loss of communication, if
    /// loaded.
    pub loss_of_comm: Option<FlightPlan<'a>>,
    /// The contingency procedure selected by explicit command, if
    /// loaded.
    pub contingency: Option<FlightPlan<'a>>,
}

impl<'a> PlanSet<'a> {
    /// A set holding only the mission plan; the `with_*` builders add
    /// the optional procedures.
    #[must_use]
    pub const fn new(mission: FlightPlan<'a>) -> Self {
        Self {
            mission,
            loss_of_comm: None,
            contingency: None,
        }
    }

    /// This set with a loss-of-communication procedure loaded.
    #[must_use]
    pub const fn with_loss_of_comm(mut self, plan: FlightPlan<'a>) -> Self {
        self.loss_of_comm = Some(plan);
        self
    }

    /// This set with a contingency procedure loaded.
    #[must_use]
    pub const fn with_contingency(mut self, plan: FlightPlan<'a>) -> Self {
        self.contingency = Some(plan);
        self
    }

    /// Starts executing the plan loaded for `role`. When to activate
    /// which role is the host platform's decision; this only refuses
    /// what cannot be flown.
    ///
    /// # Errors
    ///
    /// [`PlanActivationError::RoleUnavailable`] when no plan is loaded
    /// for the role; [`PlanActivationError::Invalid`] when the selected
    /// plan fails structural validation.
    pub fn activate(
        &self,
        role: PlanRole,
        config: ExecutionConfig,
    ) -> Result<PlanExecution<'a>, PlanActivationError> {
        let plan = match role {
            PlanRole::Mission => Some(self.mission),
            PlanRole::LossOfComm => self.loss_of_comm,
            PlanRole::Contingency => self.contingency,
            // `PlanRole` is non-exhaustive: a role this crate does not
            // know yet has no slot here.
            _ => None,
        };
        let plan = plan.ok_or(PlanActivationError::RoleUnavailable { role })?;
        PlanExecution::new(plan, config)
    }
}

/// Why a plan could not become an execution.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PlanActivationError {
    /// No plan is loaded for the requested role.
    #[error("no plan is loaded for role {role:?}")]
    RoleUnavailable {
        /// The role requested.
        role: PlanRole,
    },
    /// The selected plan fails structural validation.
    #[error(transparent)]
    Invalid(#[from] PlanValidationError),
    /// A waypoint's speed constraint is below the approach-speed floor
    /// (NAV-VC-003): refused at activation, never clamped in flight.
    #[error(
        "plan {plan} waypoint {ident} demands {max_speed_mps} m/s, below the {floor_mps} m/s approach floor"
    )]
    SpeedBelowFloor {
        /// Offending plan id.
        plan: Ident,
        /// Offending waypoint identifier.
        ident: Ident,
        /// The constraint that cannot be flown.
        max_speed_mps: f64,
        /// The configured floor it violates.
        floor_mps: f64,
    },
    /// A fixed leg is no longer than the capture radius: the leg would
    /// sequence while the vehicle is still at the previous fix.
    #[error(
        "plan {plan} leg to {ident} is {leg_length_m} m, no longer than the {capture_radius_m} m capture radius"
    )]
    CaptureRadiusExceedsLeg {
        /// Offending plan id.
        plan: Ident,
        /// The waypoint the too-short leg runs to.
        ident: Ident,
        /// The fixed leg's length.
        leg_length_m: f64,
        /// The configured capture radius it does not clear.
        capture_radius_m: f64,
    },
    /// An execution-config value cannot feed the comparisons it exists
    /// for (non-finite, non-positive, or an unflyable bank limit).
    #[error("execution config {field} = {value} is not flyable")]
    InvalidConfig {
        /// The offending config field.
        field: &'static str,
        /// The rejected value.
        value: f64,
    },
    /// A leg declares a path terminator this build has no sequencer for
    /// (NAV-LG-005). The plan is valid; this executor cannot fly it.
    #[error("plan {plan} leg {index} ({ident}) declares a {path} leg this build cannot fly")]
    UnsupportedLegKind {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint index in fly order.
        index: usize,
        /// The waypoint the leg runs to.
        ident: Ident,
        /// The path terminator that has no sequencer.
        path: LegPath,
    },
    /// A course-to-fix leg declares its course against magnetic north.
    /// Navigate has no magnetic variation model, and it does not guess
    /// a variation (NAV-LG-005).
    #[error(
        "plan {plan} leg {index} ({ident}) declares a magnetic course {course_rad} and no variation model exists"
    )]
    MagneticCourseUnsupported {
        /// Offending plan id.
        plan: Ident,
        /// Waypoint index in fly order.
        index: usize,
        /// The waypoint the leg runs to.
        ident: Ident,
        /// The declared magnetic course.
        course_rad: f64,
    },
}

#[cfg(test)]
mod tests {
    use navigate_contract::{Ellipsoid, GeodeticPosition, Waypoint};

    use super::*;

    fn id(text: &str) -> Ident {
        Ident::try_new(text).expect("valid ident")
    }

    fn home_waypoint() -> Waypoint {
        Waypoint::new(id("HOME"), GeodeticPosition::new(0.0, 0.0, 0.0))
    }

    #[test]
    fn activating_a_missing_role_names_the_role() {
        let waypoints = [home_waypoint()];
        let mission = FlightPlan::new(id("m"), PlanRole::Mission, Ellipsoid::WGS84, &waypoints);
        let set = PlanSet::new(mission);
        let refused = set.activate(PlanRole::LossOfComm, ExecutionConfig::default());
        assert!(matches!(
            refused,
            Err(PlanActivationError::RoleUnavailable {
                role: PlanRole::LossOfComm
            })
        ));
    }

    #[test]
    fn activating_a_loaded_role_starts_at_the_first_waypoint() {
        let waypoints = [home_waypoint()];
        let mission = FlightPlan::new(id("m"), PlanRole::Mission, Ellipsoid::WGS84, &waypoints);
        let loss_of_comm =
            FlightPlan::new(id("lc"), PlanRole::LossOfComm, Ellipsoid::WGS84, &waypoints);
        let set = PlanSet::new(mission).with_loss_of_comm(loss_of_comm);
        let exec = set
            .activate(PlanRole::LossOfComm, ExecutionConfig::default())
            .expect("loaded role activates");
        assert_eq!(exec.plan().id, "lc");
        assert_eq!(exec.active_index(), 0);
        assert!(!exec.is_complete());
    }

    #[test]
    fn activating_an_invalid_plan_reports_the_defect() {
        let waypoints = [home_waypoint()];
        let mission = FlightPlan::new(id("m"), PlanRole::Mission, Ellipsoid::WGS84, &waypoints);
        let empty = FlightPlan::new(id("bad"), PlanRole::Contingency, Ellipsoid::WGS84, &[]);
        let set = PlanSet::new(mission).with_contingency(empty);
        let refused = set.activate(PlanRole::Contingency, ExecutionConfig::default());
        assert!(matches!(
            refused,
            Err(PlanActivationError::Invalid(
                PlanValidationError::Empty { .. }
            ))
        ));
    }
}
