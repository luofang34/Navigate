//! The flight-director trait and the route director.

use navigate_contract::{GuidanceSetpoint, MonotonicNanos, NavigationSolution};
use navigate_guidance::{GuidanceConfig, GuidanceRefusal, VelocityGuidanceConfig};

mod route;
pub use route::RouteDirector;

/// The guidance law of a director.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GuidanceLaw {
    /// Deviation tracking: cross-track and vertical deviation of the leg
    /// (`navigate_guidance::guide`).
    DeviationTracking(GuidanceConfig),
    /// A velocity command along and toward the leg
    /// (`navigate_guidance::guide_velocity`).
    Velocity(VelocityGuidanceConfig),
}

/// A waypoint that the director sequenced at one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectorEvent {
    /// The sequenced waypoint.
    pub waypoint_index: usize,
    /// Whether it completed the plan.
    pub plan_complete: bool,
    /// Whether turn anticipation sequenced it.
    pub anticipated: bool,
}

/// What a director decided at one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectorOutput {
    /// The guidance setpoint, or `None` without a solution, after the plan,
    /// or on a refusal.
    pub setpoint: Option<GuidanceSetpoint>,
    /// Speed to fly on the active leg, m/s.
    pub commanded_speed_mps: f64,
    /// A sequencing decision, if one was made.
    pub event: Option<DirectorEvent>,
    /// The waypoint being flown to, or `None` once the plan is complete.
    pub active_index: Option<usize>,
    /// Why the guidance law refused the solution, if it did.
    pub refusal: Option<GuidanceRefusal>,
}

/// A flight director: route sequencing and guidance from a solution. It
/// holds no control authority, no command lease, and no actuator output.
pub trait FlightDirector {
    /// One step at `now` on the navigator's solution, if there is one.
    fn step(
        &mut self,
        solution: Option<&NavigationSolution>,
        now: MonotonicNanos,
    ) -> DirectorOutput;

    /// The waypoint being flown to, or `None` once the plan is complete.
    fn active_index(&self) -> Option<usize>;

    /// A complete copy of the state.
    fn snapshot(&self) -> Box<dyn FlightDirector>;

    /// Restores the state of `snapshot`.
    ///
    /// # Errors
    ///
    /// Refuses a snapshot of another director type.
    fn restore(&mut self, snapshot: &dyn FlightDirector) -> Result<(), crate::OnboardError>;

    /// The director as [`core::any::Any`], so that
    /// [`FlightDirector::restore`] can check the type of a snapshot.
    fn as_any(&self) -> &dyn core::any::Any;
}
