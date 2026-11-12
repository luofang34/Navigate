//! A director over one flight plan with Navigate's sequencer and guidance.

use navigate_contract::{
    ClockDomainId, Ellipsoid, FlightPlan, Ident, MonotonicNanos, NavigationSolution, PlanRole,
    Waypoint,
};
use navigate_guidance::{guide, guide_velocity};
use navigate_route::{
    ExecutionConfig, ExecutionState, HoldStatus, PlanExecution, SequenceEvent, SequenceReason,
};

use super::{DirectorEvent, DirectorOutput, FlightDirector, GuidanceLaw, SequenceCause};
use crate::OnboardError;

/// Executes one flight plan on WGS84: leg sequencing with capture, turn
/// anticipation, and hold racetracks, plus the guidance law on the active
/// leg.
#[derive(Debug, Clone)]
pub struct RouteDirector {
    route_id: Ident,
    waypoints: Vec<Waypoint>,
    config: ExecutionConfig,
    law: GuidanceLaw,
    cruise_speed_mps: f64,
    clock: ClockDomainId,
    state: ExecutionState,
}

impl RouteDirector {
    /// Builds a director for `waypoints` named `route_id`.
    ///
    /// # Errors
    ///
    /// Refuses an unusable route identifier, a plan that does not activate,
    /// and a cruise speed that is not finite and positive.
    pub fn new(
        route_id: &str,
        waypoints: Vec<Waypoint>,
        config: ExecutionConfig,
        law: GuidanceLaw,
        cruise_speed_mps: f64,
        clock: ClockDomainId,
    ) -> Result<Self, OnboardError> {
        let invalid = |detail: String| OnboardError::Config {
            field: "route",
            detail,
        };
        if !(cruise_speed_mps.is_finite() && cruise_speed_mps > 0.0) {
            return Err(invalid(format!("cruise speed {cruise_speed_mps}")));
        }
        let route_id =
            Ident::try_new(route_id).map_err(|e| invalid(format!("{route_id}: {e:?}")))?;
        let state = {
            let plan = FlightPlan::new(route_id, PlanRole::Mission, Ellipsoid::WGS84, &waypoints);
            PlanExecution::new(plan, config)
                .map_err(|e| invalid(format!("{e}")))?
                .state()
        };
        Ok(Self {
            route_id,
            waypoints,
            config,
            law,
            cruise_speed_mps,
            clock,
            state,
        })
    }

    fn execution(&self) -> Result<PlanExecution<'_>, navigate_route::PlanActivationError> {
        let plan = FlightPlan::new(
            self.route_id,
            PlanRole::Mission,
            Ellipsoid::WGS84,
            &self.waypoints,
        );
        PlanExecution::resume_state(plan, self.config, self.state)
    }

    /// Arms manual exit when the active leg is a manual hold (`HM`).
    pub fn exit_hold(&mut self) -> bool {
        let Ok(mut execution) = self.execution() else {
            return false;
        };
        let armed = execution.exit_hold();
        self.state = execution.state();
        armed
    }

    /// Returns observable progress when the active leg is a hold.
    #[must_use]
    pub const fn hold_status(&self) -> Option<HoldStatus> {
        self.state.hold_status()
    }

    /// Speed on the leg toward waypoint `index`: the cruise speed under the
    /// limit of that waypoint.
    #[must_use]
    pub fn leg_speed(&self, index: usize) -> f64 {
        self.waypoints
            .get(index)
            .and_then(|w| w.max_speed_mps)
            .map_or(self.cruise_speed_mps, |limit| {
                limit.min(self.cruise_speed_mps)
            })
    }
}

/// The cause of a sequencing decision, each known reason by name.
fn cause(reason: SequenceReason) -> SequenceCause {
    match reason {
        SequenceReason::Anticipated => SequenceCause::Anticipated,
        SequenceReason::Overflown => SequenceCause::Overflown,
        SequenceReason::Passed { cross_track_m } => SequenceCause::Passed { cross_track_m },
        _ => SequenceCause::Other,
    }
}

impl FlightDirector for RouteDirector {
    fn step(
        &mut self,
        solution: Option<&NavigationSolution>,
        now: MonotonicNanos,
    ) -> DirectorOutput {
        let index = self.state.active_index();
        let mut output = DirectorOutput {
            setpoint: None,
            commanded_speed_mps: self.leg_speed(index),
            event: None,
            active_index: self.active_index(),
            refusal: None,
        };
        let (Some(solution), Ok(mut execution)) = (solution, self.execution()) else {
            return output;
        };
        output.event = match execution.advance(&solution.position, self.leg_speed(index)) {
            SequenceEvent::LegAdvanced { reason, .. } => Some((false, reason)),
            SequenceEvent::PlanComplete { reason, .. } => Some((true, reason)),
            _ => None,
        }
        .map(|(plan_complete, reason)| DirectorEvent {
            waypoint_index: index,
            plan_complete,
            cause: cause(reason),
        });
        let leg = execution.active_leg();
        output.active_index = leg.map(|leg| leg.index);
        if let Some(leg) = leg {
            output.commanded_speed_mps = self.leg_speed(leg.index);
            let command = match &self.law {
                GuidanceLaw::DeviationTracking(config) => guide(
                    solution,
                    leg.lateral_reference(),
                    leg.to,
                    now,
                    self.clock,
                    config,
                ),
                GuidanceLaw::Velocity(config) => guide_velocity(
                    solution,
                    leg.lateral_reference(),
                    leg.to,
                    now,
                    self.clock,
                    config,
                ),
            };
            match command {
                Ok(command) => output.setpoint = Some(command.setpoint),
                Err(refusal) => output.refusal = Some(refusal),
            }
        }
        self.state = execution.state();
        output
    }

    fn active_index(&self) -> Option<usize> {
        (!self.state.is_complete() && self.state.active_index() < self.waypoints.len())
            .then_some(self.state.active_index())
    }

    fn snapshot(&self) -> Box<dyn FlightDirector> {
        Box::new(self.clone())
    }

    fn restore(&mut self, snapshot: &dyn FlightDirector) -> Result<(), OnboardError> {
        let other =
            snapshot
                .as_any()
                .downcast_ref::<Self>()
                .ok_or_else(|| OnboardError::Config {
                    field: "snapshot",
                    detail: "a snapshot of another director type".into(),
                })?;
        self.clone_from(other);
        Ok(())
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

#[cfg(test)]
mod tests;
