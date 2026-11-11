//! A director over one flight plan with Navigate's sequencer and guidance.

use navigate_contract::{
    ClockDomainId, Ellipsoid, FlightPlan, Ident, MonotonicNanos, NavigationSolution, PlanRole,
    Waypoint,
};
use navigate_guidance::{guide, guide_velocity};
use navigate_route::{ExecutionConfig, PlanExecution, SequenceEvent, SequenceReason};

use super::{DirectorEvent, DirectorOutput, FlightDirector, GuidanceLaw};
use crate::OnboardError;

/// Executes one flight plan on WGS84: leg sequencing with capture and
/// turn anticipation, and the guidance law on the active leg.
#[derive(Debug, Clone)]
pub struct RouteDirector<'w> {
    execution: PlanExecution<'w>,
    waypoints: &'w [Waypoint],
    law: GuidanceLaw,
    cruise_speed_mps: f64,
    clock: ClockDomainId,
}

impl<'w> RouteDirector<'w> {
    /// A director for `waypoints` named `route_id`.
    ///
    /// # Errors
    ///
    /// Refuses an unusable route identifier, a plan that does not activate,
    /// and a cruise speed that is not finite and positive.
    pub fn new(
        route_id: &str,
        waypoints: &'w [Waypoint],
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
        let ident = Ident::try_new(route_id).map_err(|e| invalid(format!("{route_id}: {e:?}")))?;
        let plan = FlightPlan::new(ident, PlanRole::Mission, Ellipsoid::WGS84, waypoints);
        let execution = PlanExecution::new(plan, config).map_err(|e| invalid(format!("{e}")))?;
        Ok(Self {
            execution,
            waypoints,
            law,
            cruise_speed_mps,
            clock,
        })
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

impl FlightDirector for RouteDirector<'_> {
    fn step(
        &mut self,
        solution: Option<&NavigationSolution>,
        now: MonotonicNanos,
    ) -> DirectorOutput {
        let index = self.execution.active_index();
        let mut output = DirectorOutput {
            setpoint: None,
            commanded_speed_mps: self.leg_speed(index),
            event: None,
            active_index: self.active_index(),
            refusal: None,
        };
        let Some(solution) = solution else {
            return output;
        };
        output.event = match self
            .execution
            .advance(&solution.position, self.leg_speed(index))
        {
            SequenceEvent::LegAdvanced { reason, .. } => Some((false, reason)),
            SequenceEvent::PlanComplete { reason, .. } => Some((true, reason)),
            _ => None,
        }
        .map(|(plan_complete, reason)| DirectorEvent {
            waypoint_index: index,
            plan_complete,
            anticipated: reason == SequenceReason::Anticipated,
        });
        let leg = self.execution.active_leg();
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
        output
    }

    fn active_index(&self) -> Option<usize> {
        self.execution.active_leg().map(|leg| leg.index)
    }
}

#[cfg(test)]
mod tests;
