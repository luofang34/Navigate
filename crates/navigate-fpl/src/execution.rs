//! Active-leg sequencing over a validated flight plan.
//!
//! A [`PlanExecution`] walks a [`FlightPlan`] leg by leg. The active leg
//! runs from the previously captured waypoint — or from the present
//! position on the initial direct-to leg — to the next waypoint in fly
//! order. A waypoint captures when a caller-supplied position comes
//! within the configured capture radius; after the final waypoint
//! captures, the execution is complete and further advances are inert.

use navigate_contract::{FlightPlan, GeodeticPosition, PlanValidationError, Waypoint};
use navigate_geodesy::distance_m;

/// Tunable sequencing parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ExecutionConfig {
    /// Great-circle distance to the active `to` waypoint, in meters, at
    /// which the waypoint captures and the leg sequences.
    pub capture_radius_m: f64,
}

impl ExecutionConfig {
    /// Builds a config with the given capture radius in meters.
    #[must_use]
    pub const fn new(capture_radius_m: f64) -> Self {
        Self { capture_radius_m }
    }
}

/// 100 m: a fixed capture radius; turn anticipation is a designed
/// extension of the leg-transition criteria.
impl Default for ExecutionConfig {
    fn default() -> Self {
        Self::new(100.0)
    }
}

/// A borrowed view of the leg being flown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Leg<'a> {
    /// Origin waypoint; `None` on the initial direct-to leg, which is
    /// flown from the present position to the first waypoint.
    pub from: Option<&'a Waypoint>,
    /// Destination waypoint being flown toward.
    pub to: &'a Waypoint,
    /// Index of `to` in the plan's fly order.
    pub index: usize,
}

/// What one call to [`PlanExecution::advance`] decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SequenceEvent {
    /// No waypoint captured; the active leg is unchanged.
    None,
    /// The active `to` waypoint captured and the next leg is active.
    LegAdvanced {
        /// Index of the newly active `to` waypoint.
        to_index: usize,
    },
    /// The final waypoint captured; the plan is complete.
    PlanComplete,
}

/// Sequencing state for one activated flight plan.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanExecution {
    plan: FlightPlan,
    config: ExecutionConfig,
    active_index: usize,
    complete: bool,
}

impl PlanExecution {
    /// Validates the plan and starts execution on the initial direct-to
    /// leg toward the first waypoint.
    ///
    /// # Errors
    ///
    /// Returns the plan's first structural defect; an invalid plan never
    /// becomes an execution.
    pub fn new(plan: FlightPlan, config: ExecutionConfig) -> Result<Self, PlanValidationError> {
        plan.validate()?;
        Ok(Self {
            plan,
            config,
            active_index: 0,
            complete: false,
        })
    }

    /// Feeds one position sample to the sequencer. Capture is
    /// horizontal: within [`ExecutionConfig::capture_radius_m`]
    /// great-circle meters of the active `to` waypoint, ignoring
    /// altitude. Once complete, every further call returns
    /// [`SequenceEvent::None`] without mutating anything.
    pub fn advance(&mut self, position: &GeodeticPosition) -> SequenceEvent {
        let (captured, index) = match self.active_leg() {
            None => return SequenceEvent::None,
            Some(leg) => (
                distance_m(position, &leg.to.position) <= self.config.capture_radius_m,
                leg.index,
            ),
        };
        if !captured {
            return SequenceEvent::None;
        }
        let next = index.wrapping_add(1);
        if next >= self.plan.waypoints.len() {
            self.complete = true;
            SequenceEvent::PlanComplete
        } else {
            self.active_index = next;
            SequenceEvent::LegAdvanced { to_index: next }
        }
    }

    /// The leg being flown, or `None` once the plan is complete.
    #[must_use]
    pub fn active_leg(&self) -> Option<Leg<'_>> {
        if self.complete {
            return None;
        }
        let to = self.plan.waypoints.get(self.active_index)?;
        let from = self
            .active_index
            .checked_sub(1)
            .and_then(|previous| self.plan.waypoints.get(previous));
        Some(Leg {
            from,
            to,
            index: self.active_index,
        })
    }

    /// The plan being executed.
    #[must_use]
    pub const fn plan(&self) -> &FlightPlan {
        &self.plan
    }

    /// Index of the active `to` waypoint. After completion this stays at
    /// the final waypoint's index.
    #[must_use]
    pub const fn active_index(&self) -> usize {
        self.active_index
    }

    /// Waypoints not yet captured, in fly order; empty once complete.
    #[must_use]
    pub fn remaining_waypoints(&self) -> &[Waypoint] {
        if self.complete {
            return &[];
        }
        self.plan.waypoints.get(self.active_index..).unwrap_or(&[])
    }

    /// Whether the final waypoint has captured.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use navigate_contract::PlanRole;

    use super::*;

    fn pos(lat_deg: f64, lon_deg: f64) -> GeodeticPosition {
        GeodeticPosition::new(lat_deg.to_radians(), lon_deg.to_radians(), 0.0)
    }

    fn wp(ident: &str, lat_deg: f64, lon_deg: f64) -> Waypoint {
        Waypoint::new(ident.to_owned(), pos(lat_deg, lon_deg))
    }

    fn three_waypoint_execution() -> PlanExecution {
        let plan = FlightPlan::new(
            "mission".into(),
            PlanRole::Mission,
            vec![wp("W0", 0.0, 0.0), wp("W1", 0.0, 1.0), wp("W2", 1.0, 1.0)],
        );
        PlanExecution::new(plan, ExecutionConfig::default()).expect("valid plan")
    }

    #[test]
    fn sequences_three_waypoints_capturing_exactly_at_the_radius() {
        let mut exec = three_waypoint_execution();

        // Initial direct-to leg: no origin waypoint.
        let leg = exec.active_leg().expect("active leg");
        assert!(leg.from.is_none());
        assert_eq!(leg.index, 0);
        assert_eq!(leg.to.ident, "W0");
        assert_eq!(exec.remaining_waypoints().len(), 3);

        // 0.001° of latitude ≈ 111 m: just outside the 100 m radius.
        assert_eq!(exec.advance(&pos(0.001, 0.0)), SequenceEvent::None);
        assert_eq!(exec.active_index(), 0);
        // 0.0008° ≈ 89 m: just inside, so W0 captures.
        assert_eq!(
            exec.advance(&pos(0.0008, 0.0)),
            SequenceEvent::LegAdvanced { to_index: 1 }
        );
        let leg = exec.active_leg().expect("active leg");
        assert_eq!(leg.from.map(|w| w.ident.as_str()), Some("W0"));
        assert_eq!(leg.to.ident, "W1");

        // Mid-leg, far from W1: no capture.
        assert_eq!(exec.advance(&pos(0.0, 0.5)), SequenceEvent::None);
        assert_eq!(
            exec.advance(&pos(0.0, 1.0)),
            SequenceEvent::LegAdvanced { to_index: 2 }
        );
        assert_eq!(exec.remaining_waypoints().len(), 1);
        assert!(!exec.is_complete());

        assert_eq!(exec.advance(&pos(1.0, 1.0)), SequenceEvent::PlanComplete);
        assert!(exec.is_complete());
        assert!(exec.active_leg().is_none());
        assert!(exec.remaining_waypoints().is_empty());
    }

    #[test]
    fn post_complete_advance_is_inert_even_at_a_waypoint() {
        let mut exec = three_waypoint_execution();
        for position in [pos(0.0, 0.0), pos(0.0, 1.0), pos(1.0, 1.0)] {
            exec.advance(&position);
        }
        assert!(exec.is_complete());
        let snapshot = exec.clone();
        assert_eq!(exec.advance(&pos(1.0, 1.0)), SequenceEvent::None);
        assert_eq!(exec, snapshot);
        assert_eq!(exec.active_index(), 2);
    }

    #[test]
    fn a_position_far_from_everything_advances_nothing() {
        let mut exec = three_waypoint_execution();
        for _ in 0..3 {
            assert_eq!(exec.advance(&pos(45.0, -120.0)), SequenceEvent::None);
        }
        assert_eq!(exec.active_index(), 0);
        assert!(!exec.is_complete());
        assert_eq!(exec.remaining_waypoints().len(), 3);
    }

    #[test]
    fn an_invalid_plan_is_refused_at_construction() {
        let empty = FlightPlan::new("empty".into(), PlanRole::Mission, Vec::new());
        assert!(matches!(
            PlanExecution::new(empty, ExecutionConfig::default()),
            Err(PlanValidationError::Empty { .. })
        ));
    }
}
