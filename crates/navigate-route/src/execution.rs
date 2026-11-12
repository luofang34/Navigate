//! Active-leg sequencing over a validated flight plan.
//!
//! A [`PlanExecution`] walks a [`FlightPlan`] leg by leg on the reference
//! ellipsoid of the plan. Every leg carries a path terminator (NAV-LG-001).
//! A waypoint that declares none gets one by position in the fly order
//! (NAV-LG-002).
//!
//! A fly-by waypoint between two track-to-fix legs sequences early on the
//! line abeam the distance of turn anticipation before it when the turn
//! stays within the 121° boundary (NAV-TT-003, NAV-LG-010). Fly-over
//! waypoints and terminal fixes capture inside the capture radius
//! (NAV-TT-004). Fixed inbound legs that pass their fix outside the
//! capture radius sequence on crossing the perpendicular through the fix
//! (NAV-TT-006). Hold legs (`HA`, `HF`, `HM`) enter and fly their
//! racetracks until their termination condition is met (NAV-LG-011,
//! NAV-LG-015).

use navigate_contract::{
    FlightPlan, GeodeticPosition, LegPath, PlanValidationError, TurnType, Waypoint,
};
use navigate_geodesy::{BodyEllipsoid, distance_m};

use crate::plan_set::PlanActivationError;
use crate::turn::{turn_anticipation_m, turn_radius_m};

mod activation;
mod corner;
mod hold;
mod passing;
mod types;

#[cfg(test)]
mod tests;

use activation::{resolve_path, screen_leg_paths, skips_minimum_length_guard};
pub use corner::Corner;
use corner::{corner, distance_to_go_m};
use hold::LegState;
pub use hold::{HoldEntry, HoldStage, HoldStatus, hold_entry};
use passing::passed_cross_track_m;
use types::validate_config;
pub use types::{ExecutionConfig, ExecutionState, Leg, SequenceEvent, SequenceReason};

/// Sequencing state for one activated flight plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanExecution<'a> {
    plan: FlightPlan<'a>,
    ellipsoid: BodyEllipsoid,
    config: ExecutionConfig,
    active_index: usize,
    complete: bool,
    leg_state: LegState,
}

impl<'a> PlanExecution<'a> {
    /// Validates the plan and config, then starts execution at index `0`.
    ///
    /// # Errors
    ///
    /// Returns the first validation, config, speed-floor, unsupported-leg,
    /// or minimum-leg-length refusal.
    pub fn new(plan: FlightPlan<'a>, config: ExecutionConfig) -> Result<Self, PlanActivationError> {
        plan.validate()?;
        let ellipsoid = BodyEllipsoid::try_from_contract(plan.ellipsoid)
            .map_err(|_| PlanValidationError::ImplausibleEllipsoid { plan: plan.id })?;
        validate_config(&config)?;
        screen_leg_paths(&plan)?;
        for waypoint in plan.waypoints {
            if let Some(max_speed_mps) = waypoint.max_speed_mps
                && max_speed_mps < config.min_approach_speed_mps
            {
                return Err(PlanActivationError::SpeedBelowFloor {
                    plan: plan.id,
                    ident: waypoint.ident,
                    max_speed_mps,
                    floor_mps: config.min_approach_speed_mps,
                });
            }
        }
        for (index, pair) in plan.waypoints.windows(2).enumerate() {
            let from_path = resolve_path(index, pair[0].path);
            let to_path = resolve_path(index.wrapping_add(1), pair[1].path);
            if skips_minimum_length_guard(to_path)
                || (matches!(from_path, LegPath::CourseToAltitude { .. })
                    && to_path != LegPath::TrackToFix)
            {
                continue;
            }
            let leg_length_m = distance_m(&pair[0].position, &pair[1].position, ellipsoid);
            if leg_length_m <= config.capture_radius_m {
                return Err(PlanActivationError::CaptureRadiusExceedsLeg {
                    plan: plan.id,
                    ident: pair[1].ident,
                    leg_length_m,
                    capture_radius_m: config.capture_radius_m,
                });
            }
        }
        let first_path = resolve_path(0, plan.waypoints.first().and_then(|wp| wp.path));
        Ok(Self {
            plan,
            ellipsoid,
            config,
            active_index: 0,
            complete: false,
            leg_state: LegState::for_path(first_path),
        })
    }

    /// Restores sequencing from `active_index` and `complete` with fresh
    /// per-leg state.
    ///
    /// # Errors
    ///
    /// Returns any refusal of [`Self::new`] or
    /// [`PlanActivationError::InvalidResumeState`].
    pub fn resume(
        plan: FlightPlan<'a>,
        config: ExecutionConfig,
        active_index: usize,
        complete: bool,
    ) -> Result<Self, PlanActivationError> {
        let leg_state = plan
            .waypoints
            .get(active_index)
            .map(|wp| LegState::for_path(resolve_path(active_index, wp.path)))
            .unwrap_or_default();
        Self::resume_state(
            plan,
            config,
            ExecutionState {
                active_index,
                complete,
                leg_state,
            },
        )
    }

    /// Restores full sequencing state, including hold circuit progress.
    ///
    /// # Errors
    ///
    /// Returns any refusal of [`Self::new`] or
    /// [`PlanActivationError::InvalidResumeState`].
    pub fn resume_state(
        plan: FlightPlan<'a>,
        config: ExecutionConfig,
        state: ExecutionState,
    ) -> Result<Self, PlanActivationError> {
        let mut execution = Self::new(plan, config)?;
        let waypoints = plan.waypoints.len();
        let last = waypoints.checked_sub(1);
        let fits =
            state.active_index < waypoints && (!state.complete || Some(state.active_index) == last);
        if !fits {
            return Err(PlanActivationError::InvalidResumeState {
                plan: plan.id,
                active_index: state.active_index,
                complete: state.complete,
                waypoints,
            });
        }
        let path = resolve_path(
            state.active_index,
            plan.waypoints
                .get(state.active_index)
                .and_then(|wp| wp.path),
        );
        execution.active_index = state.active_index;
        execution.complete = state.complete;
        execution.leg_state = if path.hold().is_some() == state.leg_state.status().is_some() {
            state.leg_state
        } else {
            LegState::for_path(path)
        };
        Ok(execution)
    }

    /// Exports the portable [`ExecutionState`] snapshot.
    #[must_use]
    pub const fn state(&self) -> ExecutionState {
        ExecutionState {
            active_index: self.active_index,
            complete: self.complete,
            leg_state: self.leg_state,
        }
    }

    /// Returns observable progress when the active leg is a hold.
    #[must_use]
    pub const fn hold_status(&self) -> Option<HoldStatus> {
        self.state().hold_status()
    }

    /// Arms manual exit on an active `HM` (`HoldManual`) leg. Returns
    /// `true` when the active leg is `HoldManual`.
    pub fn exit_hold(&mut self) -> bool {
        if self.complete {
            return false;
        }
        let Some(to) = self.plan.waypoints.get(self.active_index) else {
            return false;
        };
        self.leg_state
            .arm_exit(resolve_path(self.active_index, to.path))
    }

    /// Feeds one position sample to the sequencer at `groundspeed_mps`.
    pub fn advance(&mut self, position: &GeodeticPosition, groundspeed_mps: f64) -> SequenceEvent {
        let Some(leg) = self.active_leg() else {
            return SequenceEvent::None;
        };
        let index = leg.index;
        let turn = leg.to.turn;
        let reason = if let Some(hold) = leg.path.hold() {
            let Some(reason) = self.leg_state.advance_hold(
                &self.plan,
                index,
                leg.to,
                leg.path,
                hold,
                position,
                groundspeed_mps,
                &self.config,
                self.ellipsoid,
            ) else {
                return SequenceEvent::None;
            };
            reason
        } else if matches!(leg.path, LegPath::CourseToAltitude { .. }) {
            let Some(reason) = self.leg_state.advance_altitude(position, leg.to.altitude) else {
                return SequenceEvent::None;
            };
            reason
        } else {
            let Some(reason) = self.advance_horizontal(&leg, position, groundspeed_mps) else {
                return SequenceEvent::None;
            };
            reason
        };
        self.transition_after(index, turn, reason)
    }

    fn advance_horizontal(
        &self,
        leg: &Leg<'_>,
        position: &GeodeticPosition,
        groundspeed_mps: f64,
    ) -> Option<SequenceReason> {
        let distance = distance_m(position, &leg.to.position, self.ellipsoid);
        if !distance.is_finite() {
            return None;
        }
        let anticipation = self.anticipation_distance(leg, groundspeed_mps);
        let anticipated = anticipation > self.config.capture_radius_m;
        if distance < self.config.capture_radius_m {
            Some(if anticipated {
                SequenceReason::Anticipated
            } else {
                SequenceReason::Overflown
            })
        } else if anticipated && self.inside_turn_start(leg, position, anticipation) {
            Some(SequenceReason::Anticipated)
        } else {
            passed_cross_track_m(leg, position, self.ellipsoid)
                .map(|cross_track_m| SequenceReason::Passed { cross_track_m })
        }
    }

    fn transition_after(
        &mut self,
        index: usize,
        turn: TurnType,
        reason: SequenceReason,
    ) -> SequenceEvent {
        let next = index.wrapping_add(1);
        if next >= self.plan.waypoints.len() {
            self.complete = true;
            SequenceEvent::PlanComplete { turn, reason }
        } else {
            let next_path =
                resolve_path(next, self.plan.waypoints.get(next).and_then(|wp| wp.path));
            self.active_index = next;
            self.leg_state = LegState::for_path(next_path);
            SequenceEvent::LegAdvanced {
                to_index: next,
                turn,
                reason,
            }
        }
    }

    fn inside_turn_start(
        &self,
        leg: &Leg<'_>,
        position: &GeodeticPosition,
        anticipation: f64,
    ) -> bool {
        let Some(from) = leg.from else {
            return false;
        };
        distance_to_go_m(position, &from.position, &leg.to.position, self.ellipsoid)
            .is_some_and(|to_go| (0.0..anticipation).contains(&to_go))
    }

    fn anticipation_distance(&self, leg: &Leg<'_>, groundspeed_mps: f64) -> f64 {
        let Corner::Anticipated { track_change_rad } = self.corner(leg.index) else {
            return 0.0;
        };
        let (Some(from), Some(next)) =
            (leg.from, self.plan.waypoints.get(leg.index.wrapping_add(1)))
        else {
            return 0.0;
        };
        let inbound_length = distance_m(&from.position, &leg.to.position, self.ellipsoid);
        let outbound_length = distance_m(&leg.to.position, &next.position, self.ellipsoid);
        let radius = turn_radius_m(
            groundspeed_mps,
            self.config.bank_limit_rad,
            self.config.surface_gravity_mps2,
        );
        turn_anticipation_m(radius, track_change_rad).min(0.5 * inbound_length.min(outbound_length))
    }

    /// Returns the [`Corner`] classification at `index` (NAV-TT-003).
    #[must_use]
    pub fn corner(&self, index: usize) -> Corner {
        corner(&self.plan, index, self.ellipsoid)
    }

    /// Iterates the [`Corner`] classifications for all waypoints in fly
    /// order.
    pub fn corners(&self) -> impl Iterator<Item = Corner> + '_ {
        (0..self.plan.waypoints.len()).map(|index| self.corner(index))
    }

    /// Returns the active leg, or `None` once the plan is complete.
    #[must_use]
    pub fn active_leg(&self) -> Option<Leg<'a>> {
        if self.complete {
            return None;
        }
        let to = self.plan.waypoints.get(self.active_index)?;
        let from = self
            .active_index
            .checked_sub(1)
            .and_then(|previous| self.plan.waypoints.get(previous));
        let path = resolve_path(self.active_index, to.path);
        let hold_reference = path.hold().map(|hold| {
            self.leg_state
                .lateral_reference(to, hold, &self.config, self.ellipsoid)
        });
        Some(Leg {
            from,
            to,
            index: self.active_index,
            path,
            hold_reference,
        })
    }

    /// Returns the flight plan being executed.
    #[must_use]
    pub const fn plan(&self) -> &FlightPlan<'a> {
        &self.plan
    }

    /// Returns the derived ellipsoid of the plan.
    #[must_use]
    pub const fn ellipsoid(&self) -> BodyEllipsoid {
        self.ellipsoid
    }

    /// Returns the index of the active `to` waypoint.
    #[must_use]
    pub const fn active_index(&self) -> usize {
        self.active_index
    }

    /// Returns the remaining unsequenced waypoints, or `&[]` when complete.
    #[must_use]
    pub fn remaining_waypoints(&self) -> &'a [Waypoint] {
        if self.complete {
            return &[];
        }
        self.plan.waypoints.get(self.active_index..).unwrap_or(&[])
    }

    /// Returns `true` when the final waypoint has sequenced.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete
    }
}
