//! Active-leg sequencing over a validated flight plan.
//!
//! A [`PlanExecution`] walks a [`FlightPlan`] leg by leg. The active leg
//! runs from the previously captured waypoint — or from the present
//! position on the initial direct-to leg — to the next waypoint in fly
//! order. A fly-by waypoint sequences early by the distance of turn
//! anticipation when the turn geometry earns it (NAV-TT-003); a fly-over
//! waypoint and the terminal waypoint sequence only inside the capture
//! radius (NAV-TT-004). After the final waypoint captures, the execution
//! is complete and further advances are inert.

use navigate_contract::{FlightPlan, GeodeticPosition, TurnType, Waypoint};
use navigate_geodesy::{distance_m, initial_bearing_rad};

use crate::plan_set::PlanActivationError;
use crate::turn::{turn_anticipation_m, turn_radius_m};

#[cfg(test)]
mod tests;

/// Below this track separation a course is numerically meaningless and
/// turn geometry is skipped (mirrors the geodesy degenerate-track floor).
const MIN_TRACK_SEPARATION_M: f64 = 1e-3;

/// Largest track change fly-by anticipation applies to: 120°. Beyond it
/// `tan(Δ/2)` grows toward a reversal's blowup, and a course reversal
/// has no fly-by solution — holds and radius-to-fix legs own that
/// geometry and are out of scope (NAV-TT-003).
const MAX_ANTICIPATED_TRACK_CHANGE_RAD: f64 = 2.0 * core::f64::consts::FRAC_PI_3;

/// Folds a course difference into `[0, π]` — the magnitude of a track
/// change.
fn fold_track_change(angle_rad: f64) -> f64 {
    let wrapped = angle_rad.rem_euclid(core::f64::consts::TAU);
    if wrapped > core::f64::consts::PI {
        core::f64::consts::TAU - wrapped
    } else {
        wrapped
    }
}

/// Tunable sequencing parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ExecutionConfig {
    /// Great-circle distance to the active `to` waypoint, in meters, at
    /// which the waypoint captures regardless of turn geometry.
    pub capture_radius_m: f64,
    /// Bank-angle limit for the fly-by turn-performance model, radians
    /// (NAV-TT-002). The default is the 18° RNP fly-by standard recorded
    /// in `docs/procedure-requirements.md`.
    pub bank_limit_rad: f64,
    /// Floor under waypoint speed constraints, meters per second
    /// (NAV-VC-003): a plan demanding a slower approach is refused at
    /// activation, never silently clamped in flight.
    pub min_approach_speed_mps: f64,
}

impl ExecutionConfig {
    /// Builds a config with the given capture radius in meters and the
    /// default bank limit and approach-speed floor.
    #[must_use]
    pub fn new(capture_radius_m: f64) -> Self {
        Self {
            capture_radius_m,
            bank_limit_rad: 18.0_f64.to_radians(),
            min_approach_speed_mps: 0.3,
        }
    }
}

/// 100 m capture radius, the 18° fly-by bank standard, and a 0.3 m/s
/// approach-speed floor.
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

/// Why a waypoint sequenced (NAV-TT-005).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SequenceReason {
    /// The distance of turn anticipation drove an advance beyond the
    /// capture radius — only fly-by geometry earns this.
    Anticipated,
    /// The fix captured inside the capture radius: every fly-over
    /// crossing, and any fly-by fix whose anticipation distance stays
    /// inside the capture radius.
    Overflown,
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
        /// How the captured waypoint was flown.
        turn: TurnType,
        /// Why it sequenced.
        reason: SequenceReason,
    },
    /// The final waypoint captured; the plan is complete.
    PlanComplete {
        /// How the terminal waypoint was flown.
        turn: TurnType,
        /// Why it sequenced (always capture-driven: the terminal fix has
        /// no onward course to anticipate).
        reason: SequenceReason,
    },
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
    /// Validates the plan and the config, then starts execution on the
    /// initial direct-to leg toward the first waypoint.
    ///
    /// # Errors
    ///
    /// The plan's first structural defect, an unflyable config value, or
    /// a waypoint speed constraint below the approach-speed floor
    /// (NAV-VC-003) — an invalid plan never becomes an execution.
    pub fn new(plan: FlightPlan, config: ExecutionConfig) -> Result<Self, PlanActivationError> {
        plan.validate()?;
        validate_config(&config)?;
        for waypoint in &plan.waypoints {
            if let Some(max_speed_mps) = waypoint.max_speed_mps
                && max_speed_mps < config.min_approach_speed_mps
            {
                return Err(PlanActivationError::SpeedBelowFloor {
                    plan: plan.id.clone(),
                    ident: waypoint.ident.clone(),
                    max_speed_mps,
                    floor_mps: config.min_approach_speed_mps,
                });
            }
        }
        Ok(Self {
            plan,
            config,
            active_index: 0,
            complete: false,
        })
    }

    /// Feeds one position sample to the sequencer at the commanded
    /// groundspeed. Capture is horizontal and ignores altitude: a fly-by
    /// waypoint between two fixed legs sequences strictly within
    /// `max(capture_radius, min(DTA, half the shorter adjoining leg))`
    /// of the fix (NAV-TT-003); fly-over waypoints, the terminal
    /// waypoint, direct-to legs, and track changes beyond the
    /// anticipation limit sequence within the capture radius only
    /// (NAV-TT-004). A non-finite or negative groundspeed anticipates
    /// nothing; a non-finite position captures nothing. Once complete,
    /// every further call returns [`SequenceEvent::None`] without
    /// mutating anything.
    pub fn advance(&mut self, position: &GeodeticPosition, groundspeed_mps: f64) -> SequenceEvent {
        let Some(leg) = self.active_leg() else {
            return SequenceEvent::None;
        };
        let index = leg.index;
        let turn = leg.to.turn;
        let distance = distance_m(position, &leg.to.position);
        if !distance.is_finite() {
            return SequenceEvent::None;
        }
        let anticipation = self.anticipation_distance(&leg, groundspeed_mps);
        let threshold = self.config.capture_radius_m.max(anticipation);
        // Strict: a capped anticipation can equal a leg-start distance,
        // and a leg must never sequence at the instant it begins.
        if distance >= threshold {
            return SequenceEvent::None;
        }
        // The reason names the RULE that authorizes early sequencing,
        // not the sample that happened to arrive (NAV-TT-005): a fly-by
        // fix whose anticipation exceeds the capture radius is an
        // anticipated transition even when a sparse sample lands inside
        // the radius.
        let reason = if anticipation > self.config.capture_radius_m {
            SequenceReason::Anticipated
        } else {
            SequenceReason::Overflown
        };
        let next = index.wrapping_add(1);
        if next >= self.plan.waypoints.len() {
            self.complete = true;
            SequenceEvent::PlanComplete { turn, reason }
        } else {
            self.active_index = next;
            SequenceEvent::LegAdvanced {
                to_index: next,
                turn,
                reason,
            }
        }
    }

    /// The fly-by anticipation distance for the active leg: the DTA
    /// bounded by half the shorter adjoining leg, so each leg keeps a
    /// flyable middle and both of its ends may anticipate (NAV-TT-003).
    /// Zero — capture-radius sequencing — for fly-over fixes, the
    /// terminal fix, direct-to legs (their inbound geometry is the live
    /// position, not a fixed track), track changes beyond
    /// [`MAX_ANTICIPATED_TRACK_CHANGE_RAD`] (a reversal has no fly-by
    /// solution), and geometry too degenerate to define a course.
    fn anticipation_distance(&self, leg: &Leg<'_>, groundspeed_mps: f64) -> f64 {
        if leg.to.turn != TurnType::FlyBy {
            return 0.0;
        }
        let Some(from) = leg.from else {
            return 0.0;
        };
        let Some(next) = self.plan.waypoints.get(leg.index.wrapping_add(1)) else {
            return 0.0;
        };
        let inbound_length = distance_m(&from.position, &leg.to.position);
        let outbound_length = distance_m(&leg.to.position, &next.position);
        if inbound_length < MIN_TRACK_SEPARATION_M || outbound_length < MIN_TRACK_SEPARATION_M {
            return 0.0;
        }
        // The inbound course AT the fix is the reciprocal of the course
        // back to the leg origin; the initial bearing from the origin
        // would mis-size the corner by the great-circle convergence.
        let inbound = initial_bearing_rad(&leg.to.position, &from.position) + core::f64::consts::PI;
        let outbound = initial_bearing_rad(&leg.to.position, &next.position);
        let track_change = fold_track_change(outbound - inbound);
        if track_change > MAX_ANTICIPATED_TRACK_CHANGE_RAD {
            return 0.0;
        }
        let radius = turn_radius_m(groundspeed_mps, self.config.bank_limit_rad);
        turn_anticipation_m(radius, track_change).min(0.5 * inbound_length.min(outbound_length))
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

/// Screens the config's own numbers: a non-finite or non-positive value
/// in any of them would silently disable the comparison it feeds.
fn validate_config(config: &ExecutionConfig) -> Result<(), PlanActivationError> {
    let defect =
        |field: &'static str, value: f64| PlanActivationError::InvalidConfig { field, value };
    if !config.capture_radius_m.is_finite() || config.capture_radius_m <= 0.0 {
        return Err(defect("capture_radius_m", config.capture_radius_m));
    }
    if !config.bank_limit_rad.is_finite()
        || config.bank_limit_rad <= 0.0
        || config.bank_limit_rad >= core::f64::consts::FRAC_PI_2
    {
        return Err(defect("bank_limit_rad", config.bank_limit_rad));
    }
    if !config.min_approach_speed_mps.is_finite() || config.min_approach_speed_mps <= 0.0 {
        return Err(defect(
            "min_approach_speed_mps",
            config.min_approach_speed_mps,
        ));
    }
    Ok(())
}
