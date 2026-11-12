//! Sequencing configuration, portable execution state, active-leg view,
//! and sequencing event types.

use navigate_contract::{LateralReference, LegPath, TurnDirection, TurnType, Waypoint};

use super::hold::{HoldStatus, LegState};
use crate::plan_set::PlanActivationError;
use crate::turn::EARTH_STANDARD_GRAVITY_MPS2;

/// Tunable sequencing parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ExecutionConfig {
    /// Great-circle distance to the active `to` waypoint, in meters, at
    /// which the waypoint captures regardless of turn geometry.
    pub capture_radius_m: f64,
    /// Bank-angle limit for the fly-by and hold turn-performance model,
    /// radians (NAV-TT-002). Defaults to the 18° RNP fly-by standard.
    pub bank_limit_rad: f64,
    /// Floor under waypoint speed constraints, meters per second
    /// (NAV-VC-003).
    pub min_approach_speed_mps: f64,
    /// Surface gravitational acceleration of the body in meters per second
    /// squared, used to compute turn radius `v² / (g · tan φ)`.
    pub surface_gravity_mps2: f64,
}

impl ExecutionConfig {
    /// Builds a config with `capture_radius_m` and the default bank limit,
    /// approach-speed floor, and Earth surface gravity.
    #[must_use]
    pub const fn new(capture_radius_m: f64) -> Self {
        Self {
            capture_radius_m,
            bank_limit_rad: 18.0_f64.to_radians(),
            min_approach_speed_mps: 0.3,
            surface_gravity_mps2: EARTH_STANDARD_GRAVITY_MPS2,
        }
    }

    /// Sets the surface gravitational acceleration in meters per second
    /// squared.
    #[must_use]
    pub const fn with_surface_gravity_mps2(mut self, surface_gravity_mps2: f64) -> Self {
        self.surface_gravity_mps2 = surface_gravity_mps2;
        self
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self::new(100.0)
    }
}

/// Portable `Copy` snapshot of [`super::PlanExecution`] progress.
///
/// Owners that store waypoints alongside execution state save this value
/// between steps and pass it to [`super::PlanExecution::resume_state`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecutionState {
    pub(super) active_index: usize,
    pub(super) complete: bool,
    pub(super) leg_state: LegState,
}

impl ExecutionState {
    /// Index of the active `to` waypoint.
    #[must_use]
    pub const fn active_index(&self) -> usize {
        self.active_index
    }

    /// Returns `true` when the final waypoint has sequenced.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete
    }

    /// Returns the hold progress when the active leg is a hold and the
    /// plan is not complete.
    #[must_use]
    pub const fn hold_status(&self) -> Option<HoldStatus> {
        if self.complete {
            None
        } else {
            self.leg_state.status()
        }
    }
}

/// A borrowed view of the leg being flown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Leg<'a> {
    /// Origin waypoint; `None` on the initial leg.
    pub from: Option<&'a Waypoint>,
    /// Destination waypoint being flown toward.
    pub to: &'a Waypoint,
    /// Index of `to` in the plan's fly order.
    pub index: usize,
    /// Resolved path terminator of this leg (NAV-LG-002).
    pub path: LegPath,
    pub(super) hold_reference: Option<LateralReference>,
}

impl Leg<'_> {
    /// Returns the lateral geometry that this leg gives a guidance
    /// derivation (NAV-LG-014).
    #[must_use]
    pub fn lateral_reference(&self) -> LateralReference {
        if let Some(reference) = self.hold_reference {
            return reference;
        }
        match self.path {
            LegPath::CourseToFix { course_rad, .. }
            | LegPath::CourseToAltitude { course_rad, .. } => LateralReference::course(course_rad),
            LegPath::TrackToFix => self.from.map_or(LateralReference::PresentPosition, |from| {
                LateralReference::track(from.position)
            }),
            _ => LateralReference::PresentPosition,
        }
    }

    /// Returns the mandatory turn direction onto or inside this leg, when
    /// one is declared.
    #[must_use]
    pub const fn turn_direction(&self) -> Option<TurnDirection> {
        if let Some(hold) = self.path.hold() {
            Some(hold.turn)
        } else {
            self.to.turn_direction
        }
    }
}

/// Why a waypoint sequenced (NAV-TT-005).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SequenceReason {
    /// Turn anticipation drove an advance on the abeam line.
    Anticipated,
    /// The fix captured inside the capture radius without anticipation.
    Overflown,
    /// The vehicle crossed the perpendicular through the fix outside the
    /// capture distance (NAV-TT-006).
    Passed {
        /// Signed cross-track distance from the inbound course line in
        /// meters, positive right of course.
        cross_track_m: f64,
    },
    /// A `CA` or `HA` leg achieved its altitude constraint.
    AltitudeReached,
    /// An `HF` leg completed its circuit at the hold fix.
    HoldCircuitComplete,
    /// An `HM` leg completed its exit at the hold fix after
    /// [`super::PlanExecution::exit_hold`].
    HoldExited,
}

/// What one call to [`super::PlanExecution::advance`] decided.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SequenceEvent {
    /// No waypoint sequenced; the active leg is unchanged.
    None,
    /// The active `to` waypoint sequenced and the next leg is active.
    LegAdvanced {
        /// Index of the newly active `to` waypoint.
        to_index: usize,
        /// How the sequenced waypoint was configured to turn.
        turn: TurnType,
        /// Why it sequenced.
        reason: SequenceReason,
    },
    /// The final waypoint sequenced; the plan is complete.
    PlanComplete {
        /// How the terminal waypoint was configured to turn.
        turn: TurnType,
        /// Why it sequenced.
        reason: SequenceReason,
    },
}

pub(super) fn validate_config(config: &ExecutionConfig) -> Result<(), PlanActivationError> {
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
    if !config.surface_gravity_mps2.is_finite() || config.surface_gravity_mps2 <= 0.0 {
        return Err(defect("surface_gravity_mps2", config.surface_gravity_mps2));
    }
    Ok(())
}
