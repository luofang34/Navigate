//! Racetrack entry, circuit state, and altitude-leg termination for
//! `CA`, `HA`, `HF`, and `HM` legs.

use core::f64::consts::{FRAC_PI_2, FRAC_PI_6, PI, TAU};

use navigate_contract::{
    AltitudeConstraint, FlightPlan, GeodeticPosition, HoldGeometry, HoldLegLength,
    LateralReference, LegPath, TurnDirection, Waypoint,
};
#[cfg(test)]
use navigate_geodesy::cross_track_from_course_m;
use navigate_geodesy::{
    BodyEllipsoid, along_track_from_course_m, destination_point, distance_m, initial_bearing_rad,
};

use super::activation::resolve_path;
use super::{ExecutionConfig, SequenceReason};
use crate::float_math::rem_euclid;
use crate::turn::turn_radius_m;

/// Cosine of the 30° teardrop offset (`cos(π / 6)`).
const COS_30_DEG: f64 = 0.866_025_403_784_438_6;

/// Minimum groundspeed used to size a timed hold leg in still or slow
/// conditions, meters per second.
const MIN_HOLD_SPEED_MPS: f64 = 1.0;

/// Minimum fix separation in meters to derive an arrival bearing from a
/// preceding waypoint.
const MIN_ARRIVAL_SEPARATION_M: f64 = 1.0;

/// Standard ICAO / FAA AIM 5-3-8 entry into a holding pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HoldEntry {
    /// Turn at the fix in the hold direction onto the outbound leg.
    Direct,
    /// Fly the outbound course on the non-holding side, then turn back
    /// toward the holding side to intercept the inbound course.
    Parallel,
    /// Fly 30° into the holding side from the outbound course, then turn
    /// in the hold direction onto the inbound course.
    Teardrop,
}

/// Active segment of a holding pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HoldStage {
    /// Approaching the hold fix before pattern entry.
    ToFix,
    /// Flying the outbound segment of a parallel or teardrop entry.
    EntryOutbound {
        /// Selected entry sector.
        entry: HoldEntry,
    },
    /// Flying the displaced outbound leg on the holding side.
    Outbound,
    /// Flying the inbound course back to the hold fix.
    Inbound,
}

/// Observable progress of an active hold leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HoldStatus {
    /// Entry sector selected at fix crossing, or `None` while still in
    /// [`HoldStage::ToFix`].
    pub entry: Option<HoldEntry>,
    /// Current segment of the pattern.
    pub stage: HoldStage,
    /// Number of full circuits completed at the hold fix.
    pub circuits_completed: u32,
    /// Whether manual exit has been armed via
    /// [`super::PlanExecution::exit_hold`].
    pub exit_armed: bool,
}

/// Selects the standard holding entry for a vehicle reaching the hold fix
/// on `arrival_course_rad`.
///
/// For right turns, courses in `(0°, 110°]` clockwise of the outbound
/// course select [`HoldEntry::Parallel`], courses in `(110°, 290°)` select
/// [`HoldEntry::Direct`], and the remaining `70°` sector selects
/// [`HoldEntry::Teardrop`]. Left turns mirror the sectors.
#[must_use]
pub fn hold_entry(
    inbound_course_rad: f64,
    turn: TurnDirection,
    arrival_course_rad: f64,
) -> HoldEntry {
    let relative = turn.sign() * (arrival_course_rad - inbound_course_rad);
    let from_outbound = rem_euclid(relative.to_degrees() - 180.0, 360.0);
    if from_outbound > 0.0 && from_outbound <= 110.0 {
        HoldEntry::Parallel
    } else if from_outbound > 110.0 && from_outbound < 290.0 {
        HoldEntry::Direct
    } else {
        HoldEntry::Teardrop
    }
}

/// Internal per-leg state carried inside [`super::ExecutionState`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(super) struct LegState {
    entry: Option<HoldEntry>,
    stage: Option<HoldStage>,
    circuits_completed: u32,
    exit_armed: bool,
    outbound_radius_m: f64,
    entry_altitude_m: Option<f64>,
    last_position: Option<GeodeticPosition>,
}

impl LegState {
    pub(super) const fn status(&self) -> Option<HoldStatus> {
        let Some(stage) = self.stage else {
            return None;
        };
        Some(HoldStatus {
            entry: self.entry,
            stage,
            circuits_completed: self.circuits_completed,
            exit_armed: self.exit_armed,
        })
    }

    pub(super) const fn for_path(path: LegPath) -> Self {
        let stage = if path.hold().is_some() {
            Some(HoldStage::ToFix)
        } else {
            None
        };
        Self {
            entry: None,
            stage,
            circuits_completed: 0,
            exit_armed: false,
            outbound_radius_m: 0.0,
            entry_altitude_m: None,
            last_position: None,
        }
    }

    pub(super) fn arm_exit(&mut self, path: LegPath) -> bool {
        if matches!(path, LegPath::HoldManual { .. }) {
            self.exit_armed = true;
            true
        } else {
            false
        }
    }

    pub(super) fn advance_altitude(
        &mut self,
        position: &GeodeticPosition,
        constraint: Option<AltitudeConstraint>,
    ) -> Option<SequenceReason> {
        if !position.altitude_m.is_finite() {
            return None;
        }
        let initial = *self.entry_altitude_m.get_or_insert(position.altitude_m);
        altitude_achieved(position.altitude_m, initial, constraint)
            .then_some(SequenceReason::AltitudeReached)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "hold advancement evaluates plan context, config, ellipsoid, and live state together"
    )]
    pub(super) fn advance_hold(
        &mut self,
        plan: &FlightPlan<'_>,
        index: usize,
        to: &Waypoint,
        path: LegPath,
        hold: HoldGeometry,
        position: &GeodeticPosition,
        groundspeed_mps: f64,
        config: &ExecutionConfig,
        ellipsoid: BodyEllipsoid,
    ) -> Option<SequenceReason> {
        if !position.altitude_m.is_finite() {
            return None;
        }
        let dist_m = distance_m(position, &to.position, ellipsoid);
        if !dist_m.is_finite() {
            return None;
        }
        let initial_alt = *self.entry_altitude_m.get_or_insert(position.altitude_m);
        let speed = if groundspeed_mps.is_finite() && groundspeed_mps > MIN_HOLD_SPEED_MPS {
            groundspeed_mps
        } else {
            MIN_HOLD_SPEED_MPS
        };
        let radius_m = turn_radius_m(speed, config.bank_limit_rad, config.surface_gravity_mps2)
            .max(config.capture_radius_m);
        let leg_m = match hold.leg {
            HoldLegLength::Distance { meters } => meters,
            HoldLegLength::Time { seconds } => (seconds * speed).max(2.0 * config.capture_radius_m),
        };
        let x_m =
            along_track_from_course_m(position, &to.position, hold.inbound_course_rad, ellipsoid);
        let stage = *self.stage.get_or_insert(HoldStage::ToFix);
        match stage {
            HoldStage::ToFix => {
                if dist_m >= config.capture_radius_m {
                    self.last_position = Some(*position);
                    return None;
                }
                if matches!(path, LegPath::HoldToAltitude { .. })
                    && altitude_achieved(position.altitude_m, initial_alt, to.altitude)
                {
                    return Some(SequenceReason::AltitudeReached);
                }
                if matches!(path, LegPath::HoldManual { .. }) && self.exit_armed {
                    return Some(SequenceReason::HoldExited);
                }
                let arrival = self
                    .last_position
                    .filter(|prev| distance_m(prev, position, ellipsoid) > 0.5)
                    .map(|prev| initial_bearing_rad(&prev, position))
                    .unwrap_or_else(|| {
                        plan_arrival_course(
                            plan,
                            index,
                            &to.position,
                            hold.inbound_course_rad,
                            ellipsoid,
                        )
                    });
                let entry = hold_entry(hold.inbound_course_rad, hold.turn, arrival);
                self.entry = Some(entry);
                self.outbound_radius_m = radius_m;
                self.stage = Some(match entry {
                    HoldEntry::Direct => HoldStage::Outbound,
                    HoldEntry::Parallel | HoldEntry::Teardrop => HoldStage::EntryOutbound { entry },
                });
                None
            }
            HoldStage::EntryOutbound { entry } => {
                if matches!(path, LegPath::HoldToAltitude { .. })
                    && altitude_achieved(position.altitude_m, initial_alt, to.altitude)
                {
                    return Some(SequenceReason::AltitudeReached);
                }
                let turn_station_m = match entry {
                    HoldEntry::Teardrop => -leg_m * COS_30_DEG,
                    HoldEntry::Parallel | HoldEntry::Direct => -leg_m,
                };
                if x_m <= turn_station_m {
                    self.stage = Some(HoldStage::Inbound);
                }
                None
            }
            HoldStage::Outbound => {
                self.outbound_radius_m = radius_m;
                if matches!(path, LegPath::HoldToAltitude { .. })
                    && altitude_achieved(position.altitude_m, initial_alt, to.altitude)
                {
                    return Some(SequenceReason::AltitudeReached);
                }
                if x_m <= -leg_m {
                    self.stage = Some(HoldStage::Inbound);
                }
                None
            }
            HoldStage::Inbound => {
                if matches!(path, LegPath::HoldToAltitude { .. })
                    && altitude_achieved(position.altitude_m, initial_alt, to.altitude)
                {
                    return Some(SequenceReason::AltitudeReached);
                }
                if dist_m < config.capture_radius_m || x_m >= 0.0 {
                    self.circuits_completed = self.circuits_completed.wrapping_add(1);
                    if matches!(path, LegPath::HoldToFix { .. }) {
                        return Some(SequenceReason::HoldCircuitComplete);
                    }
                    if matches!(path, LegPath::HoldManual { .. }) && self.exit_armed {
                        return Some(SequenceReason::HoldExited);
                    }
                    self.outbound_radius_m = radius_m;
                    self.stage = Some(HoldStage::Outbound);
                }
                None
            }
        }
    }

    pub(super) fn lateral_reference(
        &self,
        to: &Waypoint,
        hold: HoldGeometry,
        config: &ExecutionConfig,
        ellipsoid: BodyEllipsoid,
    ) -> LateralReference {
        let outbound_rad = wrap_tau(hold.inbound_course_rad + PI);
        match self.stage.unwrap_or(HoldStage::ToFix) {
            HoldStage::ToFix => LateralReference::PresentPosition,
            HoldStage::EntryOutbound {
                entry: HoldEntry::Parallel,
            } => LateralReference::course(outbound_rad),
            HoldStage::EntryOutbound {
                entry: HoldEntry::Teardrop,
            } => {
                let teardrop_rad = wrap_tau(outbound_rad - hold.turn.sign() * FRAC_PI_6);
                LateralReference::course(teardrop_rad)
            }
            HoldStage::EntryOutbound {
                entry: HoldEntry::Direct,
            }
            | HoldStage::Outbound => {
                let radius_m = if self.outbound_radius_m > 0.0 {
                    self.outbound_radius_m
                } else {
                    config.capture_radius_m
                };
                let abeam_rad = wrap_tau(hold.inbound_course_rad + hold.turn.sign() * FRAC_PI_2);
                let anchor = destination_point(&to.position, abeam_rad, 2.0 * radius_m, ellipsoid);
                LateralReference::displaced_course(anchor, outbound_rad)
            }
            HoldStage::Inbound => LateralReference::course(hold.inbound_course_rad),
        }
    }
}

/// Signed cross-track distance in meters from the hold's inbound course,
/// positive on the holding side.
#[cfg(test)]
pub(super) fn holding_side_offset_m(
    position: &GeodeticPosition,
    fix: &GeodeticPosition,
    hold: HoldGeometry,
    ellipsoid: BodyEllipsoid,
) -> f64 {
    hold.turn.sign() * cross_track_from_course_m(position, fix, hold.inbound_course_rad, ellipsoid)
}

fn altitude_achieved(
    altitude_m: f64,
    initial_altitude_m: f64,
    constraint: Option<AltitudeConstraint>,
) -> bool {
    match constraint {
        None => false,
        Some(AltitudeConstraint::At(target_m)) => {
            if initial_altitude_m <= target_m {
                altitude_m >= target_m
            } else {
                altitude_m <= target_m
            }
        }
        Some(AltitudeConstraint::AtOrAbove(target_m)) => altitude_m >= target_m,
        Some(AltitudeConstraint::AtOrBelow(target_m)) => altitude_m <= target_m,
        Some(AltitudeConstraint::Window { lower_m, upper_m }) => {
            altitude_m >= lower_m && altitude_m <= upper_m
        }
        Some(_) => false,
    }
}

fn plan_arrival_course(
    plan: &FlightPlan<'_>,
    index: usize,
    fix: &GeodeticPosition,
    fallback_rad: f64,
    ellipsoid: BodyEllipsoid,
) -> f64 {
    let mut cursor = index;
    while let Some(prev_idx) = cursor.checked_sub(1) {
        let Some(prev) = plan.waypoints.get(prev_idx) else {
            break;
        };
        let prev_path = resolve_path(prev_idx, prev.path);
        if let LegPath::CourseToFix { course_rad, .. }
        | LegPath::CourseToAltitude { course_rad, .. } = prev_path
        {
            return course_rad;
        }
        if distance_m(&prev.position, fix, ellipsoid) >= MIN_ARRIVAL_SEPARATION_M {
            return wrap_tau(initial_bearing_rad(fix, &prev.position) + PI);
        }
        cursor = prev_idx;
    }
    fallback_rad
}

fn wrap_tau(angle_rad: f64) -> f64 {
    let wrapped = rem_euclid(angle_rad, TAU);
    if wrapped >= TAU { 0.0 } else { wrapped }
}
