//! Drag-versus-energy tracking of an atmospheric entry corridor.
//!
//! The specific mechanical energy `E = v²/2 − μ/r` decreases at the rate
//! `Ė = −D·v` during an entry, where `D` is the drag acceleration. An
//! [`EntryCorridor`] gives the reference drag `D_ref(E)` and its rate
//! `Ḋ_ref(E)` as linear segments between caller-owned [`EntryNode`]
//! knots. The law commands the vertical lift-to-drag ratio
//!
//! `(L/D)_v = (g − v²/r)/D_ref + f₁·(D − D_ref) + f₂·(Ḋ − Ḋ_ref)`,
//!
//! where the first term is the equilibrium glide. With the density scale
//! height `H`, the gains `f₁ = ω²·H/D_ref²` and `f₂ = 2ζω·H/D_ref²` give
//! the drag error the natural frequency `ω` and the damping ratio `ζ`.
//! The bank magnitude is `acos((L/D)_v / (L/D))`. The bank sign changes
//! when the heading error to the landing site goes out of the deadband
//! on the side of the current bank. [`EntryState`] holds the sign inline.

use nalgebra::Vector3;
use navigate_astrodynamics::PlanetaryBody;
use navigate_contract::FrameId;

use crate::error::TrajectoryError;
use crate::float_math::{acos, atan2, rem_euclid};
use crate::state::{StateVector, unit};

mod heading;
mod law;

#[cfg(test)]
mod tests;

pub use law::EntryLaw;

use heading::Heading;

/// One knot of the reference drag profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntryNode {
    /// The specific mechanical energy in J/kg.
    pub energy_j_per_kg: f64,
    /// The reference drag in m/s². It is more than zero.
    pub reference_drag_mps2: f64,
    /// The reference rate of change of the drag in m/s³.
    pub reference_drag_rate_mps3: f64,
}

impl EntryNode {
    /// Builds a knot from its parts.
    #[must_use]
    pub const fn new(
        energy_j_per_kg: f64,
        reference_drag_mps2: f64,
        reference_drag_rate_mps3: f64,
    ) -> Self {
        Self {
            energy_j_per_kg,
            reference_drag_mps2,
            reference_drag_rate_mps3,
        }
    }
}

/// The reference drag profile of one entry, on one body, toward one
/// landing site.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EntryCorridor<'a> {
    /// The body. Its body-fixed frame is the frame of the state and of
    /// the landing site, so that the velocity is relative to the
    /// atmosphere.
    pub body: PlanetaryBody,
    /// The knots, in strictly decreasing energy.
    pub nodes: &'a [EntryNode],
    /// The landing site in meters in the body-fixed frame.
    pub landing_site_m: Vector3<f64>,
    /// At this energy in J/kg or less, the entry is complete and the
    /// terminal area energy management (TAEM) starts.
    pub taem_energy_j_per_kg: f64,
    /// The vehicle and the gains of the law.
    pub law: EntryLaw,
}

impl<'a> EntryCorridor<'a> {
    /// Builds a corridor from its parts. [`Self::validate`] checks the
    /// values.
    #[must_use]
    pub const fn new(
        body: PlanetaryBody,
        nodes: &'a [EntryNode],
        landing_site_m: Vector3<f64>,
        taem_energy_j_per_kg: f64,
        law: EntryLaw,
    ) -> Self {
        Self {
            body,
            nodes,
            landing_site_m,
            taem_energy_j_per_kg,
            law,
        }
    }

    /// The body-fixed frame of the corridor.
    #[must_use]
    pub const fn frame(&self) -> FrameId {
        self.body.ellipsoid.frame()
    }

    /// Checks the body, the knots, the landing site, the TAEM energy,
    /// and the law.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::InvalidCorridor`] for knots that are too few,
    /// not finite, or not in strictly decreasing energy, and
    /// [`TrajectoryError::InvalidParameter`] for another value.
    pub fn validate(&self) -> Result<(), TrajectoryError> {
        if !self.body.is_plausible() {
            return Err(TrajectoryError::InvalidParameter { name: "body" });
        }
        let knot_valid = |node: &EntryNode| {
            node.energy_j_per_kg.is_finite()
                && node.reference_drag_mps2.is_finite()
                && node.reference_drag_mps2 > 0.0
                && node.reference_drag_rate_mps3.is_finite()
        };
        let ordered = self
            .nodes
            .windows(2)
            .all(|pair| matches!(pair, [a, b] if a.energy_j_per_kg > b.energy_j_per_kg));
        if self.nodes.len() < 2 || !ordered || !self.nodes.iter().all(knot_valid) {
            return Err(TrajectoryError::InvalidCorridor);
        }
        if !self.landing_site_m.iter().all(|c| c.is_finite()) {
            return Err(TrajectoryError::InvalidParameter {
                name: "landing_site_m",
            });
        }
        if !self.taem_energy_j_per_kg.is_finite() {
            return Err(TrajectoryError::InvalidParameter {
                name: "taem_energy_j_per_kg",
            });
        }
        self.law.validate()
    }

    /// The reference drag in m/s² and its rate in m/s³ at the energy
    /// `energy_j_per_kg`. Outside the knots, the value of the nearest
    /// end knot applies.
    #[must_use]
    pub fn reference_at(&self, energy_j_per_kg: f64) -> (f64, f64) {
        let at = |node: &EntryNode| (node.reference_drag_mps2, node.reference_drag_rate_mps3);
        if let Some(first) = self.nodes.first()
            && energy_j_per_kg >= first.energy_j_per_kg
        {
            return at(first);
        }
        for pair in self.nodes.windows(2) {
            if let [high, low] = pair
                && energy_j_per_kg >= low.energy_j_per_kg
            {
                let span = high.energy_j_per_kg - low.energy_j_per_kg;
                let w = (energy_j_per_kg - low.energy_j_per_kg) / span;
                let lerp = |a: f64, b: f64| b + (a - b) * w;
                return (
                    lerp(high.reference_drag_mps2, low.reference_drag_mps2),
                    lerp(high.reference_drag_rate_mps3, low.reference_drag_rate_mps3),
                );
            }
        }
        self.nodes.last().map_or((0.0, 0.0), at)
    }
}

/// The inline state of the entry law: the sign of the bank and the
/// number of bank reversals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntryState {
    /// `1.0` for a bank to the right, `-1.0` for a bank to the left.
    pub bank_sign: f64,
    /// The number of bank reversals. It wraps at `u32::MAX`.
    pub reversals: u32,
}

impl EntryState {
    /// A state that banks to the right and has no reversals.
    pub const INITIAL: Self = Self {
        bank_sign: 1.0,
        reversals: 0,
    };

    /// This state after one bank reversal.
    #[must_use]
    pub const fn reversed(self) -> Self {
        Self {
            bank_sign: if self.bank_sign > 0.0 { -1.0 } else { 1.0 },
            reversals: self.reversals.wrapping_add(1),
        }
    }
}

impl Default for EntryState {
    fn default() -> Self {
        Self::INITIAL
    }
}

/// The output of one evaluation of [`evaluate_entry`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EntryCommand {
    /// The commanded bank angle in radians, in `[-π, π]`.
    pub bank_rad: f64,
    /// The reference drag at the current energy in m/s².
    pub reference_drag_mps2: f64,
    /// The sensed drag minus the reference drag in m/s².
    pub drag_error_mps2: f64,
    /// The azimuth from the vehicle to the landing site in radians,
    /// true, in `[0, 2π)`.
    pub heading_course_rad: f64,
    /// The heading of the velocity minus the azimuth to the landing
    /// site in radians, in `[-π, π)`. It is positive when the velocity
    /// points to the right of the site.
    pub heading_error_rad: f64,
    /// The specific mechanical energy in J/kg.
    pub energy_j_per_kg: f64,
    /// `false` while the drag is below the tracking threshold. Then the
    /// bank is zero, so the full lift points up.
    pub tracking: bool,
    /// `true` when the energy is at the TAEM energy or less.
    pub taem_reached: bool,
}

/// The specific mechanical energy `v²/2 − μ/r` of `state` in J/kg.
#[must_use]
pub fn specific_energy_j_per_kg(state: &StateVector, gm_m3_per_s2: f64) -> f64 {
    0.5 * state.velocity_mps.norm_squared() - gm_m3_per_s2 / state.position_m.norm()
}

/// Evaluates the entry law at the body-fixed `vehicle` state with the
/// sensed drag `sensed_drag_mps2`, and gives the command and the next
/// inline state.
///
/// # Errors
///
/// The errors of [`EntryCorridor::validate`],
/// [`TrajectoryError::NonFiniteState`],
/// [`TrajectoryError::InvalidParameter`] for a sensed drag that is not
/// finite and non-negative, [`TrajectoryError::DegenerateOrbit`] at the
/// center of the body, and [`TrajectoryError::UndefinedHeading`] on the
/// polar axis, at zero horizontal velocity, or above the landing site.
pub fn evaluate_entry(
    corridor: &EntryCorridor<'_>,
    state: EntryState,
    vehicle: &StateVector,
    sensed_drag_mps2: f64,
) -> Result<(EntryCommand, EntryState), TrajectoryError> {
    corridor.validate()?;
    let vehicle = vehicle.checked()?;
    if !(sensed_drag_mps2.is_finite() && sensed_drag_mps2 >= 0.0) {
        return Err(TrajectoryError::InvalidParameter {
            name: "sensed_drag_mps2",
        });
    }
    unit(&vehicle.position_m).ok_or(TrajectoryError::DegenerateOrbit)?;
    let heading = Heading::new(&vehicle, &corridor.landing_site_m)?;
    let gm = corridor.body.gm_m3_per_s2;
    let energy_j_per_kg = specific_energy_j_per_kg(&vehicle, gm);
    let (reference_drag_mps2, reference_rate) = corridor.reference_at(energy_j_per_kg);
    let tracking = sensed_drag_mps2 >= corridor.law.min_tracking_drag_mps2;
    let heading_error_rad = heading.error_rad();
    let deadband = corridor.law.heading_deadband_rad;
    let next = if tracking && heading_error_rad * state.bank_sign > deadband {
        state.reversed()
    } else {
        state
    };
    let bank_rad = if tracking {
        let cos_bank = corridor.law.cos_bank(
            &vehicle,
            gm,
            sensed_drag_mps2,
            reference_drag_mps2,
            reference_rate,
        );
        next.bank_sign * acos(cos_bank)
    } else {
        0.0
    };
    let command = EntryCommand {
        bank_rad,
        reference_drag_mps2,
        drag_error_mps2: sensed_drag_mps2 - reference_drag_mps2,
        heading_course_rad: rem_euclid(heading.course_rad, core::f64::consts::TAU),
        heading_error_rad,
        energy_j_per_kg,
        tracking,
        taem_reached: energy_j_per_kg <= corridor.taem_energy_j_per_kg,
    };
    Ok((command, next))
}

/// The angle `a − b` in radians, wrapped into `[-π, π)`.
pub(crate) fn wrap_pi(a: f64, b: f64) -> f64 {
    use core::f64::consts::{PI, TAU};
    rem_euclid(a - b + PI, TAU) - PI
}

/// The azimuth of the horizontal vector `v` in the local axes `east`
/// and `north`, in radians, clockwise from north.
pub(crate) fn azimuth_rad(v: &Vector3<f64>, east: &Vector3<f64>, north: &Vector3<f64>) -> f64 {
    atan2(v.dot(east), v.dot(north))
}
