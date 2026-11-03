//! Closed-loop steering of an orbital burn on the velocity to go.
//!
//! The velocity to go is `v_go = v_req(r) − v`. The required velocity
//! `v_req` comes from a [`BurnTarget`]. The law applies the
//! cross-product steering of Battin: with `Q = ∂v_req/∂r` and
//! `p = −Q·v_go`, the thrust acceleration is
//! `a_T = p⊥ + √(a² − |p⊥|²)·î_go`, where `p⊥` is the part of `p` normal
//! to `v_go` and `a` is the specific force. Then `d(v_go)/dt` stays
//! parallel to `v_go`, and all thrust goes to reduce `|v_go|`. For a
//! fixed required velocity, `Q = 0` and the thrust is along `v_go`.

use nalgebra::Vector3;
use navigate_contract::{FrameId, MonotonicNanos};

use crate::error::TrajectoryError;
use crate::float_math::sqrt;
use crate::state::{StateVector, unit};

/// The step in meters of the finite difference that gives `Q·v_go`.
const GRADIENT_STEP_M: f64 = 10.0;

/// A radius above the apoapsis by less than this fraction counts as the
/// apoapsis, because the radius at apoapsis is not exact in a burn.
const APOAPSIS_TOLERANCE: f64 = 1e-3;

/// The sign of the radial velocity on a target conic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RadialBranch {
    /// The radius increases: from periapsis to apoapsis.
    Ascending,
    /// The radius decreases: from apoapsis to periapsis.
    Descending,
}

/// The velocity that a burn must give.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum BurnTarget {
    /// A fixed required velocity in m/s in the frame axes. Use it for a
    /// burn that is short relative to the orbit period, or refresh it in
    /// each cycle.
    Velocity {
        /// The required velocity in m/s.
        velocity_mps: Vector3<f64>,
    },
    /// A velocity change in m/s from the velocity at ignition. The
    /// required velocity is `ignition_velocity_mps + delta_v_mps`, with
    /// the same short-burn condition as [`Self::Velocity`].
    DeltaV {
        /// The velocity at ignition in m/s.
        ignition_velocity_mps: Vector3<f64>,
        /// The velocity change in m/s.
        delta_v_mps: Vector3<f64>,
    },
    /// The conic with the given periapsis and apoapsis radii in the
    /// current orbit plane. The required velocity is a function of the
    /// position, and a coast along the conic keeps it. Thus the law
    /// stays correct over a long burn, for example a deorbit burn.
    Apsides {
        /// The periapsis radius in meters.
        periapsis_radius_m: f64,
        /// The apoapsis radius in meters.
        apoapsis_radius_m: f64,
        /// The gravitational parameter `μ` of the central body in
        /// m³/s².
        gm_m3_per_s2: f64,
        /// The branch of the conic that the vehicle flies after the
        /// burn.
        branch: RadialBranch,
    },
}

/// One planned burn in one Cartesian frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct BurnArc {
    /// The frame of the state, the target, and the thrust direction.
    pub frame: FrameId,
    /// The velocity that the burn must give.
    pub target: BurnTarget,
    /// The earliest ignition time.
    pub ignition_open: MonotonicNanos,
    /// The latest ignition time.
    pub ignition_close: MonotonicNanos,
    /// The engine stops when `|v_go|` is this value or less, in m/s.
    pub cutoff_velocity_to_go_mps: f64,
    /// The specific force of the engine in m/s².
    pub max_specific_force_mps2: f64,
}

impl BurnArc {
    /// Builds a burn from its parts. [`Self::validate`] checks the
    /// values.
    #[must_use]
    pub const fn new(
        frame: FrameId,
        target: BurnTarget,
        ignition_open: MonotonicNanos,
        ignition_close: MonotonicNanos,
        cutoff_velocity_to_go_mps: f64,
        max_specific_force_mps2: f64,
    ) -> Self {
        Self {
            frame,
            target,
            ignition_open,
            ignition_close,
            cutoff_velocity_to_go_mps,
            max_specific_force_mps2,
        }
    }

    /// Whether ignition is permitted at `now`.
    #[must_use]
    pub fn window_contains(&self, now: MonotonicNanos) -> bool {
        self.ignition_open <= now && now <= self.ignition_close
    }

    /// Whether the ignition window closed before `now`.
    #[must_use]
    pub fn window_closed(&self, now: MonotonicNanos) -> bool {
        now > self.ignition_close
    }

    /// Checks the window, the cutoff threshold, the specific force, and
    /// the target.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::InvalidParameter`] with the name of the first
    /// value that is not finite or is out of its range.
    pub fn validate(&self) -> Result<(), TrajectoryError> {
        let invalid = |name| Err(TrajectoryError::InvalidParameter { name });
        if self.ignition_close < self.ignition_open {
            return invalid("ignition_close");
        }
        if !(self.cutoff_velocity_to_go_mps.is_finite() && self.cutoff_velocity_to_go_mps > 0.0) {
            return invalid("cutoff_velocity_to_go_mps");
        }
        if !(self.max_specific_force_mps2.is_finite() && self.max_specific_force_mps2 > 0.0) {
            return invalid("max_specific_force_mps2");
        }
        self.target.validate()
    }
}

impl BurnTarget {
    fn validate(&self) -> Result<(), TrajectoryError> {
        let finite = |v: &Vector3<f64>| v.iter().all(|c| c.is_finite());
        let valid = match self {
            Self::Velocity { velocity_mps } => finite(velocity_mps),
            Self::DeltaV {
                ignition_velocity_mps,
                delta_v_mps,
            } => finite(ignition_velocity_mps) && finite(delta_v_mps),
            Self::Apsides {
                periapsis_radius_m,
                apoapsis_radius_m,
                gm_m3_per_s2,
                ..
            } => {
                periapsis_radius_m.is_finite()
                    && *periapsis_radius_m > 0.0
                    && apoapsis_radius_m.is_finite()
                    && apoapsis_radius_m >= periapsis_radius_m
                    && gm_m3_per_s2.is_finite()
                    && *gm_m3_per_s2 > 0.0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(TrajectoryError::InvalidParameter { name: "target" })
        }
    }

    /// The required velocity in m/s at the position `position_m`. The
    /// orbit normal `normal_unit` sets the plane of an
    /// [`Self::Apsides`] target.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::DegenerateOrbit`] when the position is at the
    /// center, and [`TrajectoryError::ApsidesUnreachable`] when the
    /// radius is outside the target conic.
    pub fn required_velocity(
        &self,
        position_m: &Vector3<f64>,
        normal_unit: &Vector3<f64>,
    ) -> Result<Vector3<f64>, TrajectoryError> {
        match *self {
            Self::Velocity { velocity_mps } => Ok(velocity_mps),
            Self::DeltaV {
                ignition_velocity_mps,
                delta_v_mps,
            } => Ok(ignition_velocity_mps + delta_v_mps),
            Self::Apsides {
                periapsis_radius_m,
                apoapsis_radius_m,
                gm_m3_per_s2,
                branch,
            } => {
                let radial_unit = unit(position_m).ok_or(TrajectoryError::DegenerateOrbit)?;
                let r = position_m.norm();
                if r < periapsis_radius_m || r > apoapsis_radius_m * (1.0 + APOAPSIS_TOLERANCE) {
                    return Err(TrajectoryError::ApsidesUnreachable {
                        radius_m: r,
                        periapsis_radius_m,
                        apoapsis_radius_m,
                    });
                }
                let sum = periapsis_radius_m + apoapsis_radius_m;
                let h = sqrt(2.0 * gm_m3_per_s2 * periapsis_radius_m * apoapsis_radius_m / sum);
                let speed_sq = gm_m3_per_s2 * (2.0 / r - 2.0 / sum);
                let horizontal = h / r;
                let radial = sqrt((speed_sq - horizontal * horizontal).max(0.0));
                let radial = match branch {
                    RadialBranch::Ascending => radial,
                    RadialBranch::Descending => -radial,
                };
                let along_unit = normal_unit.cross(&radial_unit);
                Ok(along_unit * horizontal + radial_unit * radial)
            }
        }
    }

    const fn depends_on_position(&self) -> bool {
        matches!(self, Self::Apsides { .. })
    }
}

/// The output of one evaluation of [`evaluate_burn`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct BurnCommand {
    /// The thrust direction, a unit vector in the frame axes. At cutoff
    /// it is the direction of `v_go`, or zero when `v_go` is zero.
    pub direction_unit: Vector3<f64>,
    /// The commanded specific force in m/s². It is `0.0` at cutoff.
    pub specific_force_mps2: f64,
    /// The velocity to go in m/s, in the frame axes.
    pub velocity_to_go_mps: Vector3<f64>,
    /// `true` when `|v_go|` is the cutoff threshold or less.
    pub cutoff: bool,
}

/// Evaluates the burn steering law at `state`.
///
/// The engine runs at [`BurnArc::max_specific_force_mps2`] until
/// `|v_go|` is [`BurnArc::cutoff_velocity_to_go_mps`] or less. This
/// function does not check the ignition window: the caller decides when
/// the burn starts with [`BurnArc::window_contains`].
///
/// # Errors
///
/// The errors of [`BurnArc::validate`] and of
/// [`BurnTarget::required_velocity`], [`TrajectoryError::NonFiniteState`],
/// and [`TrajectoryError::DegenerateOrbit`] when an [`BurnTarget::Apsides`]
/// target has no orbit plane at `state`.
pub fn evaluate_burn(arc: &BurnArc, state: &StateVector) -> Result<BurnCommand, TrajectoryError> {
    arc.validate()?;
    let state = state.checked()?;
    let normal_unit = if arc.target.depends_on_position() {
        unit(&state.position_m.cross(&state.velocity_mps))
            .ok_or(TrajectoryError::DegenerateOrbit)?
    } else {
        Vector3::zeros()
    };
    let required = arc
        .target
        .required_velocity(&state.position_m, &normal_unit)?;
    let v_go = required - state.velocity_mps;
    let v_go_norm = v_go.norm();
    let go_unit = match unit(&v_go) {
        Some(go_unit) if v_go_norm > arc.cutoff_velocity_to_go_mps => go_unit,
        last => {
            return Ok(BurnCommand {
                direction_unit: last.unwrap_or_else(Vector3::zeros),
                specific_force_mps2: 0.0,
                velocity_to_go_mps: v_go,
                cutoff: true,
            });
        }
    };
    let p = if arc.target.depends_on_position() {
        let shifted = state.position_m + go_unit * GRADIENT_STEP_M;
        let shifted_required = arc.target.required_velocity(&shifted, &normal_unit)?;
        -(shifted_required - required) * (v_go_norm / GRADIENT_STEP_M)
    } else {
        Vector3::zeros()
    };
    Ok(BurnCommand {
        direction_unit: cross_product_direction(&p, &go_unit, arc.max_specific_force_mps2),
        specific_force_mps2: arc.max_specific_force_mps2,
        velocity_to_go_mps: v_go,
        cutoff: false,
    })
}

/// The thrust direction that keeps `d(v_go)/dt` parallel to `go_unit`
/// at the specific force `force`. When `|p⊥|` is the force or more, no
/// direction can do that, and the thrust goes along `go_unit`.
fn cross_product_direction(p: &Vector3<f64>, go_unit: &Vector3<f64>, force: f64) -> Vector3<f64> {
    let p_normal = p - go_unit * p.dot(go_unit);
    let normal_sq = p_normal.norm_squared();
    let force_sq = force * force;
    if !normal_sq.is_finite() || normal_sq >= force_sq {
        return *go_unit;
    }
    let thrust = p_normal + go_unit * sqrt(force_sq - normal_sq);
    unit(&thrust).unwrap_or(*go_unit)
}

#[cfg(test)]
mod tests;
