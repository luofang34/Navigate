//! Rotations of a position and a velocity between a body-fixed frame
//! and an inertial frame (ADR-0016).
//!
//! A [`FrameTransform`] holds the rotation from the source axes to the
//! target axes and the angular velocity of the target axes relative to
//! the source axes. [`FrameTransform::transform_state`] applies the
//! rotation to a position and adds the transport velocity `−ω × r` to a
//! velocity.
//!
//! For the Earth, [`era_rad`] gives the IAU 2000 Earth rotation angle at
//! a UT1 instant, and [`polar_motion_matrix`] gives the polar motion
//! matrix `W(x_p, y_p)`. [`FrameTransform::celestial_to_terrestrial`]
//! combines them into the rotation from the celestial intermediate
//! reference system (CIRS) to the international terrestrial reference
//! system (ITRS). The rotation omits the TIO locator `s′`, which is less
//! than 0.1 milliarcsecond for the present century.
//! [`FrameTransform::axial_spin`] gives the rotation of a body that
//! turns at a uniform rate about its `z` axis, for the Moon, Mars, and
//! other bodies.

use nalgebra::{Matrix3, Vector3};
use navigate_contract::{PhysicalInstant, TimeScale};

use crate::float_math::{floor, sin_cos};
use crate::time_scales::{EarthOrientation, TimeScaleError, expect_scale};

#[cfg(test)]
mod tests;

/// The Earth rotation angle at J2000.0 UT1, in turns (IAU 2000).
const ERA_AT_J2000_TURNS: f64 = 0.779_057_273_264;

/// The part of the Earth rotation rate above one turn per UT1 day, in
/// turns per day (IAU 2000).
const ERA_EXCESS_TURNS_PER_DAY: f64 = 0.002_737_811_911_354_48;

/// Seconds in one day.
const SECONDS_PER_DAY: i64 = 86_400;

/// The rotation rate of the Earth about the celestial intermediate pole
/// in rad/s of UT1: `2π·1.00273781191135448 / 86400`.
pub const EARTH_ROTATION_RATE_RAD_PER_S: f64 = 7.292_115_146_706_979e-5;

/// The rotation from the source axes to the target axes, and the
/// angular velocity of the target axes relative to the source axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTransform {
    /// The direction cosine matrix from the source axes to the target
    /// axes: `r_target = C·r_source`. Row `i` is target axis `i` in the
    /// source axes.
    pub dcm_source_to_target: [[f64; 3]; 3],
    /// The angular velocity `ω` of the target axes relative to the
    /// source axes, in rad/s in the source axes.
    pub angular_rate_source_rad_s: [f64; 3],
}

impl FrameTransform {
    /// The transform that changes nothing.
    pub const IDENTITY: Self = Self {
        dcm_source_to_target: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        angular_rate_source_rad_s: [0.0; 3],
    };

    /// The transform from a non-rotating frame to the body-fixed frame of
    /// a body that turns about the common `z` axis. `angle_rad` is the
    /// rotation angle of the body-fixed `x` axis from the non-rotating
    /// `x` axis, and `rate_rad_s` is its rate. The angle increases in the
    /// positive sense about `z`.
    #[must_use]
    pub fn axial_spin(angle_rad: f64, rate_rad_s: f64) -> Self {
        Self {
            dcm_source_to_target: rotation_z(angle_rad),
            angular_rate_source_rad_s: [0.0, 0.0, rate_rad_s],
        }
    }

    /// The transform from the CIRS to the ITRS at the UT1 instant `ut1`:
    /// `C = W(x_p, y_p)ᵀ·R_z(ERA)`, with the Earth rotation about the
    /// `z` axis of the CIRS.
    ///
    /// # Errors
    ///
    /// [`TimeScaleError::WrongScale`] for an instant that is not on UT1,
    /// and [`TimeScaleError::ImplausibleEarthOrientation`].
    pub fn celestial_to_terrestrial(
        ut1: PhysicalInstant,
        orientation: &EarthOrientation,
    ) -> Result<Self, TimeScaleError> {
        if !orientation.is_plausible() {
            return Err(TimeScaleError::ImplausibleEarthOrientation);
        }
        let spin = Self::axial_spin(era_rad(ut1)?, EARTH_ROTATION_RATE_RAD_PER_S);
        let polar = to_matrix(&polar_motion_matrix(
            orientation.polar_motion_xp_rad,
            orientation.polar_motion_yp_rad,
        ));
        Ok(spin.then(&Self {
            dcm_source_to_target: from_matrix(&polar.transpose()),
            angular_rate_source_rad_s: [0.0; 3],
        }))
    }

    /// The position and the velocity in the target frame of the
    /// position `pos_m` in meters and the velocity `vel_mps` in m/s in
    /// the source frame: `r_t = C·r_s` and `v_t = C·(v_s − ω × r_s)`.
    #[must_use]
    pub fn transform_state(&self, pos_m: [f64; 3], vel_mps: [f64; 3]) -> ([f64; 3], [f64; 3]) {
        let c = to_matrix(&self.dcm_source_to_target);
        let r = Vector3::from(pos_m);
        let omega = Vector3::from(self.angular_rate_source_rad_s);
        let transport = omega.cross(&r);
        let v = Vector3::from(vel_mps) - transport;
        ((c * r).into(), (c * v).into())
    }

    /// The transform from the target frame back to the source frame. The
    /// angular velocity of the source axes relative to the target axes is
    /// `−C·ω` in the target axes.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let c = to_matrix(&self.dcm_source_to_target);
        let omega = Vector3::from(self.angular_rate_source_rad_s);
        Self {
            dcm_source_to_target: from_matrix(&c.transpose()),
            angular_rate_source_rad_s: (-(c * omega)).into(),
        }
    }

    /// This transform followed by `next`, which starts at the target
    /// frame of this transform. The angular velocities add in the source
    /// axes: `ω = ω₁ + C₁ᵀ·ω₂`.
    #[must_use]
    pub fn then(&self, next: &Self) -> Self {
        let first = to_matrix(&self.dcm_source_to_target);
        let second = to_matrix(&next.dcm_source_to_target);
        let omega = Vector3::from(self.angular_rate_source_rad_s)
            + first.transpose() * Vector3::from(next.angular_rate_source_rad_s);
        Self {
            dcm_source_to_target: from_matrix(&(second * first)),
            angular_rate_source_rad_s: omega.into(),
        }
    }
}

/// The IAU 2000 Earth rotation angle in radians, in `[0, 2π)`, at the
/// UT1 instant `ut1`:
/// `ERA = 2π·(0.7790572732640 + 1.00273781191135448·T_u)`, with `T_u`
/// the days of UT1 from J2000.0.
///
/// The function separates the whole days from the fraction of a day
/// before the multiplication. Thus the angle keeps a resolution of a few
/// microarcseconds over several centuries.
///
/// # Errors
///
/// [`TimeScaleError::WrongScale`] for an instant that is not on UT1.
pub fn era_rad(ut1: PhysicalInstant) -> Result<f64, TimeScaleError> {
    expect_scale(ut1, TimeScale::Ut1)?;
    let seconds = ut1.seconds_since_j2000();
    let whole_days = seconds.div_euclid(SECONDS_PER_DAY);
    let second_of_day = seconds.rem_euclid(SECONDS_PER_DAY);
    let subsecond = ut1.subsec_attoseconds() as f64 / 1e18;
    let day_fraction = (second_of_day as f64 + subsecond) / SECONDS_PER_DAY as f64;
    let days = whole_days as f64 + day_fraction;
    let turns = ERA_AT_J2000_TURNS + day_fraction + ERA_EXCESS_TURNS_PER_DAY * days;
    // A small negative `turns` can round its fraction up to one turn.
    let fraction = turns - floor(turns);
    let fraction = if fraction < 1.0 { fraction } else { 0.0 };
    Ok(fraction * core::f64::consts::TAU)
}

/// The polar motion matrix `W = R_2(x_p)·R_1(y_p)` from the ITRS to the
/// terrestrial intermediate reference system (TIRS), with the polar
/// motion `x_p` and `y_p` in radians. `R_1` and `R_2` rotate the axes
/// about `x` and `y`.
#[must_use]
pub fn polar_motion_matrix(xp_rad: f64, yp_rad: f64) -> [[f64; 3]; 3] {
    let (sx, cx) = sin_cos(xp_rad);
    let (sy, cy) = sin_cos(yp_rad);
    let r2 = Matrix3::new(cx, 0.0, -sx, 0.0, 1.0, 0.0, sx, 0.0, cx);
    let r1 = Matrix3::new(1.0, 0.0, 0.0, 0.0, cy, sy, 0.0, -sy, cy);
    from_matrix(&(r2 * r1))
}

/// The rotation `R_3(θ)` of the axes about `z` by `angle_rad`.
fn rotation_z(angle_rad: f64) -> [[f64; 3]; 3] {
    let (s, c) = sin_cos(angle_rad);
    [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn to_matrix(rows: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| rows[i][j])
}

fn from_matrix(m: &Matrix3<f64>) -> [[f64; 3]; 3] {
    [0, 1, 2].map(|i| [m[(i, 0)], m[(i, 1)], m[(i, 2)]])
}
