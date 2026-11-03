//! Slant range to sloped terrain along a body-fixed beam (ADR-0014).
//!
//! The terrain near the nominal horizontal position is the plane
//! `T(n, e) = T₀ + s_N·n + s_E·e` in the local NED axes at the vehicle.
//! The beam `u = C_bl·b` meets it at the range `ρ = (h − T₀)/κ` with
//! `m = [s_N, s_E, 1]` and `κ = m·u`. A beam close to the terrain plane
//! has a small `κ`, and the range is then very sensitive to the attitude.

use nalgebra::{SMatrix, Vector3};

use super::{plane_to_local_at, scalar, vehicle_position};
use crate::environment::SurfaceFrame;
use crate::inertial::estimate::{InertialCore, Linearized};
use crate::inertial::nominal::{ATT, N, POS};
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

/// Smallest `κ = m·u` that the model accepts. Below it the beam is close
/// to parallel with the terrain or points away from it.
const MIN_INCIDENCE: f64 = 0.05;

/// The linearized terrain clearance. The range depends on the position
/// through the height and the terrain slope, and on the attitude through
/// the beam direction:
/// `∂ρ/∂p = −(1/κ)·mᵀ·R_pl` and `∂ρ/∂δθ = (ρ/κ)·mᵀ·C_bl·[b×]`.
/// The barometric bias does not enter: a surface altimeter measures a
/// geometric range. A beam with `κ ≤ 0.05` or a vehicle at or below the
/// terrain is refused with [`RejectionReason::ImplausibleCovariance`].
pub(crate) fn terrain_clearance(
    core: &InertialCore,
    surface: &SurfaceFrame,
    value: &ObservationValue,
) -> Result<Linearized<1>, RejectionReason> {
    let ObservationValue::TerrainClearance {
        clearance_m,
        terrain_height_m,
        terrain_slope_north_east: [slope_north, slope_east],
        beam_body_unit,
        variance_m2,
    } = value
    else {
        return Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() });
    };
    let vehicle = vehicle_position(core, surface);
    let plane_to_local = plane_to_local_at(surface, vehicle)?;
    let c_bl = plane_to_local * core.nominal.c_b_to_f();
    let beam = Vector3::from(*beam_body_unit);
    let m = Vector3::new(*slope_north, *slope_east, 1.0);
    let incidence = m.dot(&(c_bl * beam));
    let height_above_terrain_m = vehicle.altitude_m - terrain_height_m;
    if !(incidence > MIN_INCIDENCE && height_above_terrain_m > 0.0) {
        return Err(RejectionReason::ImplausibleCovariance);
    }
    let rho = height_above_terrain_m / incidence;
    let mut h = SMatrix::<f64, 1, N>::zeros();
    h.fixed_view_mut::<1, 3>(0, POS)
        .copy_from(&(-(1.0 / incidence) * m.transpose() * plane_to_local));
    h.fixed_view_mut::<1, 3>(0, ATT)
        .copy_from(&((rho / incidence) * m.transpose() * c_bl * beam.cross_matrix()));
    Ok(scalar(clearance_m - rho, h, *variance_m2))
}
