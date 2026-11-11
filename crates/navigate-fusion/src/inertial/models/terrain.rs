//! Slant range to sloped terrain along a body-fixed beam (ADR-0014,
//! ADR-0017).
//!
//! The terrain near the nominal horizontal position is the plane
//! `T(n, e) = T₀ + s_N·n + s_E·e` in the local NED axes at the vehicle.
//! The beam `u = C_bl·b` meets it at the range `ρ = (h − T₀)/κ` with
//! `m = [s_N, s_E, 1]` and `κ = m·u`. A beam close to the terrain plane
//! has a small `κ`, and the range is then very sensitive to the attitude.

use nalgebra::{Const, DimMin, SMatrix, Vector3};
use navigate_geodesy::BodyEllipsoid;

use super::{scalar, vehicle_level};
use crate::inertial::estimate::{InertialCore, Linearized};
use crate::inertial::nominal::{ATT, POS};
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

/// Smallest `κ = m·u` that the model accepts. Below it the beam is close
/// to parallel with the terrain or points away from it.
const MIN_INCIDENCE: f64 = 0.05;

/// The linearized terrain clearance. The range depends on the position
/// through the height and the terrain slope, and on the attitude through
/// the beam direction:
/// `∂ρ/∂p = −(1/κ)·mᵀ·R_fl` and `∂ρ/∂δθ = (ρ/κ)·mᵀ·C_bl·[b×]`. Here
/// `R_fl` is the rotation from the body-fixed axes to the local axes,
/// and `C_bl = R_fl·C_bf`. The barometric bias does not enter: a surface altimeter measures a
/// geometric range. A beam with `κ ≤ 0.05` or a vehicle at or below the
/// terrain is refused with [`RejectionReason::ImplausibleCovariance`].
pub(crate) fn terrain_clearance<const N: usize>(
    core: &InertialCore<N>,
    ellipsoid: BodyEllipsoid,
    value: &ObservationValue,
) -> Result<Linearized<1, N>, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
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
    let local = vehicle_level(core, ellipsoid);
    let frame_to_local = local.bf_to_ned();
    let c_bl = frame_to_local * core.nominal.c_b_to_f();
    let beam = Vector3::from(*beam_body_unit);
    let m = Vector3::new(*slope_north, *slope_east, 1.0);
    let incidence = m.dot(&(c_bl * beam));
    let height_above_terrain_m = local.position.altitude_m - terrain_height_m;
    if !(incidence > MIN_INCIDENCE && height_above_terrain_m > 0.0) {
        return Err(RejectionReason::ImplausibleCovariance);
    }
    let rho = height_above_terrain_m / incidence;
    let mut h = SMatrix::<f64, 1, N>::zeros();
    h.fixed_view_mut::<1, 3>(0, POS)
        .copy_from(&(-(1.0 / incidence) * m.transpose() * frame_to_local));
    h.fixed_view_mut::<1, 3>(0, ATT)
        .copy_from(&((rho / incidence) * m.transpose() * c_bl * beam.cross_matrix()));
    Ok(scalar(clearance_m - rho, h, *variance_m2))
}
