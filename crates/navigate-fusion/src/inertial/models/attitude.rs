//! Attitude fixes and visual poses (ADR-0014).

use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use navigate_contract::{AttitudeQuaternion, SymmetricCov3};

use super::{block, plane_to_local_at, quaternion_of};
use crate::environment::SurfaceFrame;
use crate::filter::cov3_to_matrix;
use crate::inertial::estimate::{InertialCore, Linearized};
use crate::inertial::nominal::{ATT, N, POS, rotation_vector, to_unit_quaternion};
use crate::navigation_filter::reanchor::vehicle_to_plane;
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

/// The residual of a measured attitude: the body-axis rotation vector
/// `2·log(q_nom⁻¹ ⊗ q_meas)` in the hemisphere `w ≥ 0`. `to_reference`
/// rotates the frame axes into the reference axes of the measurement.
fn attitude_residual(
    core: &InertialCore,
    to_reference: &UnitQuaternion<f64>,
    attitude: &AttitudeQuaternion,
) -> (Vector3<f64>, UnitQuaternion<f64>) {
    let nominal = to_reference * core.nominal.q;
    let residual = rotation_vector(&(nominal.inverse() * to_unit_quaternion(attitude)));
    (residual, nominal)
}

/// An attitude fix with `H_θ = I`. The covariance is in the body axes.
pub(super) fn attitude_fix(
    core: &InertialCore,
    to_reference: &UnitQuaternion<f64>,
    attitude: &AttitudeQuaternion,
    covariance: &SymmetricCov3,
) -> Linearized<3> {
    let (residual, _) = attitude_residual(core, to_reference, attitude);
    block(residual, ATT, cov3_to_matrix(covariance))
}

/// A visual pose: the position error in the plane axes and the attitude
/// error in the body axes. The position covariance rotates from the
/// local NED axes at the pose into the plane axes. The attitude
/// covariance rotates from the same local axes into the body axes:
/// `R_body = C_blᵀ·R_ned·C_bl`.
pub(super) fn visual_pose(
    core: &InertialCore,
    surface: &SurfaceFrame,
    value: &ObservationValue,
) -> Result<Linearized<6>, RejectionReason> {
    let ObservationValue::VisualPose {
        position,
        position_covariance,
        attitude,
        attitude_covariance,
    } = value
    else {
        return Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() });
    };
    let plane = &surface.plane;
    let offset = plane.to_ned(position);
    let z = Vector3::new(offset.north_m, offset.east_m, offset.down_m);
    let (_, r_position) =
        vehicle_to_plane(plane, *position, z, cov3_to_matrix(position_covariance));
    let to_local = quaternion_of(&plane_to_local_at(surface, *position)?);
    let (r_attitude, nominal_local) = attitude_residual(core, &to_local, attitude);
    let c_bl = nominal_local.to_rotation_matrix().into_inner();
    let r_attitude_body = c_bl.transpose() * cov3_to_matrix(attitude_covariance) * c_bl;
    let mut residual = SVector::<f64, 6>::zeros();
    residual
        .fixed_rows_mut::<3>(0)
        .copy_from(&(z - core.nominal.p));
    residual.fixed_rows_mut::<3>(3).copy_from(&r_attitude);
    let mut h = SMatrix::<f64, 6, N>::zeros();
    h.fixed_view_mut::<3, 3>(0, POS)
        .copy_from(&Matrix3::identity());
    h.fixed_view_mut::<3, 3>(3, ATT)
        .copy_from(&Matrix3::identity());
    let mut r = SMatrix::<f64, 6, 6>::zeros();
    r.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_position);
    r.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&((r_attitude_body + r_attitude_body.transpose()) * 0.5));
    Ok(Linearized { residual, h, r })
}
