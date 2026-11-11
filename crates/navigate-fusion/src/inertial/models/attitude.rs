//! Attitude fixes and visual poses (ADR-0014, ADR-0017).

use nalgebra::{Const, DimMin, Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use navigate_contract::{AttitudeQuaternion, SymmetricCov3};
use navigate_geodesy::BodyEllipsoid;

use super::block;
use crate::inertial::estimate::{InertialCore, Linearized};
use crate::inertial::local_level::{LocalLevel, body_fixed};
use crate::inertial::nominal::{ATT, POS, rotation_vector, to_unit_quaternion};
use crate::kalman::cov3_to_matrix;
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

/// The residual of a measured attitude: the body-axis rotation vector
/// `2·log(q_nom⁻¹ ⊗ q_meas)` in the hemisphere `w ≥ 0`. `to_reference`
/// rotates the frame axes into the reference axes of the measurement.
fn attitude_residual<const N: usize>(
    core: &InertialCore<N>,
    to_reference: &UnitQuaternion<f64>,
    attitude: &AttitudeQuaternion,
) -> (Vector3<f64>, UnitQuaternion<f64>)
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let nominal = to_reference * core.nominal.q;
    let residual = rotation_vector(&(nominal.inverse() * to_unit_quaternion(attitude)));
    (residual, nominal)
}

/// An attitude fix with `H_θ = I`. The covariance is in the body axes.
/// `to_reference` rotates the frame axes into the reference axes of the
/// fix: the local NED axes at the vehicle when the regime has a
/// reference ellipsoid, and the frame axes when it has none.
pub(super) fn attitude_fix<const N: usize>(
    core: &InertialCore<N>,
    to_reference: &UnitQuaternion<f64>,
    attitude: &AttitudeQuaternion,
    covariance: &SymmetricCov3,
) -> Linearized<3, N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let (residual, _) = attitude_residual(core, to_reference, attitude);
    block(residual, ATT, cov3_to_matrix(covariance))
}

/// A visual pose: the position error in the body-fixed axes and the
/// attitude error in the body axes. The position covariance rotates from
/// the local NED axes at the pose into the body-fixed axes. The pose
/// attitude is relative to the same local axes, and its covariance
/// rotates into the body axes: `R_body = C_blᵀ·R_ned·C_bl`.
pub(super) fn visual_pose<const N: usize>(
    core: &InertialCore<N>,
    ellipsoid: BodyEllipsoid,
    value: &ObservationValue,
) -> Result<Linearized<6, N>, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let ObservationValue::VisualPose {
        position,
        position_covariance,
        attitude,
        attitude_covariance,
    } = value
    else {
        return Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() });
    };
    let z = body_fixed(position, ellipsoid);
    let local = LocalLevel::at(*position);
    let r_position = local.ned_to_bf_cov(&cov3_to_matrix(position_covariance));
    let to_local = local.bf_to_ned_quaternion();
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
