//! Fusion observations from one visual estimate: a position fix or a visual pose.
//!
//! Each builder declares the body-fixed frame of the reference ellipsoid of the
//! visual frame (ADR-0013). An estimate in a [`LocalFrame::BodyTangentEnu`] frame
//! on the Moon gives a lunar observation, so a lunar filter can admit it.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use navigate_contract::{
    AttitudeQuaternion, GeodeticPosition, MonotonicNanos, ObservationStamp, SensorClass,
    SourceComposition, SymmetricCov3, WrappingSequence,
};
use navigate_fusion::{Observation, ObservationValue};

use crate::fusion_bridge::{
    VerticalDatum, VisualAttitudeBudget, VisualErrorBudget, VisualFusionError, VisualSourceIdentity,
};
use crate::{Estimate, LocalFrame};

/// Build a position fix from one accepted visual estimate.
///
/// The fix is the position of the camera optical centre on the reference
/// ellipsoid of the visual frame. The covariance is in the local north, east,
/// down axes at the fix. It adds the declared budget and the point-coordinate
/// displacement to the image-geometry covariance. This function does not screen
/// repeated evidence or frame order. [`crate::fusion_bridge::VisualFixSource`]
/// does that for one capture stream.
///
/// # Errors
/// Refuses a budget term that is not finite and positive, a vertical datum that
/// the frame cannot use, positions outside geodetic range, and unusable
/// covariances.
pub fn position_fix_observation(
    estimate: &Estimate,
    identity: &VisualSourceIdentity,
    budget: &VisualErrorBudget,
) -> Result<Observation, VisualFusionError> {
    position_fix_observation_propagated(
        estimate,
        identity,
        budget,
        estimate.stamp.capture_time_ns,
        estimate.stamp.sequence,
        &Matrix3::zeros(),
    )
}

/// Build a position fix for a delayed anchor propagated to `arrival_time_ns`.
///
/// The builder adds `q_vo_enu_m2` in frame axes before it rotates the covariance
/// into north, east, down axes. It stamps the observation at `arrival_time_ns`
/// and declares the body-fixed frame of [`LocalFrame::reference_ellipsoid`].
///
/// # Errors
/// Refuses negative lag, a zero horizontal propagation covariance when
/// `arrival_time_ns > capture_time_ns`, and all conditions of
/// [`position_fix_observation`].
pub fn position_fix_observation_propagated(
    estimate: &Estimate,
    identity: &VisualSourceIdentity,
    budget: &VisualErrorBudget,
    arrival_time_ns: u64,
    arrival_sequence: u64,
    q_vo_enu_m2: &Matrix3<f64>,
) -> Result<Observation, VisualFusionError> {
    if arrival_time_ns < estimate.stamp.capture_time_ns {
        return Err(VisualFusionError::FrameOrder {
            previous_ns: estimate.stamp.capture_time_ns,
            received_ns: arrival_time_ns,
        });
    }
    budget.validate()?;
    let require_vo = arrival_time_ns > estimate.stamp.capture_time_ns;
    if !is_valid_vo_cov(q_vo_enu_m2, require_vo) {
        return Err(VisualFusionError::InvalidCovariance);
    }
    let position = geodetic_position(estimate, budget)?;
    let covariance = position_covariance(estimate, budget, q_vo_enu_m2, 2.0)?;
    Ok(stamped_at(
        estimate,
        identity,
        arrival_time_ns,
        arrival_sequence,
        ObservationValue::PositionFix {
            position,
            covariance: to_cov3(&covariance),
        },
    ))
}

/// Build a six-degree-of-freedom visual pose from one accepted visual estimate.
///
/// The position is the camera optical centre, as in [`position_fix_observation`].
/// The attitude is the vehicle body attitude: the rotation from the body axes
/// (forward, right, down) to the local north, east, down axes at the position.
/// [`VisualAttitudeBudget::eye_to_body`] gives the camera mounting.
///
/// The image geometry gives one joint covariance of position and attitude. A
/// visual pose carries one block for each. Two times each diagonal block bounds
/// the joint covariance in Loewner order, so the builder doubles both
/// image-geometry blocks before it adds the budgets. The attitude covariance is
/// about the local north, east, down axes at the position.
///
/// # Errors
/// Refuses the conditions of [`position_fix_observation`], an attitude budget
/// term that is not finite and positive, and a mounting rotation that is not
/// finite.
pub fn visual_pose_observation(
    estimate: &Estimate,
    identity: &VisualSourceIdentity,
    budget: &VisualErrorBudget,
    attitude_budget: &VisualAttitudeBudget,
) -> Result<Observation, VisualFusionError> {
    visual_pose_observation_propagated(
        estimate,
        identity,
        budget,
        attitude_budget,
        estimate.stamp.capture_time_ns,
        estimate.stamp.sequence,
        &Matrix3::zeros(),
        &Matrix3::zeros(),
    )
}

/// Build a six-degree-of-freedom visual pose propagated to `arrival_time_ns`.
///
/// The builder adds `q_vo_enu_m2` in frame axes and `q_vo_att_rad2` in camera
/// axes before it transforms and scales both covariance blocks. It stamps the
/// observation at `arrival_time_ns` and declares the body-fixed frame of
/// [`LocalFrame::reference_ellipsoid`].
///
/// # Errors
/// Refuses negative lag, a zero propagation covariance when
/// `arrival_time_ns > capture_time_ns`, and all conditions of
/// [`visual_pose_observation`].
#[expect(
    clippy::too_many_arguments,
    reason = "propagated 6-DOF pose observation takes estimate, identity, position/attitude budgets, arrival stamp, and position/attitude VO covariances"
)]
pub fn visual_pose_observation_propagated(
    estimate: &Estimate,
    identity: &VisualSourceIdentity,
    budget: &VisualErrorBudget,
    attitude_budget: &VisualAttitudeBudget,
    arrival_time_ns: u64,
    arrival_sequence: u64,
    q_vo_enu_m2: &Matrix3<f64>,
    q_vo_att_rad2: &Matrix3<f64>,
) -> Result<Observation, VisualFusionError> {
    if arrival_time_ns < estimate.stamp.capture_time_ns {
        return Err(VisualFusionError::FrameOrder {
            previous_ns: estimate.stamp.capture_time_ns,
            received_ns: arrival_time_ns,
        });
    }
    budget.validate()?;
    attitude_budget.validate()?;
    let require_vo = arrival_time_ns > estimate.stamp.capture_time_ns;
    if !is_valid_vo_cov(q_vo_enu_m2, require_vo) || !is_valid_att_vo_cov(q_vo_att_rad2, require_vo)
    {
        return Err(VisualFusionError::InvalidCovariance);
    }
    let position = geodetic_position(estimate, budget)?;
    let position_cov = position_covariance(estimate, budget, q_vo_enu_m2, 4.0)?;
    let ned_from_eye = ned_from_frame(estimate)? * estimate.pose.orientation;
    let body_to_ned = ned_from_eye * attitude_budget.eye_to_body.inverse();
    let rotation = ned_from_eye.to_rotation_matrix().into_inner();
    let eye: Matrix3<f64> = estimate
        .geometry_covariance
        .fixed_view::<3, 3>(3, 3)
        .into_owned();
    let eye_total = eye + q_vo_att_rad2;
    let geometric = rotation * eye_total * rotation.transpose();
    if (geometric - geometric.transpose()).norm() > 1e-12 || !usable(&geometric) {
        return Err(VisualFusionError::InvalidCovariance);
    }
    let external = attitude_budget.variance_rad2();
    let attitude_cov = geometric * 4.0 + Matrix3::identity() * (2.0 * external);
    if !usable(&attitude_cov) {
        return Err(VisualFusionError::InvalidCovariance);
    }
    let q = body_to_ned.quaternion();
    Ok(stamped_at(
        estimate,
        identity,
        arrival_time_ns,
        arrival_sequence,
        ObservationValue::VisualPose {
            position,
            position_covariance: to_cov3(&position_cov),
            attitude: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
            attitude_covariance: to_cov3(&attitude_cov),
        },
    ))
}

/// Geodetic camera position with the declared vertical datum applied.
fn geodetic_position(
    estimate: &Estimate,
    budget: &VisualErrorBudget,
) -> Result<GeodeticPosition, VisualFusionError> {
    let exact_ellipsoid = matches!(
        estimate.frame,
        LocalFrame::Wgs84Enu { .. } | LocalFrame::BodyTangentEnu { .. }
    );
    if exact_ellipsoid && budget.vertical_datum != VerticalDatum::Wgs84Ellipsoid {
        return Err(VisualFusionError::FrameDatumMismatch);
    }
    let [lat, lon, alt] = estimate.frame.geodetic(estimate.pose.position);
    let geodetic = GeodeticPosition::new(
        lat.to_radians(),
        lon.to_radians(),
        budget.ellipsoid_height_m(alt),
    );
    if !geodetic.is_plausible() {
        return Err(VisualFusionError::ImplausiblePosition);
    }
    Ok(geodetic)
}

/// Observation stamp at `(stamp_time_ns, stamp_sequence)` with the body-fixed frame.
fn stamped_at(
    estimate: &Estimate,
    identity: &VisualSourceIdentity,
    stamp_time_ns: u64,
    stamp_sequence: u64,
    value: ObservationValue,
) -> Observation {
    let stamp = ObservationStamp::new(
        identity.source,
        identity.epoch,
        WrappingSequence::new(low_bits(stamp_sequence)),
        MonotonicNanos::from_nanos(stamp_time_ns),
        identity.clock,
    );
    Observation::new(
        stamp,
        value,
        SourceComposition::of(SensorClass::VisualLandmark),
    )
    .with_frame(estimate.frame.reference_ellipsoid().frame())
}

/// Low 32 bits keep the wrap order of a wrapping 64-bit frame sequence.
fn low_bits(sequence: u64) -> u32 {
    u32::try_from(sequence & u64::from(u32::MAX)).unwrap_or(u32::MAX)
}

/// Rotation from the frame axes to the local north, east, down axes at the camera.
fn ned_from_frame(estimate: &Estimate) -> Result<UnitQuaternion<f64>, VisualFusionError> {
    let [lat, lon, _] = estimate.frame.geodetic(estimate.pose.position);
    let local_to_frame = estimate
        .frame
        .local_rotation([lat, lon])
        .map_err(|_| VisualFusionError::ImplausiblePosition)?;
    let enu_to_ned = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
        Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    ));
    Ok(enu_to_ned * local_to_frame.inverse())
}

/// Position fix covariance in north, east, down axes.
#[cfg(test)]
pub(super) fn ned_covariance(
    estimate: &Estimate,
    budget: &VisualErrorBudget,
) -> Result<SymmetricCov3, VisualFusionError> {
    position_covariance(estimate, budget, &Matrix3::zeros(), 2.0).map(|ned| to_cov3(&ned))
}

/// Rotate the image-geometry position covariance from frame axes into north,
/// east, down, scale it, and add the declared budget.
fn position_covariance(
    estimate: &Estimate,
    budget: &VisualErrorBudget,
    q_vo_enu_m2: &Matrix3<f64>,
    geometric_scale: f64,
) -> Result<Matrix3<f64>, VisualFusionError> {
    let enu_geom: Matrix3<f64> = estimate
        .geometry_covariance
        .fixed_view::<3, 3>(0, 0)
        .into_owned();
    let enu = enu_geom + q_vo_enu_m2;
    let axes = ned_from_frame(estimate)?.to_rotation_matrix().into_inner();
    let geometric = axes * enu * axes.transpose();
    if (geometric - geometric.transpose()).norm() > 1e-9 || !usable(&geometric) {
        return Err(VisualFusionError::InvalidCovariance);
    }
    let point_displacement_m = estimate.frame.model_error_m(estimate.pose.position);
    let (horizontal, vertical) = budget.variances_m2(point_displacement_m);
    let external = Matrix3::from_diagonal(&Vector3::new(
        8.0 * horizontal,
        8.0 * horizontal,
        4.0 * vertical,
    ));
    let ned = geometric * geometric_scale + external;
    if !usable(&ned) {
        return Err(VisualFusionError::InvalidCovariance);
    }
    Ok(ned)
}

fn is_valid_vo_cov(q: &Matrix3<f64>, require_positive: bool) -> bool {
    if !q.iter().all(|v| v.is_finite()) || (q - q.transpose()).norm() > 1e-9 {
        return false;
    }
    if q.symmetric_eigen().eigenvalues.iter().any(|v| *v < -1e-12) {
        return false;
    }
    !require_positive || (q[(0, 0)] + q[(1, 1)]) > 0.0
}

fn is_valid_att_vo_cov(q: &Matrix3<f64>, require_positive: bool) -> bool {
    if !q.iter().all(|v| v.is_finite()) || (q - q.transpose()).norm() > 1e-12 {
        return false;
    }
    if q.symmetric_eigen().eigenvalues.iter().any(|v| *v < -1e-12) {
        return false;
    }
    !require_positive || q.trace() > 0.0
}

/// Whether a covariance is finite and positive semidefinite.
fn usable(covariance: &Matrix3<f64>) -> bool {
    covariance.iter().all(|v| v.is_finite())
        && covariance
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .all(|v| *v >= 0.0)
}

fn to_cov3(m: &Matrix3<f64>) -> SymmetricCov3 {
    SymmetricCov3::from_upper_triangle([
        m[(0, 0)],
        m[(0, 1)],
        m[(0, 2)],
        m[(1, 1)],
        m[(1, 2)],
        m[(2, 2)],
    ])
}

#[cfg(test)]
mod tests;
