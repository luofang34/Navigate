//! Measurement models of the inertial filter (ADR-0014).
//!
//! Each model linearizes one observation at the nominal state and gives
//! a [`Linearized`] update of the error state. The models read the
//! regime of the core that they update, so a replay after a frame
//! change uses the frame of the replayed core.

use nalgebra::{Matrix3, Rotation3, SMatrix, SVector, UnitQuaternion, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::{LocalTangentPlane, NedOffset};

use super::config::InertialConfig;
use super::estimate::{InertialCore, Linearized, Mode};
use super::nominal::N;
use super::regime::Regime;
use crate::environment::SurfaceFrame;
use crate::filter::rotation_between;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;

mod attitude;
mod cartesian;
mod magnetic;
mod surface;
mod terrain;

#[cfg(test)]
pub(crate) use terrain::terrain_clearance;

/// Updates `core` with `obs` at the state time of `core` and returns the
/// chi-square statistic. On an error, `core` is not changed.
pub(crate) fn apply_observation(
    core: &mut InertialCore,
    obs: &Observation,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason> {
    let fusion = &config.fusion;
    let vector_gate = fusion.innovation_gate_chi2;
    let scalar_gate = fusion.scalar_gate_chi2;
    let unsupported = || RejectionReason::UnsupportedMeasurement {
        kind: obs.value.kind(),
    };
    match &core.regime {
        Regime::BodySurface(surface) => {
            let surface = **surface;
            match &obs.value {
                ObservationValue::PositionFix { .. }
                | ObservationValue::VelocityFix { .. }
                | ObservationValue::Range { .. } => {
                    surface::apply_vector_or_range(core, &surface, obs, config, mode)
                }
                ObservationValue::Altitude {
                    altitude_m,
                    variance_m2,
                } => {
                    let model = surface::altitude(
                        core,
                        &surface,
                        *altitude_m,
                        *variance_m2,
                        obs.composition,
                    )?;
                    core.update(&model, scalar_gate, mode)
                }
                ObservationValue::TerrainClearance { .. } => {
                    let model = terrain::terrain_clearance(core, &surface, &obs.value)?;
                    core.update(&model, scalar_gate, mode)
                }
                ObservationValue::MagneticAnomaly { .. } => {
                    magnetic::apply(core, &surface, obs, config, mode)
                }
                ObservationValue::AttitudeFix {
                    attitude,
                    covariance,
                } => {
                    let to_local = vehicle_plane_to_local(core, &surface)?;
                    let model = attitude::attitude_fix(core, &to_local, attitude, covariance);
                    core.update(&model, vector_gate, mode)
                }
                ObservationValue::VisualPose { .. } => {
                    let model = attitude::visual_pose(core, &surface, &obs.value)?;
                    core.update(&model, fusion.pose_gate_chi2, mode)
                }
                _ => Err(unsupported()),
            }
        }
        Regime::CartesianSpace(_) => match &obs.value {
            ObservationValue::AttitudeFix {
                attitude,
                covariance,
            } => {
                let model =
                    attitude::attitude_fix(core, &UnitQuaternion::identity(), attitude, covariance);
                core.update(&model, vector_gate, mode)
            }
            ObservationValue::CartesianPositionFix { .. }
            | ObservationValue::CartesianVelocityFix { .. }
            | ObservationValue::CartesianRange { .. } => {
                cartesian::apply(core, &obs.value, config, mode)
            }
            _ => Err(unsupported()),
        },
    }
}

/// The rotation from the plane axes to the local NED axes at `position`.
pub(crate) fn plane_to_local_at(
    surface: &SurfaceFrame,
    position: GeodeticPosition,
) -> Result<Matrix3<f64>, RejectionReason> {
    let local = LocalTangentPlane::with_ellipsoid(position, surface.plane.ellipsoid())
        .map_err(|_| RejectionReason::NonFiniteValue)?;
    Ok(rotation_between(&surface.plane, &local))
}

/// The geodetic position of the nominal state.
pub(crate) fn vehicle_position(core: &InertialCore, surface: &SurfaceFrame) -> GeodeticPosition {
    let p = core.nominal.p;
    surface.plane.from_ned(&NedOffset::new(p.x, p.y, p.z))
}

/// The rotation from the plane axes to the local NED axes at the
/// vehicle, as a unit quaternion.
pub(crate) fn vehicle_plane_to_local(
    core: &InertialCore,
    surface: &SurfaceFrame,
) -> Result<UnitQuaternion<f64>, RejectionReason> {
    let rotation = plane_to_local_at(surface, vehicle_position(core, surface))?;
    Ok(quaternion_of(&rotation))
}

pub(crate) fn quaternion_of(rotation: &Matrix3<f64>) -> UnitQuaternion<f64> {
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*rotation))
}

/// A scalar update with the row `h` and the variance `variance`.
fn scalar(residual: f64, h: SMatrix<f64, 1, N>, variance: f64) -> Linearized<1> {
    Linearized {
        residual: SVector::<f64, 1>::new(residual),
        h,
        r: SMatrix::<f64, 1, 1>::new(variance),
    }
}

/// A direct update of the three error components at `offset`.
fn block(residual: Vector3<f64>, offset: usize, r: Matrix3<f64>) -> Linearized<3> {
    let mut h = SMatrix::<f64, 3, N>::zeros();
    h.fixed_view_mut::<3, 3>(0, offset)
        .copy_from(&Matrix3::identity());
    Linearized { residual, h, r }
}

/// The range row: the unit line of sight on the position error.
fn range(
    p: &Vector3<f64>,
    target: &Vector3<f64>,
    range_m: f64,
    variance_m2: f64,
) -> Result<Linearized<1>, RejectionReason> {
    let los = p - target;
    let distance = los.norm();
    if distance < MIN_RANGE_M {
        return Err(RejectionReason::ImplausibleCovariance);
    }
    let unit = los / distance;
    let mut h = SMatrix::<f64, 1, N>::zeros();
    h.fixed_view_mut::<1, 3>(0, 0).copy_from(&unit.transpose());
    Ok(scalar(range_m - distance, h, variance_m2))
}

/// Range below which the range Jacobian direction is undefined, in meters.
const MIN_RANGE_M: f64 = 1.0;
