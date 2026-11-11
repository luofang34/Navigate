//! Measurement models of the inertial filter (ADR-0014, ADR-0017,
//! ADR-0018).
//!
//! Each model linearizes one observation at the nominal state and gives
//! a [`Linearized`] update of the error state. The position and the
//! velocity errors are in the body-centered Cartesian axes of the
//! regime. A geodetic model converts its value into these axes with the
//! local NED axes of [`LocalLevel`], so no model has a tangent plane and
//! no model has a limit at a pole. The models read the regime of the
//! core that they update, so a replay after a frame change uses the
//! frame of the replayed core.

use nalgebra::{Const, DimMin, Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use navigate_geodesy::BodyEllipsoid;

use super::config::InertialConfig;
use super::estimate::{InertialCore, Linearized, Mode};
use super::local_level::LocalLevel;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;

mod attitude;
mod cartesian;
pub(crate) mod line_of_sight;
mod magnetic;
pub(crate) mod radiometric;
mod surface;
mod terrain;

#[cfg(test)]
pub(crate) use terrain::terrain_clearance;

/// Updates `core` with `obs` at the state time of `core` and returns the
/// chi-square statistic. On an error, `core` is not changed.
pub(crate) fn apply_observation<const N: usize>(
    core: &mut InertialCore<N>,
    obs: &Observation,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let kind = obs.value.kind();
    if !core.regime.supports(kind) {
        return Err(RejectionReason::UnsupportedMeasurement { kind });
    }
    let Some(declared) = &obs.correlated else {
        return apply_white(core, obs, config, mode);
    };
    let mut candidate = *core;
    let pending = match candidate.prepare_correlated(&obs.stamp, &obs.value, declared) {
        Ok(pending) => pending,
        // A delayed fix of another source can take a block before this
        // fix in the replay, as it would have in order. The fix then has
        // no place, as it would have had none in order, and the replay
        // goes on without it.
        Err(
            RejectionReason::CorrelatedErrorRepeated { .. }
            | RejectionReason::CorrelatedSourceUntracked
            | RejectionReason::CorrelatedProcessRetired,
        ) if mode == Mode::Replay => return Ok(0.0),
        Err(reason) => return Err(reason),
    };
    candidate.pending = Some(pending.plan);
    let chi2 = apply_white(&mut candidate, obs, config, mode)?;
    candidate.pending = None;
    candidate.correlated.commit(&pending);
    *core = candidate;
    Ok(chi2)
}

/// Updates `core` with the model of `obs`, as [`apply_observation`]
/// does after the correlated error of `obs` is prepared.
fn apply_white<const N: usize>(
    core: &mut InertialCore<N>,
    obs: &Observation,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let fusion = &config.fusion;
    match &obs.value {
        ObservationValue::CartesianPositionFix { .. }
        | ObservationValue::CartesianVelocityFix { .. }
        | ObservationValue::CartesianRange { .. } => {
            cartesian::apply(core, &obs.value, config, mode)
        }
        ObservationValue::AttitudeFix {
            attitude,
            covariance,
        } => {
            let to_reference = core
                .regime
                .surface
                .map_or_else(UnitQuaternion::identity, |ellipsoid| {
                    vehicle_level(core, ellipsoid).bf_to_ned_quaternion()
                });
            let model = attitude::attitude_fix(core, &to_reference, attitude, covariance);
            core.update(&model, fusion.innovation_gate_chi2, mode)
        }
        ObservationValue::Pseudorange { .. }
        | ObservationValue::PseudorangeRate { .. }
        | ObservationValue::TwoWayRange { .. }
        | ObservationValue::TwoWayRangeRate { .. } => {
            radiometric::apply(core, &obs.value, config, mode)
        }
        ObservationValue::LineOfSight { .. } => {
            line_of_sight::apply(core, &obs.value, config, mode)
        }
        _ => apply_geodetic(core, obs, config, mode),
    }
}

/// Applies a model that needs the reference ellipsoid of the regime.
fn apply_geodetic<const N: usize>(
    core: &mut InertialCore<N>,
    obs: &Observation,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let unsupported = RejectionReason::UnsupportedMeasurement {
        kind: obs.value.kind(),
    };
    let Some(ellipsoid) = core.regime.surface else {
        return Err(unsupported);
    };
    let fusion = &config.fusion;
    match &obs.value {
        ObservationValue::PositionFix { .. }
        | ObservationValue::VelocityFix { .. }
        | ObservationValue::Range { .. } => {
            surface::apply_vector_or_range(core, ellipsoid, obs, config, mode)
        }
        ObservationValue::Altitude {
            altitude_m,
            variance_m2,
        } => {
            let model =
                surface::altitude(core, ellipsoid, *altitude_m, *variance_m2, obs.composition);
            let up = vehicle_level(core, ellipsoid).up();
            core.update_vertical(&model, &up, fusion.scalar_gate_chi2, mode)
        }
        ObservationValue::TerrainClearance { .. } => {
            let model = terrain::terrain_clearance(core, ellipsoid, &obs.value)?;
            core.update(&model, fusion.scalar_gate_chi2, mode)
        }
        ObservationValue::MagneticAnomaly { .. } => {
            magnetic::apply(core, ellipsoid, obs, config, mode)
        }
        ObservationValue::VisualPose { .. } => {
            let model = attitude::visual_pose(core, ellipsoid, &obs.value)?;
            core.update(&model, fusion.pose_gate_chi2, mode)
        }
        _ => Err(unsupported),
    }
}

/// The geodetic position and the local NED axes of the nominal state.
pub(crate) fn vehicle_level<const N: usize>(
    core: &InertialCore<N>,
    ellipsoid: BodyEllipsoid,
) -> LocalLevel {
    LocalLevel::at_body_fixed(&core.nominal.p, ellipsoid)
}

/// A scalar update with the row `h` and the variance `variance`.
fn scalar<const N: usize>(residual: f64, h: SMatrix<f64, 1, N>, variance: f64) -> Linearized<1, N> {
    Linearized {
        residual: SVector::<f64, 1>::new(residual),
        h,
        r: SMatrix::<f64, 1, 1>::new(variance),
    }
}

/// A direct update of the three error components at `offset`.
fn block<const N: usize>(
    residual: Vector3<f64>,
    offset: usize,
    r: Matrix3<f64>,
) -> Linearized<3, N> {
    let mut h = SMatrix::<f64, 3, N>::zeros();
    h.fixed_view_mut::<3, 3>(0, offset)
        .copy_from(&Matrix3::identity());
    Linearized { residual, h, r }
}

/// The range row: the unit line of sight on the position error.
fn range<const N: usize>(
    p: &Vector3<f64>,
    target: &Vector3<f64>,
    range_m: f64,
    variance_m2: f64,
) -> Result<Linearized<1, N>, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
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
pub(crate) const MIN_RANGE_M: f64 = 1.0;
