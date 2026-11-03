//! Position, velocity, range, and altitude on a body surface.

use nalgebra::{SMatrix, Vector3};
use navigate_contract::{SensorClass, SourceComposition};

use super::{block, plane_to_local_at, range, scalar, vehicle_position};
use crate::environment::SurfaceFrame;
use crate::filter::cov3_to_matrix;
use crate::inertial::config::InertialConfig;
use crate::inertial::estimate::{InertialCore, Linearized, Mode};
use crate::inertial::nominal::{ALT_BIAS, N, POS, VEL};
use crate::navigation_filter::reanchor::vehicle_to_plane;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;

/// Applies a position fix, a velocity fix, or a range. The position and
/// velocity covariances arrive in the local NED axes at the fix and at
/// the vehicle. The model rotates them into the plane axes.
pub(super) fn apply_vector_or_range(
    core: &mut InertialCore,
    surface: &SurfaceFrame,
    obs: &Observation,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason> {
    let plane = &surface.plane;
    let vector_gate = config.fusion.innovation_gate_chi2;
    match &obs.value {
        ObservationValue::PositionFix {
            position,
            covariance,
        } => {
            let offset = plane.to_ned(position);
            let z = Vector3::new(offset.north_m, offset.east_m, offset.down_m);
            let (_, r) = vehicle_to_plane(plane, *position, z, cov3_to_matrix(covariance));
            let model = block(z - core.nominal.p, POS, r);
            core.update(&model, vector_gate, mode)
        }
        ObservationValue::VelocityFix {
            velocity,
            covariance,
        } => {
            let v = Vector3::new(velocity.north_mps, velocity.east_mps, velocity.down_mps);
            let vehicle = vehicle_position(core, surface);
            let (z, r) = vehicle_to_plane(plane, vehicle, v, cov3_to_matrix(covariance));
            let model = block(z - core.nominal.v, VEL, r);
            core.update(&model, vector_gate, mode)
        }
        ObservationValue::Range {
            station,
            range_m,
            variance_m2,
        } => {
            let offset = plane.to_ned(station);
            let target = Vector3::new(offset.north_m, offset.east_m, offset.down_m);
            let model = range(&core.nominal.p, &target, *range_m, *variance_m2)?;
            core.update(&model, config.fusion.scalar_gate_chi2, mode)
        }
        _ => Err(RejectionReason::UnsupportedMeasurement {
            kind: obs.value.kind(),
        }),
    }
}

/// Ellipsoidal altitude. A barometric altitude also measures the
/// barometric bias: `h(x) = h_ellipsoid(p) + b_alt`. Any other source
/// does not touch the bias.
pub(super) fn altitude(
    core: &InertialCore,
    surface: &SurfaceFrame,
    altitude_m: f64,
    variance_m2: f64,
    composition: SourceComposition,
) -> Result<Linearized<1>, RejectionReason> {
    let vehicle = vehicle_position(core, surface);
    let plane_to_local = plane_to_local_at(surface, vehicle)?;
    let up_plane = plane_to_local.transpose() * Vector3::new(0.0, 0.0, -1.0);
    let barometric = composition.contains(SensorClass::Barometric);
    let bias = if barometric { core.nominal.b_alt } else { 0.0 };
    let mut h = SMatrix::<f64, 1, N>::zeros();
    h.fixed_view_mut::<1, 3>(0, POS)
        .copy_from(&up_plane.transpose());
    if barometric {
        h[(0, ALT_BIAS)] = 1.0;
    }
    Ok(scalar(
        altitude_m - (vehicle.altitude_m + bias),
        h,
        variance_m2,
    ))
}
