//! Geodetic position, velocity, range, and altitude in the body-fixed
//! frame of a reference ellipsoid (ADR-0017).

use nalgebra::{SMatrix, Vector3};
use navigate_contract::{SensorClass, SourceComposition};
use navigate_geodesy::BodyEllipsoid;

use super::{block, range, scalar, vehicle_level};
use crate::inertial::config::InertialConfig;
use crate::inertial::estimate::{InertialCore, Linearized, Mode};
use crate::inertial::local_level::{LocalLevel, body_fixed};
use crate::inertial::nominal::{ALT_BIAS, N, POS, VEL};
use crate::kalman::cov3_to_matrix;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;

/// Applies a position fix, a velocity fix, or a range.
///
/// - A position fix converts into the body-fixed frame exactly, so the
///   residual is `r(fix) − p` with `H = I`. Its covariance rotates from
///   the local NED axes at the fix into the body-fixed axes.
/// - A velocity fix is in the local NED axes at the vehicle. It rotates
///   into the body-fixed axes with the axes at the nominal position. The
///   model does not include the change of the local axes with the
///   position error, because that term divides by `cos φ`. Close to a
///   pole a small position error turns the local axes, so a NED velocity
///   has little value there.
/// - A range converts the station into the body-fixed frame.
pub(super) fn apply_vector_or_range(
    core: &mut InertialCore,
    ellipsoid: BodyEllipsoid,
    obs: &Observation,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason> {
    let vector_gate = config.fusion.innovation_gate_chi2;
    match &obs.value {
        ObservationValue::PositionFix {
            position,
            covariance,
        } => {
            let z = body_fixed(position, ellipsoid);
            let r = LocalLevel::at(*position).ned_to_bf_cov(&cov3_to_matrix(covariance));
            let model = block(z - core.nominal.p, POS, r);
            core.update(&model, vector_gate, mode)
        }
        ObservationValue::VelocityFix {
            velocity,
            covariance,
        } => {
            let local = vehicle_level(core, ellipsoid);
            let v_ned = Vector3::new(velocity.north_mps, velocity.east_mps, velocity.down_mps);
            let z = local.ned_to_bf * v_ned;
            let r = local.ned_to_bf_cov(&cov3_to_matrix(covariance));
            let model = block(z - core.nominal.v, VEL, r);
            core.update(&model, vector_gate, mode)
        }
        ObservationValue::Range {
            station,
            range_m,
            variance_m2,
        } => {
            let target = body_fixed(station, ellipsoid);
            let model = range(&core.nominal.p, &target, *range_m, *variance_m2)?;
            core.update(&model, config.fusion.scalar_gate_chi2, mode)
        }
        _ => Err(RejectionReason::UnsupportedMeasurement {
            kind: obs.value.kind(),
        }),
    }
}

/// Ellipsoidal altitude: `h(x) = h_ellipsoid(p) + b_alt` for a
/// barometric source, and `h_ellipsoid(p)` for any other source. The
/// position row is the outward normal of the ellipsoid,
/// `[cos φ·cos λ, cos φ·sin λ, sin φ]`, which is the exact gradient of
/// the ellipsoidal height.
pub(super) fn altitude(
    core: &InertialCore,
    ellipsoid: BodyEllipsoid,
    altitude_m: f64,
    variance_m2: f64,
    composition: SourceComposition,
) -> Linearized<1> {
    let local = vehicle_level(core, ellipsoid);
    let barometric = composition.contains(SensorClass::Barometric);
    let bias = if barometric { core.nominal.b_alt } else { 0.0 };
    let mut h = SMatrix::<f64, 1, N>::zeros();
    h.fixed_view_mut::<1, 3>(0, POS)
        .copy_from(&local.up().transpose());
    if barometric {
        h[(0, ALT_BIAS)] = 1.0;
    }
    scalar(
        altitude_m - (local.position.altitude_m + bias),
        h,
        variance_m2,
    )
}
