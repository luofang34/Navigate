//! Position, velocity, and range in the Cartesian frame of the regime
//! (ADR-0014, ADR-0017). Each regime has these models, because the state
//! is always in that frame.

use nalgebra::Vector3;
use nalgebra::{Const, DimMin};

use super::{block, range};
use crate::inertial::config::InertialConfig;
use crate::inertial::estimate::{InertialCore, Mode};
use crate::inertial::nominal::{POS, VEL};
use crate::kalman::cov3_to_matrix;
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

/// Applies a Cartesian position fix, velocity fix, or range. The values
/// and covariances are in the frame axes.
pub(super) fn apply<const N: usize>(
    core: &mut InertialCore<N>,
    value: &ObservationValue,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let vector_gate = config.fusion.innovation_gate_chi2;
    match value {
        ObservationValue::CartesianPositionFix {
            position,
            covariance,
        } => {
            let z = Vector3::new(position.x_m, position.y_m, position.z_m);
            let model = block(z - core.nominal.p, POS, cov3_to_matrix(covariance));
            core.update(&model, vector_gate, mode)
        }
        ObservationValue::CartesianVelocityFix {
            velocity,
            covariance,
        } => {
            let z = Vector3::new(velocity.vx_mps, velocity.vy_mps, velocity.vz_mps);
            let model = block(z - core.nominal.v, VEL, cov3_to_matrix(covariance));
            core.update(&model, vector_gate, mode)
        }
        ObservationValue::CartesianRange {
            beacon,
            range_m,
            variance_m2,
        } => {
            let target = Vector3::new(beacon.x_m, beacon.y_m, beacon.z_m);
            let model = range(&core.nominal.p, &target, *range_m, *variance_m2)?;
            core.update(&model, config.fusion.scalar_gate_chi2, mode)
        }
        _ => Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() }),
    }
}
