//! A unit direction measured in the body axes (ADR-0018).
//!
//! The predicted direction in the body axes is `û_b = Cᵀ·û_f`, with `C`
//! the rotation from the body axes to the frame axes and `û_f` the
//! reference direction in the frame axes. The attitude error is a right
//! perturbation, so `û_b(δθ) = û_b + [û_b×]·δθ`. A direction has two
//! degrees of freedom. The update uses an orthonormal basis `B` of the
//! plane that is orthogonal to `û_b`, so its covariance `σ²·I₂` has full
//! rank. The residual is `Bᵀ·z` for the measured direction `z`, and the
//! rows are
//!
//! - `H_θ = Bᵀ·[û_b×]` for each reference,
//! - `H_p = −Bᵀ·Cᵀ/|d|` for a target at the distance `|d|`, because the
//!   direction to the target turns when the vehicle moves across it.

use nalgebra::{Const, DimMin, Matrix2, SMatrix, Vector3};

use super::MIN_RANGE_M;
use crate::inertial::config::InertialConfig;
use crate::inertial::estimate::{InertialCore, Linearized, Mode};
use crate::inertial::nominal::{ATT, POS};
use crate::observation::{LineOfSightReference, ObservationValue};
use crate::rejection::RejectionReason;

/// An orthonormal basis of the plane that is orthogonal to the unit
/// vector `u`, as the two columns of a 3×2 matrix.
pub(crate) fn tangent_basis(u: &Vector3<f64>) -> SMatrix<f64, 3, 2> {
    let (x, y, z) = (u.x.abs(), u.y.abs(), u.z.abs());
    let axis = if x <= y && x <= z {
        Vector3::x()
    } else if y <= z {
        Vector3::y()
    } else {
        Vector3::z()
    };
    let first = u.cross(&axis).normalize();
    let second = u.cross(&first);
    SMatrix::<f64, 3, 2>::from_columns(&[first, second])
}

/// Applies a line of sight to `core`.
pub(super) fn apply<const N: usize>(
    core: &mut InertialCore<N>,
    value: &ObservationValue,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let (model, cosine) = line_of_sight(core, value)?;
    let gate = config.fusion.line_of_sight_gate_chi2;
    if mode == Mode::Gated && cosine <= 0.0 {
        // The measured direction is at least 90° from the prediction, so
        // the projection does not show the full error.
        return Err(RejectionReason::InnovationGate {
            chi2: f64::INFINITY,
            threshold: gate,
        });
    }
    core.update(&model, gate, mode)
}

/// The linearized line of sight at the nominal state of `core`, and the
/// cosine of the angle between the measured and the predicted
/// directions.
pub(crate) fn line_of_sight<const N: usize>(
    core: &InertialCore<N>,
    value: &ObservationValue,
) -> Result<(Linearized<2, N>, f64), RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let ObservationValue::LineOfSight {
        measured_body_unit,
        reference,
        angular_variance_rad2,
    } = value
    else {
        return Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() });
    };
    let c = core.nominal.c_b_to_f();
    let (u_frame, target_distance) = match reference {
        LineOfSightReference::InertialDirection {
            direction_in_frame_unit,
        } => (Vector3::from(*direction_in_frame_unit).normalize(), None),
        LineOfSightReference::TargetPosition {
            position_in_frame_m,
        } => {
            let d = Vector3::from(*position_in_frame_m) - core.nominal.p;
            let distance = d.norm();
            if distance < MIN_RANGE_M {
                return Err(RejectionReason::ImplausibleCovariance);
            }
            (d / distance, Some(distance))
        }
    };
    let u_body = c.transpose() * u_frame;
    let basis = tangent_basis(&u_body);
    let measured = Vector3::from(*measured_body_unit).normalize();
    let mut h = SMatrix::<f64, 2, N>::zeros();
    h.fixed_view_mut::<2, 3>(0, ATT)
        .copy_from(&(basis.transpose() * u_body.cross_matrix()));
    if let Some(distance) = target_distance {
        let rows: SMatrix<f64, 2, 3> = -(basis.transpose() * c.transpose()) / distance;
        h.fixed_view_mut::<2, 3>(0, POS).copy_from(&rows);
    }
    let model = Linearized {
        residual: basis.transpose() * measured,
        h,
        r: Matrix2::identity() * *angular_variance_rad2,
    };
    Ok((model, measured.dot(&u_body)))
}
