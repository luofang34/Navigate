//! How one fix uses its correlated error: the plan of the fix, prepared
//! against the blocks of the core, and applied to its measurement model.

use nalgebra::{Const, DimMin, Matrix3, SMatrix, Vector3};
use navigate_contract::ObservationStamp;

use super::super::estimate::{InertialCore, Linearized};
use super::super::local_level::LocalLevel;
use super::super::nominal::InertialNominal;
use super::{ErrorKind, Pending, Plan, Process};
use crate::observation::{CorrelatedError, ObservationValue};
use crate::rejection::RejectionReason;

impl Plan {
    /// `model` with the correlated error of this fix: for a bound block,
    /// the block columns of `H` copy the columns of the core states that
    /// the error adds to, and the residual loses the block mean; for an
    /// inflated fix, `R` grows by `factor·H_b·Σ·H_bᵀ`.
    pub(crate) fn apply<const M: usize, const N: usize>(
        &self,
        model: &Linearized<M, N>,
        nominal: &InertialNominal,
    ) -> Linearized<M, N> {
        let mut out = *model;
        match *self {
            Plan::Bound {
                block,
                offset,
                kind,
                ..
            } => {
                let h_base: SMatrix<f64, M, 3> =
                    model.h.fixed_view::<M, 3>(0, kind.base()).into_owned();
                let mean = nominal.correlated.get(block).copied().unwrap_or_default();
                out.h.fixed_view_mut::<M, 3>(0, offset).copy_from(&h_base);
                out.residual -= h_base * mean;
            }
            Plan::Inflated {
                kind,
                shape,
                factor,
                ..
            } => {
                let h_base: SMatrix<f64, M, 3> =
                    model.h.fixed_view::<M, 3>(0, kind.base()).into_owned();
                let added = h_base * shape * h_base.transpose() * factor;
                out.r += (added + added.transpose()) * 0.5;
            }
        }
        out
    }
}

/// The stationary covariance of `declared` in the axes of its block,
/// from the 1-sigma in the axes of the observed quantity (see
/// [`CorrelatedError`]).
fn shape(value: &ObservationValue, declared: &CorrelatedError, has_surface: bool) -> Matrix3<f64> {
    let [a, b, c] = declared.stationary_sigma;
    let diagonal = Matrix3::from_diagonal(&Vector3::new(a * a, b * b, c * c));
    match value {
        ObservationValue::PositionFix { position, .. } if has_surface => {
            LocalLevel::at(*position).ned_to_bf_cov(&diagonal)
        }
        _ => diagonal,
    }
}

impl<const N: usize> InertialCore<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    /// Prepares the fix `value` with the declared `declared` of the
    /// source of `stamp`: refuses a value without a correlated-error
    /// model and an implausible declaration, plans the fix, and starts
    /// a new block when the plan binds one.
    pub(crate) fn prepare_correlated(
        &mut self,
        stamp: &ObservationStamp,
        value: &ObservationValue,
        declared: &CorrelatedError,
    ) -> Result<Pending, RejectionReason> {
        let kind = ErrorKind::of(value)
            .ok_or(RejectionReason::CorrelationNotModelled { kind: value.kind() })?;
        if !declared.is_plausible() {
            return Err(RejectionReason::NonFiniteValue);
        }
        let process = Process {
            source: stamp.source,
            epoch: stamp.epoch,
            kind,
            declared: *declared,
        };
        let shape = shape(value, declared, self.regime.surface.is_some());
        let pending = self
            .correlated
            .plan(process, shape, self.state_at, &self.layout)?;
        if let Plan::Bound {
            block,
            offset,
            reset: true,
            ..
        } = pending.plan
        {
            self.p.rows_mut(offset, 3).fill(0.0);
            self.p.columns_mut(offset, 3).fill(0.0);
            self.p
                .fixed_view_mut::<3, 3>(offset, offset)
                .copy_from(&shape);
            if let Some(mean) = self.nominal.correlated.get_mut(block) {
                *mean = Vector3::zeros();
            }
        }
        Ok(pending)
    }
}
