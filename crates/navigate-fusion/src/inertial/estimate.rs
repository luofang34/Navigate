//! The initialized state of the inertial filter and its propagation
//! (ADR-0014).

use nalgebra::{SMatrix, SVector};
use navigate_contract::{MonotonicNanos, SourceId};

use super::config::InertialConfig;
use super::mechanization::{ImuRates, transition};
use super::nominal::{InertialNominal, Mat17, N, Vec17, reset_covariance};
use super::regime::Regime;
use crate::admission::seconds;
use crate::filter::{KalmanState, MeasurementModel};
use crate::rejection::RejectionReason;

/// Longest propagation step in seconds. A longer interval is split into
/// equal steps, so the midpoint integration keeps its accuracy.
const MAX_STEP_S: f64 = 1.0;

/// The map that a magnetic bias belongs to: the source and the map
/// revision.
pub(crate) type MagneticKey = (SourceId, Option<u64>);

/// The state of an initialized filter at one time. The history keeps a
/// copy of it before each event, so a replay starts from it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InertialCore {
    pub(crate) regime: Regime,
    pub(crate) nominal: InertialNominal,
    pub(crate) p: Mat17,
    pub(crate) state_at: MonotonicNanos,
    /// The rates of the most recent IMU sample. A propagation without a
    /// covering sample holds them.
    pub(crate) last_rates: Option<ImuRates>,
    /// The map of the current magnetic bias estimate.
    pub(crate) mag_key: Option<MagneticKey>,
}

/// How much of the admission an update runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// A new event: the model checks and the chi-square gate run.
    Gated,
    /// A replay of an event that the filter admitted before.
    Replay,
}

impl InertialCore {
    /// Propagates the state to `at` with constant `rates`. A time at or
    /// before the state time changes nothing.
    pub(crate) fn propagate_to(
        &mut self,
        at: MonotonicNanos,
        rates: &ImuRates,
        config: &InertialConfig,
    ) -> Result<(), RejectionReason> {
        let Some(elapsed) = at.elapsed_since(self.state_at) else {
            return Ok(());
        };
        let total_s = seconds(elapsed);
        if total_s <= 0.0 {
            return Ok(());
        }
        let steps = (total_s / MAX_STEP_S).ceil().max(1.0);
        let dt_s = total_s / steps;
        let mut nominal = self.nominal;
        let mut p = self.p;
        for _ in 0..(steps as u64) {
            let (next, phi, q_d) = transition(&nominal, rates, dt_s, &self.regime, config)
                .ok_or(RejectionReason::ImplausibleCovariance)?;
            let propagated = phi * p * phi.transpose() + q_d;
            p = (propagated + propagated.transpose()) * 0.5;
            nominal = next;
        }
        self.nominal = nominal;
        self.p = p;
        self.state_at = at;
        Ok(())
    }

    /// Propagates to `at` with `rates`, or with the most recent IMU
    /// rates when `rates` is `None`. A time after the state time without
    /// any rates is refused with [`RejectionReason::NoInertialData`].
    pub(crate) fn propagate_with(
        &mut self,
        at: MonotonicNanos,
        rates: Option<ImuRates>,
        config: &InertialConfig,
    ) -> Result<(), RejectionReason> {
        if at <= self.state_at {
            return Ok(());
        }
        let rates = rates
            .or(self.last_rates)
            .ok_or(RejectionReason::NoInertialData)?;
        self.propagate_to(at, &rates, config)
    }

    /// Applies one linearized measurement of the error state. The error
    /// state is zero before the update. The update injects the estimate
    /// into the nominal state and resets the covariance. On an error the
    /// core does not change.
    pub(crate) fn update<const M: usize>(
        &mut self,
        model: &Linearized<M>,
        gate_chi2: f64,
        mode: Mode,
    ) -> Result<f64, RejectionReason> {
        let state = KalmanState::<N> {
            x: Vec17::zeros(),
            p: self.p,
        };
        let prepared = state
            .prepare_update(model)
            .ok_or(RejectionReason::ImplausibleCovariance)?;
        if mode == Mode::Gated && (prepared.chi2.is_nan() || prepared.chi2 > gate_chi2) {
            return Err(RejectionReason::InnovationGate {
                chi2: prepared.chi2,
                threshold: gate_chi2,
            });
        }
        let updated = state.apply_update(&prepared);
        self.nominal = self.nominal.inject(&updated.x);
        self.p = reset_covariance(&updated.p, &updated.x);
        Ok(prepared.chi2)
    }
}

/// A measurement linearized at the nominal state: the residual
/// `y = z − h(x_nom)`, the Jacobian `H` of the error state, and the noise
/// `R`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Linearized<const M: usize> {
    pub(crate) residual: SVector<f64, M>,
    pub(crate) h: SMatrix<f64, M, N>,
    pub(crate) r: SMatrix<f64, M, M>,
}

impl<const M: usize> MeasurementModel<M, N> for Linearized<M> {
    fn innovation(&self, x: &Vec17) -> SVector<f64, M> {
        self.residual - self.h * x
    }

    fn jacobian(&self, _x: &Vec17) -> Option<SMatrix<f64, M, N>> {
        Some(self.h)
    }

    fn noise(&self) -> SMatrix<f64, M, M> {
        self.r
    }
}
