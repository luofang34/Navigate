//! The initialized state of the inertial filter and its propagation
//! (ADR-0014, ADR-0017, ADR-0018).

use nalgebra::{SMatrix, SVector};
use navigate_contract::{MonotonicNanos, SourceId};

use super::coast::coast_transition;
use super::config::InertialConfig;
use super::imu::ImuIncrement;
use super::mechanization::{ImuRates, increment_transition, transition};
use super::nominal::{ErrorMat, ErrorVec, InertialNominal, N, reset_covariance};
use super::propagation::{ForceModelCoast, PropagationMode};
use super::regime::Regime;
use crate::admission::seconds;
use crate::float_math::{ceil, round};
use crate::kalman::{KalmanState, MeasurementModel};
use crate::rejection::RejectionReason;

/// Longest propagation step in seconds. A longer interval is split into
/// equal steps, so the midpoint integration keeps its accuracy.
const MAX_STEP_S: f64 = 1.0;

/// An elapsed time within this number of nanoseconds of the interval of
/// an increment covers the full interval.
const INCREMENT_MATCH_NS: u64 = 1_000;

/// The map that a magnetic bias belongs to: the source and the map
/// revision.
pub(crate) type MagneticKey = (SourceId, Option<u64>);

/// The state of an initialized filter at one time. The history keeps a
/// copy of it before each event, so a replay starts from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InertialCore {
    pub(crate) regime: Regime,
    pub(crate) nominal: InertialNominal,
    pub(crate) p: ErrorMat,
    pub(crate) state_at: MonotonicNanos,
    /// The rates of the most recent IMU sample. A propagation without a
    /// covering sample holds them.
    pub(crate) last_rates: Option<ImuRates>,
    /// The map of the current magnetic bias estimate.
    pub(crate) mag_key: Option<MagneticKey>,
    /// How the state propagates.
    pub(crate) mode: PropagationMode,
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
        let steps = ceil(total_s / MAX_STEP_S).max(1.0);
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

    /// Propagates the state to `at` with `increment`. When the elapsed
    /// time is the interval of the increment, the step uses the full
    /// increment. Otherwise a part of the interval is already applied, or
    /// a gap comes before the interval, and the propagation uses the mean
    /// rates of the increment. A time at or before the state time changes
    /// nothing.
    pub(crate) fn propagate_increment(
        &mut self,
        at: MonotonicNanos,
        increment: &ImuIncrement,
        config: &InertialConfig,
    ) -> Result<(), RejectionReason> {
        let Some(elapsed) = at.elapsed_since(self.state_at) else {
            return Ok(());
        };
        let interval_ns = round(increment.dt_s * 1e9) as u64;
        if elapsed.as_nanos().abs_diff(interval_ns) > INCREMENT_MATCH_NS {
            return self.propagate_to(at, &ImuRates::mean_of(increment), config);
        }
        let dt_s = seconds(elapsed);
        let (next, phi, q_d) =
            increment_transition(&self.nominal, increment, dt_s, &self.regime, config)
                .ok_or(RejectionReason::ImplausibleCovariance)?;
        let propagated = phi * self.p * phi.transpose() + q_d;
        self.p = (propagated + propagated.transpose()) * 0.5;
        self.nominal = next;
        self.state_at = at;
        Ok(())
    }

    /// Propagates to `at` in the active mode. In the sensed-IMU mode the
    /// propagation uses `rates`, or the most recent IMU rates when
    /// `rates` is `None`, and a time after the state time without any
    /// rates is refused with [`RejectionReason::NoInertialData`]. In the
    /// force-model mode the propagation does not use `rates`.
    pub(crate) fn propagate_with(
        &mut self,
        at: MonotonicNanos,
        rates: Option<ImuRates>,
        config: &InertialConfig,
    ) -> Result<(), RejectionReason> {
        if at <= self.state_at {
            return Ok(());
        }
        if let PropagationMode::ForceModel(settings) = self.mode {
            return self.coast_to(at, &settings, config);
        }
        let rates = rates
            .or(self.last_rates)
            .ok_or(RejectionReason::NoInertialData)?;
        self.propagate_to(at, &rates, config)
    }

    /// Propagates the state to `at` with the force model of the
    /// environment, in equal steps of at most
    /// [`ForceModelCoast::max_rk4_step_s`].
    fn coast_to(
        &mut self,
        at: MonotonicNanos,
        settings: &ForceModelCoast,
        config: &InertialConfig,
    ) -> Result<(), RejectionReason> {
        let Some(elapsed) = at.elapsed_since(self.state_at) else {
            return Ok(());
        };
        let total_s = seconds(elapsed);
        if total_s <= 0.0 {
            return Ok(());
        }
        let model = self.regime.env.force_model();
        let steps = ceil(total_s / settings.max_rk4_step_s).max(1.0);
        let dt_s = total_s / steps;
        let mut nominal = self.nominal;
        let mut p = self.p;
        for _ in 0..(steps as u64) {
            let (next, phi, q_d) = coast_transition(&nominal, dt_s, &model, settings, config)
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
            x: ErrorVec::zeros(),
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
    fn innovation(&self, x: &ErrorVec) -> SVector<f64, M> {
        self.residual - self.h * x
    }

    fn jacobian(&self, _x: &ErrorVec) -> Option<SMatrix<f64, M, N>> {
        Some(self.h)
    }

    fn noise(&self) -> SMatrix<f64, M, M> {
        self.r
    }
}
