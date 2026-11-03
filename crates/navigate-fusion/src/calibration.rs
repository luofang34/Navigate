//! Online Tolles–Lawson calibration of the aircraft magnetic interference
//! (ADR-0014).
//!
//! The scalar interference of the aircraft is `A(B, Ḃ)·c`. The row `A`
//! has 18 terms: 3 permanent terms `cᵢ`, 6 induced terms `|B|·cᵢ·cⱼ` for
//! `i ≤ j`, and 9 eddy-current terms `cᵢ·Ḃⱼ`. Here `cᵢ = Bᵢ/|B|` are the
//! direction cosines of the field in the body axes, `B` is in tesla, and
//! `Ḃ` is in tesla per second. A recursive Joseph-form estimator learns
//! the 18 coefficients and a first-order Gauss–Markov residual bias.
//!
//! The coefficients are observable only when the attitude changes the
//! field in the body axes. In straight and level flight the change of
//! the measured field comes from the crustal anomaly. The estimator then
//! freezes the coefficients, so that it cannot absorb the anomaly that
//! the navigation needs.

use nalgebra::{SMatrix, SVector, Vector3};
use thiserror::Error;

use crate::filter::FogmBias;

#[cfg(test)]
mod tests;

/// Number of Tolles–Lawson terms.
pub const TOLLES_LAWSON_TERMS: usize = 18;
/// Dimension of the estimator state: the terms and the residual bias.
const STATE: usize = TOLLES_LAWSON_TERMS + 1;
/// Index of the residual bias in the state.
const BIAS: usize = TOLLES_LAWSON_TERMS;
/// Below this field magnitude in tesla the direction cosines are not
/// defined.
const MIN_FIELD_T: f64 = 1e-9;

type State = SVector<f64, STATE>;
type Covariance = SMatrix<f64, STATE, STATE>;

/// The Tolles–Lawson row for the body-axis field `b_body_t` and its rate
/// `b_dot_body_t_per_s`. `None` for a non-finite input or a field below
/// 1 nT.
#[must_use]
pub fn tolles_lawson_basis(
    b_body_t: [f64; 3],
    b_dot_body_t_per_s: [f64; 3],
) -> Option<[f64; TOLLES_LAWSON_TERMS]> {
    let b = Vector3::from(b_body_t);
    let b_dot = Vector3::from(b_dot_body_t_per_s);
    let magnitude = b.norm();
    if !(magnitude.is_finite() && magnitude >= MIN_FIELD_T && b_dot.iter().all(|v| v.is_finite())) {
        return None;
    }
    let cosines = b / magnitude;
    let mut row = [0.0; TOLLES_LAWSON_TERMS];
    let mut next = 0_usize;
    let mut push = |value: f64| {
        if let Some(slot) = row.get_mut(next) {
            *slot = value;
        }
        next = next.wrapping_add(1);
    };
    for c_i in cosines.iter() {
        push(*c_i);
    }
    for (i, c_i) in cosines.iter().enumerate() {
        for c_j in cosines.iter().skip(i) {
            push(magnitude * c_i * c_j);
        }
    }
    for c_i in cosines.iter() {
        for b_dot_j in b_dot.iter() {
            push(c_i * b_dot_j);
        }
    }
    Some(row)
}

/// A refusal of the calibrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum TlError {
    /// A configuration value is not finite and positive.
    #[error("Tolles–Lawson calibrator configuration is not valid")]
    InvalidConfig,
    /// An input is not finite, or the field is below 1 nT.
    #[error("Tolles–Lawson input has a non-finite value or no field direction")]
    NonFiniteValue,
    /// The innovation variance has no positive value.
    #[error("Tolles–Lawson innovation variance is not positive")]
    ImplausibleCovariance,
}

/// Configuration of an [`OnlineTlCalibrator`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TlCalibratorConfig {
    /// Prior 1-sigma of each permanent coefficient in tesla.
    pub permanent_sigma_t: f64,
    /// Prior 1-sigma of each induced coefficient. It has no unit.
    pub induced_sigma: f64,
    /// Prior 1-sigma of each eddy-current coefficient in seconds.
    pub eddy_sigma_s: f64,
    /// Variance of the scalar measurement in tesla².
    pub measurement_variance_t2: f64,
    /// Model of the residual bias in tesla.
    pub residual_bias: FogmBias,
    /// Lowest `|Ḃ|` in tesla per second at which the coefficients learn.
    pub min_b_dot_norm_t_per_s: f64,
}

impl Default for TlCalibratorConfig {
    /// Survey-aircraft scales: 1 µT permanent, 1e-2 induced, 1 s eddy,
    /// 0.1 nT noise, a 3 nT residual bias with a correlation time of
    /// 600 s, and an excitation floor of 1e-7 T/s.
    fn default() -> Self {
        Self {
            permanent_sigma_t: 1e-6,
            induced_sigma: 1e-2,
            eddy_sigma_s: 1.0,
            measurement_variance_t2: 1e-20,
            residual_bias: FogmBias::new(600.0, 3e-9),
            min_b_dot_norm_t_per_s: 1e-7,
        }
    }
}

impl TlCalibratorConfig {
    fn is_valid(&self) -> bool {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        positive(self.permanent_sigma_t)
            && positive(self.induced_sigma)
            && positive(self.eddy_sigma_s)
            && positive(self.measurement_variance_t2)
            && self.min_b_dot_norm_t_per_s.is_finite()
            && self.min_b_dot_norm_t_per_s >= 0.0
    }
}

/// The result of one calibrator sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TlUpdate {
    /// The excitation was enough. The coefficients and the bias learned.
    Updated,
    /// The excitation was below the floor. Only the time update ran.
    Frozen,
}

/// Recursive Tolles–Lawson estimator with an excitation gate.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineTlCalibrator {
    config: TlCalibratorConfig,
    x: State,
    p: Covariance,
}

impl OnlineTlCalibrator {
    /// Starts with zero coefficients and the prior variances.
    ///
    /// # Errors
    ///
    /// [`TlError::InvalidConfig`].
    pub fn new(config: TlCalibratorConfig) -> Result<Self, TlError> {
        if !config.is_valid() {
            return Err(TlError::InvalidConfig);
        }
        let mut p = Covariance::zeros();
        for index in 0..TOLLES_LAWSON_TERMS {
            let sigma = match index {
                0..=2 => config.permanent_sigma_t,
                3..=8 => config.induced_sigma,
                _ => config.eddy_sigma_s,
            };
            p[(index, index)] = sigma * sigma;
        }
        p[(BIAS, BIAS)] = config.residual_bias.stationary_variance();
        Ok(Self {
            config,
            x: State::zeros(),
            p,
        })
    }

    /// Takes one sample after `dt_s` seconds. `residual_t` is the scalar
    /// magnetometer value minus the reference field, such as the core
    /// field and the map anomaly during a calibration flight. The bias
    /// always runs its time update. The measurement update runs only when
    /// `|Ḃ|` is at or above the floor.
    ///
    /// # Errors
    ///
    /// [`TlError::NonFiniteValue`] and [`TlError::ImplausibleCovariance`].
    /// On an error nothing changes.
    pub fn observe(
        &mut self,
        b_body_t: [f64; 3],
        b_dot_body_t_per_s: [f64; 3],
        residual_t: f64,
        dt_s: f64,
    ) -> Result<TlUpdate, TlError> {
        let row =
            tolles_lawson_basis(b_body_t, b_dot_body_t_per_s).ok_or(TlError::NonFiniteValue)?;
        if !(residual_t.is_finite() && dt_s.is_finite() && dt_s >= 0.0) {
            return Err(TlError::NonFiniteValue);
        }
        let mut x = self.x;
        let mut p = self.p;
        let (phi, q) = self.config.residual_bias.propagate(1.0, dt_s);
        x[BIAS] *= phi;
        p.row_mut(BIAS).scale_mut(phi);
        p.column_mut(BIAS).scale_mut(phi);
        p[(BIAS, BIAS)] += q;
        let excited =
            Vector3::from(b_dot_body_t_per_s).norm() >= self.config.min_b_dot_norm_t_per_s;
        let outcome = if excited {
            let mut h = SMatrix::<f64, 1, STATE>::zeros();
            for (slot, value) in h.iter_mut().zip(row) {
                *slot = value;
            }
            h[(0, BIAS)] = 1.0;
            let r = self.config.measurement_variance_t2;
            let s = (h * p * h.transpose())[(0, 0)] + r;
            if !(s.is_finite() && s > 0.0) {
                return Err(TlError::ImplausibleCovariance);
            }
            let gain = p * h.transpose() / s;
            x += gain * (residual_t - (h * x)[(0, 0)]);
            let joseph = Covariance::identity() - gain * h;
            p = joseph * p * joseph.transpose() + gain * gain.transpose() * r;
            p = (p + p.transpose()) * 0.5;
            TlUpdate::Updated
        } else {
            TlUpdate::Frozen
        };
        self.x = x;
        self.p = p;
        Ok(outcome)
    }

    /// The interference `A·c` of the learned coefficients. `None` when
    /// the basis is not defined.
    #[must_use]
    pub fn interference_t(&self, b_body_t: [f64; 3], b_dot_body_t_per_s: [f64; 3]) -> Option<f64> {
        let row = tolles_lawson_basis(b_body_t, b_dot_body_t_per_s)?;
        Some(row.iter().zip(self.x.iter()).map(|(a, c)| a * c).sum())
    }

    /// The scalar magnetometer value `measured_t` minus the learned
    /// interference. The residual bias stays in the result, because it
    /// is not an aircraft term. `None` when the basis is not defined.
    #[must_use]
    pub fn compensate(
        &self,
        b_body_t: [f64; 3],
        b_dot_body_t_per_s: [f64; 3],
        measured_t: f64,
    ) -> Option<f64> {
        Some(measured_t - self.interference_t(b_body_t, b_dot_body_t_per_s)?)
    }

    /// The learned coefficients in the order of [`tolles_lawson_basis`].
    #[must_use]
    pub fn coefficients(&self) -> [f64; TOLLES_LAWSON_TERMS] {
        let mut coefficients = [0.0; TOLLES_LAWSON_TERMS];
        for (slot, value) in coefficients.iter_mut().zip(self.x.iter()) {
            *slot = *value;
        }
        coefficients
    }

    /// The estimated residual bias in tesla.
    #[must_use]
    pub fn residual_bias_t(&self) -> f64 {
        self.x[BIAS]
    }
}
