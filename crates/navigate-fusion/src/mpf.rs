//! Marginalized particle filter for scalar field map matching
//! (ADR-0014).
//!
//! The filter follows Schön, Gustafsson, and Nordlund (2005). Each
//! particle holds a position `x_n` in the tangent plane, the conditional
//! mean `x_l = [v_n, v_e, v_d, b_alt, b_mag]` of the linear states, and a
//! log weight. The linear states have one shared covariance `P_l`,
//! because the linear models do not depend on the particle position.
//! The map enters at each particle without a linearization, so a map
//! with several matching positions gives several modes, not a wrong
//! gradient.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use navigate_contract::{
    ClockDomainId, FaultDetection, GeodeticPosition, IntegrityAssessment, MonotonicNanos,
    NavigationSolution, NedVelocity, Redundancy, SolutionQuality, SolutionStamp, SourceComposition,
    SourceEpoch, SymmetricCov3, WrappingSequence,
};
use navigate_geodesy::{BodyEllipsoid, LocalTangentPlane, NedOffset};
use thiserror::Error;

use crate::filter::{
    FogmBias, cov3_to_matrix, horizontal_1sigma_m, is_positive_definite, matrix_to_cov3,
};
use crate::navigation_filter::reanchor::plane_to_vehicle;

mod grid_map;
mod propagate;
mod rng;
mod update;

#[cfg(test)]
mod tests;

pub use grid_map::{BilinearGridMap, ScalarFieldMap};
use rng::SplitMix64;

/// Dimension of the linear state.
const L: usize = 5;
/// Index of the barometric bias in the linear state.
const ALT: usize = 3;
/// Index of the magnetic bias in the linear state.
const MAG: usize = 4;

type LinearState = SVector<f64, L>;
type LinearCov = SMatrix<f64, L, L>;

/// A refusal of the particle filter.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum MpfError {
    /// The configuration has no particle, a ratio outside `[0, 1]`, or a
    /// non-finite or negative noise or limit.
    #[error("particle filter configuration is not valid")]
    InvalidConfig,
    /// A value is not finite, or the origin is not a plausible position.
    #[error("particle filter input has a non-finite or out-of-range value")]
    NonFiniteValue,
    /// A covariance or a variance is not finite and positive definite.
    #[error("particle filter covariance is not finite and positive definite")]
    ImplausibleCovariance,
    /// No particle is on the map, so the measurement has no likelihood.
    #[error("no particle of {particle_count} is on the map")]
    NoParticleOnMap {
        /// Number of particles.
        particle_count: usize,
    },
    /// The grid dimensions, steps, or values are not valid.
    #[error("grid map of {rows}×{cols} is not valid")]
    InvalidMap {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },
}

/// Configuration of a [`MarginalizedParticleFilter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MpfConfig {
    /// Number of particles.
    pub particle_count: usize,
    /// The filter resamples when the effective sample size falls below
    /// this ratio of the particle count.
    pub resample_threshold_ratio: f64,
    /// White position noise density in m/√s on each axis.
    pub pos_process_sigma_m_sqrt_s: f64,
    /// White acceleration noise density in (m/s)/√s on each axis.
    pub vel_process_sigma_mps_sqrt_s: f64,
    /// Model of the barometric altitude bias in meters.
    pub baro_bias: FogmBias,
    /// Model of the scalar magnetic map bias in tesla.
    pub mag_bias: FogmBias,
    /// Seed of the deterministic random source.
    pub seed: u64,
    /// Largest horizontal 1-sigma of the particle cloud in meters for a
    /// usable solution. A larger spread, such as two separate modes, is
    /// not one position.
    pub unimodal_max_1sigma_m: f64,
}

impl Default for MpfConfig {
    fn default() -> Self {
        Self {
            particle_count: 2_000,
            resample_threshold_ratio: 0.5,
            pos_process_sigma_m_sqrt_s: 0.5,
            vel_process_sigma_mps_sqrt_s: 0.05,
            baro_bias: FogmBias::new(3_600.0, 3.0),
            mag_bias: FogmBias::new(600.0, 3e-9),
            seed: 0x5EED_5EED_5EED_5EED,
            unimodal_max_1sigma_m: 50.0,
        }
    }
}

impl MpfConfig {
    fn is_valid(&self) -> bool {
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        self.particle_count > 0
            && (0.0..=1.0).contains(&self.resample_threshold_ratio)
            && non_negative(self.pos_process_sigma_m_sqrt_s)
            && non_negative(self.vel_process_sigma_mps_sqrt_s)
            && self.unimodal_max_1sigma_m.is_finite()
            && self.unimodal_max_1sigma_m > 0.0
    }
}

/// The weighted estimate of the particle filter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct MpfEstimate {
    /// Mean position on the ellipsoid of the filter.
    pub position: GeodeticPosition,
    /// Mean velocity in the local NED axes at `position`.
    pub velocity: NedVelocity,
    /// Position covariance of the cloud in the local NED axes, in m².
    pub position_cov: SymmetricCov3,
    /// Velocity covariance in the local NED axes: the shared covariance
    /// plus the spread of the particle means, in (m/s)².
    pub velocity_cov: SymmetricCov3,
    /// Mean barometric bias in meters.
    pub baro_bias_m: f64,
    /// Mean magnetic map bias in tesla.
    pub mag_bias_t: f64,
    /// Horizontal 1-sigma of the cloud in meters.
    pub horizontal_1sigma_m: f64,
    /// Effective sample size `1/Σwᵢ²`.
    pub effective_sample_size: f64,
    /// Whether the horizontal 1-sigma is at or below
    /// [`MpfConfig::unimodal_max_1sigma_m`].
    pub unimodal: bool,
}

/// Rao–Blackwellized particle filter over a tangent plane (ADR-0014).
#[derive(Debug, Clone)]
pub struct MarginalizedParticleFilter {
    config: MpfConfig,
    plane: LocalTangentPlane,
    rng: SplitMix64,
    x_n: Vec<Vector3<f64>>,
    x_l: Vec<LinearState>,
    log_w: Vec<f64>,
    p_l: LinearCov,
    next_sequence: WrappingSequence,
}

impl MarginalizedParticleFilter {
    /// Draws the particles around `origin` on `ellipsoid`. The positions
    /// come from `position_cov` in the local NED axes at `origin`. Each
    /// particle starts with the mean velocity `velocity` and zero biases.
    /// The shared covariance holds `velocity_cov` and the stationary bias
    /// variances.
    ///
    /// # Errors
    ///
    /// [`MpfError::InvalidConfig`], [`MpfError::NonFiniteValue`] for an
    /// implausible origin or velocity, and
    /// [`MpfError::ImplausibleCovariance`].
    pub fn new(
        config: MpfConfig,
        origin: GeodeticPosition,
        ellipsoid: BodyEllipsoid,
        position_cov: SymmetricCov3,
        velocity: NedVelocity,
        velocity_cov: SymmetricCov3,
    ) -> Result<Self, MpfError> {
        if !config.is_valid() {
            return Err(MpfError::InvalidConfig);
        }
        if !velocity.is_finite() {
            return Err(MpfError::NonFiniteValue);
        }
        let plane =
            LocalTangentPlane::new(origin, ellipsoid).map_err(|_| MpfError::NonFiniteValue)?;
        let (p_pos, p_vel) = (cov3_to_matrix(&position_cov), cov3_to_matrix(&velocity_cov));
        if !(position_cov.is_plausible() && velocity_cov.is_plausible())
            || !(is_positive_definite(&p_pos) && is_positive_definite(&p_vel))
        {
            return Err(MpfError::ImplausibleCovariance);
        }
        let factor = p_pos.cholesky().ok_or(MpfError::ImplausibleCovariance)?.l();
        let mut rng = SplitMix64::new(config.seed);
        let count = config.particle_count;
        let x_n = (0..count)
            .map(|_| factor * Vector3::new(rng.gaussian(), rng.gaussian(), rng.gaussian()))
            .collect();
        let mean = LinearState::new(
            velocity.north_mps,
            velocity.east_mps,
            velocity.down_mps,
            0.0,
            0.0,
        );
        let mut p_l = LinearCov::zeros();
        p_l.fixed_view_mut::<3, 3>(0, 0).copy_from(&p_vel);
        p_l[(ALT, ALT)] = config.baro_bias.stationary_variance();
        p_l[(MAG, MAG)] = config.mag_bias.stationary_variance();
        Ok(Self {
            config,
            plane,
            rng,
            x_n,
            x_l: vec![mean; count],
            log_w: vec![-(count as f64).ln(); count],
            p_l,
            next_sequence: WrappingSequence::new(0),
        })
    }

    /// The normalized weights.
    fn weights(&self) -> Vec<f64> {
        self.log_w.iter().map(|w| w.exp()).collect()
    }

    /// The effective sample size `1/Σwᵢ²`.
    #[must_use]
    pub fn effective_sample_size(&self) -> f64 {
        1.0 / self.weights().iter().map(|w| w * w).sum::<f64>()
    }

    /// The weighted estimate.
    #[must_use]
    pub fn estimate(&self) -> MpfEstimate {
        let weights = self.weights();
        let mean_n = self
            .x_n
            .iter()
            .zip(&weights)
            .fold(Vector3::zeros(), |acc, (x, w)| acc + x * *w);
        let mean_l = self
            .x_l
            .iter()
            .zip(&weights)
            .fold(LinearState::zeros(), |acc, (x, w)| acc + x * *w);
        let mut spread_n = Matrix3::zeros();
        let mut spread_v = Matrix3::zeros();
        for ((x_n, x_l), w) in self.x_n.iter().zip(&self.x_l).zip(&weights) {
            let dn = x_n - mean_n;
            let dv = x_l.fixed_rows::<3>(0) - mean_l.fixed_rows::<3>(0);
            spread_n += dn * dn.transpose() * *w;
            spread_v += dv * dv.transpose() * *w;
        }
        let position = self
            .plane
            .from_ned(&NedOffset::new(mean_n.x, mean_n.y, mean_n.z));
        let v_plane = Vector3::new(mean_l[0], mean_l[1], mean_l[2]);
        let vel_cov = self.p_l.fixed_view::<3, 3>(0, 0) + spread_v;
        let (v, pos_cov, vel_cov) =
            plane_to_vehicle(&self.plane, position, v_plane, spread_n, vel_cov);
        let horizontal = horizontal_1sigma_m(&pos_cov);
        MpfEstimate {
            position,
            velocity: NedVelocity::new(v.x, v.y, v.z),
            position_cov: matrix_to_cov3(&pos_cov),
            velocity_cov: matrix_to_cov3(&vel_cov),
            baro_bias_m: mean_l[ALT],
            mag_bias_t: mean_l[MAG],
            horizontal_1sigma_m: horizontal,
            effective_sample_size: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
            unimodal: horizontal <= self.config.unimodal_max_1sigma_m,
        }
    }

    /// Publishes the estimate at `now`. A particle filter on one map has
    /// no second means to cross-check it, so the best quality is
    /// `Degraded`. A cloud wider than
    /// [`MpfConfig::unimodal_max_1sigma_m`] is `Unusable`.
    pub fn publish(
        &mut self,
        now: MonotonicNanos,
        clock: ClockDomainId,
        composition: SourceComposition,
    ) -> NavigationSolution {
        let estimate = self.estimate();
        let quality = if estimate.unimodal {
            SolutionQuality::Degraded
        } else {
            SolutionQuality::Unusable
        };
        let vertical = estimate.position_cov.diagonal().2.max(0.0).sqrt();
        let integrity = IntegrityAssessment::new(
            quality,
            composition,
            Redundancy::None,
            estimate.horizontal_1sigma_m,
            vertical,
            FaultDetection::Unavailable,
        );
        let sequence = self.next_sequence;
        self.next_sequence = sequence.next();
        NavigationSolution::new(
            SolutionStamp::new(SourceEpoch::new(0), sequence, now, clock),
            estimate.position,
            estimate.velocity,
            estimate.position_cov,
            estimate.velocity_cov,
            integrity,
            composition,
            self.plane.ellipsoid().contract(),
        )
    }
}
