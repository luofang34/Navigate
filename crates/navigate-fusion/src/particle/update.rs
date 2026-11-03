//! Measurement updates of the particle filter: scalar map values,
//! barometric altitudes, and velocity fixes.
//!
//! Each update is a Kalman update of the linear means with the shared
//! covariance, and a likelihood update of the weights. The covariance
//! update uses the Joseph form.

use nalgebra::{Matrix3, SMatrix, Vector3};
use navigate_contract::{NedVelocity, SymmetricCov3};
use navigate_geodesy::NedOffset;

use super::{
    ALT, L, LinearCov, MAG, MarginalizedParticleFilter, MpfError, MpfParticle, ScalarFieldMap,
    uniform_log_weight,
};
use crate::float_math::{exp, ln};
use crate::kalman::{cov3_to_matrix, is_positive_definite};

impl MarginalizedParticleFilter<'_> {
    /// Applies a scalar map measurement `z = map(x_n) + b_mag + e` with
    /// the variance `variance`. The map is sampled at each particle. A
    /// particle off the map gets a zero weight.
    ///
    /// # Errors
    ///
    /// [`MpfError::NonFiniteValue`], [`MpfError::ImplausibleCovariance`]
    /// for a variance that is not finite and positive, and
    /// [`MpfError::NoParticleOnMap`]. On an error nothing changes.
    pub fn ingest_scalar_map(
        &mut self,
        map: &impl ScalarFieldMap,
        measured: f64,
        variance: f64,
    ) -> Result<(), MpfError> {
        let plane = self.plane;
        self.scalar_update(MAG, measured, variance, |x| {
            map.sample(&plane.from_ned(&NedOffset::new(x.x, x.y, x.z)))
        })
    }

    /// Applies a barometric altitude `z = h(x_n) + b_alt + e`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::ingest_scalar_map`].
    pub fn ingest_altitude(&mut self, altitude_m: f64, variance_m2: f64) -> Result<(), MpfError> {
        let plane = self.plane;
        self.scalar_update(ALT, altitude_m, variance_m2, |x| {
            Some(plane.from_ned(&NedOffset::new(x.x, x.y, x.z)).altitude_m)
        })
    }

    /// Applies a velocity fix in the NED axes of the plane origin.
    ///
    /// # Errors
    ///
    /// [`MpfError::NonFiniteValue`] and
    /// [`MpfError::ImplausibleCovariance`]. On an error nothing changes.
    pub fn ingest_velocity(
        &mut self,
        velocity: NedVelocity,
        covariance: SymmetricCov3,
    ) -> Result<(), MpfError> {
        let r = cov3_to_matrix(&covariance);
        if !velocity.is_finite() {
            return Err(MpfError::NonFiniteValue);
        }
        if !(covariance.is_plausible() && is_positive_definite(&r)) {
            return Err(MpfError::ImplausibleCovariance);
        }
        let z = Vector3::new(velocity.north_mps, velocity.east_mps, velocity.down_mps);
        let mut c = SMatrix::<f64, 3, L>::zeros();
        c.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        let s = c * self.p_l * c.transpose() + r;
        let s_inverse = s
            .cholesky()
            .ok_or(MpfError::ImplausibleCovariance)?
            .inverse();
        let gain = self.p_l * c.transpose() * s_inverse;
        let (active, scratch) = self.pool.split();
        for (next, particle) in scratch.iter_mut().zip(active) {
            let innovation = z - c * particle.linear;
            *next = MpfParticle {
                linear: particle.linear + gain * innovation,
                log_weight: particle.log_weight - 0.5 * innovation.dot(&(s_inverse * innovation)),
                ..*particle
            };
        }
        let joseph = LinearCov::identity() - gain * c;
        let p_l = joseph * self.p_l * joseph.transpose() + gain * r * gain.transpose();
        self.commit(p_l)
    }

    /// The scalar update of the linear state `index` with the particle
    /// prediction `predict` of each position.
    fn scalar_update(
        &mut self,
        index: usize,
        measured: f64,
        variance: f64,
        predict: impl Fn(&Vector3<f64>) -> Option<f64>,
    ) -> Result<(), MpfError> {
        if !measured.is_finite() {
            return Err(MpfError::NonFiniteValue);
        }
        if !(variance.is_finite() && variance > 0.0) {
            return Err(MpfError::ImplausibleCovariance);
        }
        let s = self.p_l[(index, index)] + variance;
        let gain = self.p_l.column(index) / s;
        let (active, scratch) = self.pool.split();
        for (next, particle) in scratch.iter_mut().zip(active) {
            *next = match predict(&particle.position) {
                Some(h) if h.is_finite() => {
                    let innovation = measured - h - particle.linear[index];
                    MpfParticle {
                        linear: particle.linear + gain * innovation,
                        log_weight: particle.log_weight - 0.5 * innovation * innovation / s,
                        ..*particle
                    }
                }
                _ => MpfParticle {
                    log_weight: f64::NEG_INFINITY,
                    ..*particle
                },
            };
        }
        let mut c = SMatrix::<f64, 1, L>::zeros();
        c[(0, index)] = 1.0;
        let joseph = LinearCov::identity() - gain * c;
        let p_l = joseph * self.p_l * joseph.transpose() + gain * gain.transpose() * variance;
        self.commit(p_l)
    }

    /// Normalizes the weights of the scratch particles with the
    /// log-sum-exp form, makes them the active particles, keeps `p_l`,
    /// and resamples when the effective sample size is low. When no
    /// scratch particle has a weight, nothing changes.
    fn commit(&mut self, p_l: LinearCov) -> Result<(), MpfError> {
        let (_, scratch) = self.pool.split();
        let peak = scratch
            .iter()
            .map(|p| p.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        if !peak.is_finite() {
            return Err(MpfError::NoParticleOnMap {
                particle_count: scratch.len(),
            });
        }
        let total = scratch
            .iter()
            .map(|p| exp(p.log_weight - peak))
            .sum::<f64>();
        let log_total = peak + ln(total);
        for particle in scratch.iter_mut() {
            particle.log_weight -= log_total;
        }
        self.pool.swap();
        self.p_l = (p_l + p_l.transpose()) * 0.5;
        let count = self.pool.len() as f64;
        if self.effective_sample_size() < self.config.resample_threshold_ratio * count {
            self.resample();
        }
        Ok(())
    }

    /// Systematic resampling with one uniform offset. The selected
    /// particles go into the scratch area, and then the two slices swap.
    fn resample(&mut self) {
        let offset = self.rng.uniform();
        let (active, scratch) = self.pool.split();
        let count = active.len();
        let step = 1.0 / count as f64;
        let log_weight = uniform_log_weight(count);
        let mut target = offset * step;
        let mut cumulative = 0.0;
        let mut filled = 0_usize;
        let mut last = MpfParticle::ZERO;
        for particle in active.iter() {
            cumulative += exp(particle.log_weight);
            last = *particle;
            while target < cumulative && filled < count {
                if let Some(slot) = scratch.get_mut(filled) {
                    *slot = MpfParticle {
                        log_weight,
                        ..*particle
                    };
                }
                filled = filled.wrapping_add(1);
                target += step;
            }
        }
        // Rounding can leave the cumulative sum below the last target.
        for slot in scratch.iter_mut().skip(filled) {
            *slot = MpfParticle { log_weight, ..last };
        }
        self.pool.swap();
    }
}
