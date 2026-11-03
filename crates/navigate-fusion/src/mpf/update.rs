//! Measurement updates of the particle filter: scalar map values,
//! barometric altitudes, and velocity fixes.
//!
//! Each update is a Kalman update of the linear means with the shared
//! covariance, and a likelihood update of the weights. The covariance
//! update uses the Joseph form.

use nalgebra::{Matrix3, SMatrix, Vector3};
use navigate_contract::{NedVelocity, SymmetricCov3};
use navigate_geodesy::NedOffset;

use super::{ALT, L, LinearCov, MAG, MarginalizedParticleFilter, MpfError, ScalarFieldMap};
use crate::filter::{cov3_to_matrix, is_positive_definite};

impl MarginalizedParticleFilter {
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
        let predicted: Vec<Option<f64>> = self
            .x_n
            .iter()
            .map(|x| map.sample(&self.plane.from_ned(&NedOffset::new(x.x, x.y, x.z))))
            .collect();
        self.scalar_update(&predicted, MAG, measured, variance)
    }

    /// Applies a barometric altitude `z = h(x_n) + b_alt + e`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::ingest_scalar_map`].
    pub fn ingest_altitude(&mut self, altitude_m: f64, variance_m2: f64) -> Result<(), MpfError> {
        let predicted: Vec<Option<f64>> = self
            .x_n
            .iter()
            .map(|x| {
                Some(
                    self.plane
                        .from_ned(&NedOffset::new(x.x, x.y, x.z))
                        .altitude_m,
                )
            })
            .collect();
        self.scalar_update(&predicted, ALT, altitude_m, variance_m2)
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
        let mut log_w = self.log_w.clone();
        let mut x_l = self.x_l.clone();
        for (mean, weight) in x_l.iter_mut().zip(log_w.iter_mut()) {
            let innovation = z - c * *mean;
            *mean += gain * innovation;
            *weight -= 0.5 * innovation.dot(&(s_inverse * innovation));
        }
        let joseph = LinearCov::identity() - gain * c;
        let p_l = joseph * self.p_l * joseph.transpose() + gain * r * gain.transpose();
        self.commit(x_l, log_w, p_l)
    }

    /// The scalar update of the linear state `index` with the particle
    /// predictions `predicted`.
    fn scalar_update(
        &mut self,
        predicted: &[Option<f64>],
        index: usize,
        measured: f64,
        variance: f64,
    ) -> Result<(), MpfError> {
        if !measured.is_finite() {
            return Err(MpfError::NonFiniteValue);
        }
        if !(variance.is_finite() && variance > 0.0) {
            return Err(MpfError::ImplausibleCovariance);
        }
        let s = self.p_l[(index, index)] + variance;
        let gain = self.p_l.column(index) / s;
        let mut log_w = self.log_w.clone();
        let mut x_l = self.x_l.clone();
        for ((mean, weight), value) in x_l.iter_mut().zip(log_w.iter_mut()).zip(predicted) {
            match value {
                Some(h) if h.is_finite() => {
                    let innovation = measured - h - mean[index];
                    *mean += gain * innovation;
                    *weight -= 0.5 * innovation * innovation / s;
                }
                _ => *weight = f64::NEG_INFINITY,
            }
        }
        let mut c = SMatrix::<f64, 1, L>::zeros();
        c[(0, index)] = 1.0;
        let joseph = LinearCov::identity() - gain * c;
        let p_l = joseph * self.p_l * joseph.transpose() + gain * gain.transpose() * variance;
        self.commit(x_l, log_w, p_l)
    }

    /// Normalizes the weights with the log-sum-exp form, keeps the new
    /// state, and resamples when the effective sample size is low.
    fn commit(
        &mut self,
        x_l: Vec<super::LinearState>,
        mut log_w: Vec<f64>,
        p_l: LinearCov,
    ) -> Result<(), MpfError> {
        let peak = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !peak.is_finite() {
            return Err(MpfError::NoParticleOnMap {
                particle_count: log_w.len(),
            });
        }
        let total = log_w.iter().map(|w| (w - peak).exp()).sum::<f64>();
        let log_total = peak + total.ln();
        for weight in &mut log_w {
            *weight -= log_total;
        }
        self.x_l = x_l;
        self.log_w = log_w;
        self.p_l = (p_l + p_l.transpose()) * 0.5;
        let count = self.log_w.len() as f64;
        if self.effective_sample_size() < self.config.resample_threshold_ratio * count {
            self.resample();
        }
        Ok(())
    }

    /// Systematic resampling with one uniform offset.
    fn resample(&mut self) {
        let count = self.log_w.len();
        let step = 1.0 / count as f64;
        let mut target = self.rng.uniform() * step;
        let mut cumulative = 0.0;
        let mut source = 0_usize;
        let weights = self.weights();
        let mut x_n = Vec::with_capacity(count);
        let mut x_l = Vec::with_capacity(count);
        for (index, weight) in weights.iter().enumerate() {
            cumulative += weight;
            while target < cumulative && x_n.len() < count {
                x_n.push(self.x_n[index]);
                x_l.push(self.x_l[index]);
                target += step;
            }
            source = index;
        }
        // Rounding can leave the cumulative sum below the last target.
        while x_n.len() < count {
            x_n.push(self.x_n[source]);
            x_l.push(self.x_l[source]);
        }
        self.x_n = x_n;
        self.x_l = x_l;
        self.log_w = vec![-(count as f64).ln(); count];
    }
}
