//! The time update with the cross-coupling of Schön, Gustafsson, and
//! Nordlund (2005).
//!
//! The nonlinear state moves with the linear state:
//! `x_n' = x_n + A_n·x_l + w_n` with `A_n = [dt·I₃, 0]`. Thus the sampled
//! motion of each particle is a measurement of its linear state. With
//! `N = A_n·P_l·A_nᵀ + Q_n` and `L = A_l·P_l·A_nᵀ·N⁻¹`, the update is
//! `x_l' = A_l·x_l + L·(Δx_n − A_n·x_l)` and
//! `P_l' = A_l·P_l·A_lᵀ + Q_l − L·N·Lᵀ`. This is the step that lets map
//! measurements of the position correct the velocity.

use nalgebra::{Matrix3, SMatrix, Vector3};

use super::{ALT, L, LinearCov, MAG, MarginalizedParticleFilter, MpfError};
use crate::float_math::powi;

impl MarginalizedParticleFilter<'_> {
    /// Advances every particle by `dt_s` seconds.
    ///
    /// # Errors
    ///
    /// [`MpfError::NonFiniteValue`] for a `dt_s` that is not finite and
    /// positive, and [`MpfError::ImplausibleCovariance`] when the motion
    /// covariance has no factorization. On an error nothing changes.
    pub fn predict(&mut self, dt_s: f64) -> Result<(), MpfError> {
        self.predict_with_coupling(dt_s, true)
    }

    /// The time update. Without the coupling the linear means do not
    /// learn from the sampled motion. The tests use that form as the
    /// reference.
    pub(crate) fn predict_with_coupling(
        &mut self,
        dt_s: f64,
        coupled: bool,
    ) -> Result<(), MpfError> {
        if !(dt_s.is_finite() && dt_s > 0.0) {
            return Err(MpfError::NonFiniteValue);
        }
        let mut a_n = SMatrix::<f64, 3, L>::zeros();
        a_n.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::from_diagonal_element(dt_s));
        let (phi_alt, q_alt) = self.config.baro_bias.propagate(1.0, dt_s);
        let (phi_mag, q_mag) = self.config.mag_bias.propagate(1.0, dt_s);
        let mut a_l = LinearCov::identity();
        a_l[(ALT, ALT)] = phi_alt;
        a_l[(MAG, MAG)] = phi_mag;
        let q_n =
            Matrix3::from_diagonal_element(powi(self.config.pos_process_sigma_m_sqrt_s, 2) * dt_s);
        let mut q_l = LinearCov::zeros();
        let q_v = powi(self.config.vel_process_sigma_mps_sqrt_s, 2) * dt_s;
        for axis in 0..3 {
            q_l[(axis, axis)] = q_v;
        }
        q_l[(ALT, ALT)] = q_alt;
        q_l[(MAG, MAG)] = q_mag;
        let n_t = a_n * self.p_l * a_n.transpose() + q_n;
        let factor = n_t.cholesky().ok_or(MpfError::ImplausibleCovariance)?;
        let gain = a_l * self.p_l * a_n.transpose() * factor.inverse();
        let l = factor.l();
        for particle in self.pool.active_mut() {
            let (x_n, x_l) = (&mut particle.position, &mut particle.linear);
            let predicted = a_n * *x_l;
            let noise = Vector3::new(
                self.rng.gaussian(),
                self.rng.gaussian(),
                self.rng.gaussian(),
            );
            let motion = predicted + l * noise;
            *x_n += motion;
            *x_l = if coupled {
                a_l * *x_l + gain * (motion - predicted)
            } else {
                a_l * *x_l
            };
        }
        let mut p_l = a_l * self.p_l * a_l.transpose() + q_l;
        if coupled {
            p_l -= gain * n_t * gain.transpose();
        }
        self.p_l = (p_l + p_l.transpose()) * 0.5;
        Ok(())
    }
}
