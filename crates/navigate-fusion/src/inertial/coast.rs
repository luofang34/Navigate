//! Force-model propagation of the nominal state and of the error
//! covariance (ADR-0018).
//!
//! In the force-model mode no IMU drives the state. The position and the
//! velocity follow `ṗ = v` and `v̇ = a(p, v) + a_dmc`. The acceleration
//! `a` of the [`OrbitalForceModel`] holds the gravity, the centrifugal
//! and the Coriolis terms, the solar radiation pressure, and the drag.
//! The empirical acceleration `a_dmc` of the dynamic model compensation
//! decays with its first-order Gauss–Markov model. A fourth-order
//! Runge–Kutta step integrates the position and the velocity. The body
//! turns with the constant inertial rate `ω_ib` of the coast, so the
//! attitude step is the step of [`super::mechanization`] for a constant
//! rate.
//!
//! The error dynamics at the midpoint of the step are
//!
//! - `δṗ = δv`
//! - `δv̇ = (∂a/∂r)·δp + (∂a/∂v)·δv + δa_dmc`
//! - `δθ̇ = −[ω_ib×]·δθ`
//!
//! with the bias, empirical acceleration, and clock blocks of
//! [`stochastic`]. No IMU drives the state, so the IMU biases do not
//! couple into the velocity and the attitude.

use nalgebra::{Const, DimMin, Matrix3, Vector3};
use navigate_astrodynamics::OrbitalForceModel;

use super::config::InertialConfig;
use super::discretization::van_loan;
use super::nominal::{ATT, ErrorMat, InertialNominal, POS, VEL, exp_rotation, renormalized};
use super::propagation::ForceModelCoast;
use super::stochastic;
use crate::float_math::powi;

/// The values of one coast step at its midpoint, where the dynamics are
/// linearized.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CoastMidpoint {
    p_mid: Vector3<f64>,
    v_mid: Vector3<f64>,
}

/// Advances the nominal state by `dt_s` seconds with the force model.
pub(crate) fn coast(
    x: &InertialNominal,
    dt_s: f64,
    model: &OrbitalForceModel<'_>,
    settings: &ForceModelCoast,
    config: &InertialConfig,
) -> (InertialNominal, CoastMidpoint) {
    let a_dmc = x.a_dmc;
    let empirical = |t: f64| {
        config
            .dmc
            .as_ref()
            .map_or(Vector3::zeros(), |dmc| a_dmc.map(|a| dmc.propagate(a, t).0))
    };
    let (p_next, v_next) = model.rk4_step_with(&x.p, &x.v, dt_s, empirical);
    let omega = Vector3::from(model.frame_rotation_rad_per_s);
    let w_fb_b = Vector3::from(settings.body_rate_rad_s) - x.c_b_to_f().transpose() * omega;
    let next = InertialNominal {
        p: p_next,
        v: v_next,
        q: renormalized(x.q * exp_rotation(&(w_fb_b * dt_s))),
        ..stochastic::advance(x, dt_s, config)
    };
    let midpoint = CoastMidpoint {
        p_mid: (x.p + p_next) * 0.5,
        v_mid: (x.v + v_next) * 0.5,
    };
    (next, midpoint)
}

/// The continuous Jacobian `F` of the coast at `midpoint`.
pub(crate) fn coast_jacobian<const N: usize>(
    midpoint: &CoastMidpoint,
    model: &OrbitalForceModel<'_>,
    settings: &ForceModelCoast,
    config: &InertialConfig,
) -> ErrorMat<N> {
    let forces = model.evaluate(&midpoint.p_mid, &midpoint.v_mid);
    let identity = Matrix3::identity();
    let mut f = ErrorMat::<N>::zeros();
    f.fixed_view_mut::<3, 3>(POS, VEL).copy_from(&identity);
    f.fixed_view_mut::<3, 3>(VEL, POS)
        .copy_from(&forces.position_jacobian);
    f.fixed_view_mut::<3, 3>(VEL, VEL)
        .copy_from(&forces.velocity_jacobian);
    if let (Some(_), Some(dmc)) = (&config.dmc, config.layout.dmc()) {
        f.fixed_view_mut::<3, 3>(VEL, dmc).copy_from(&identity);
    }
    let body_rate = Vector3::from(settings.body_rate_rad_s);
    f.fixed_view_mut::<3, 3>(ATT, ATT)
        .copy_from(&(-body_rate.cross_matrix()));
    stochastic::add_dynamics(&mut f, config);
    f
}

/// The continuous noise density `Q_c` of the coast: the white
/// acceleration of the unmodeled forces, the white noise of the body
/// rate, and the bias, empirical acceleration, and clock densities.
pub(crate) fn coast_noise<const N: usize>(
    settings: &ForceModelCoast,
    config: &InertialConfig,
) -> ErrorMat<N> {
    let accel = powi(settings.acceleration_noise_sigma_mps2_sqrt_hz, 2);
    let rate = powi(settings.body_rate_noise_sigma_rad_s_sqrt_hz, 2);
    let mut q = ErrorMat::<N>::zeros();
    for axis in 0..3 {
        q[(VEL + axis, VEL + axis)] = accel;
        q[(ATT + axis, ATT + axis)] = rate;
    }
    stochastic::add_noise(&mut q, config);
    q
}

/// One coast step: the next nominal state, the transition `Φ`, and the
/// discrete noise `Q_d`. `None` when the discretization fails.
pub(crate) fn coast_transition<const N: usize>(
    x: &InertialNominal,
    dt_s: f64,
    model: &OrbitalForceModel<'_>,
    settings: &ForceModelCoast,
    config: &InertialConfig,
) -> Option<(InertialNominal, ErrorMat<N>, ErrorMat<N>)>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let (next, midpoint) = coast(x, dt_s, model, settings, config);
    let f = coast_jacobian(&midpoint, model, settings, config);
    let (phi, q_d) = van_loan(&f, &coast_noise(settings, config), dt_s)?;
    Some((next, phi, q_d))
}
