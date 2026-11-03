//! Strapdown mechanization in a rotating or inertial Cartesian frame and
//! the continuous error-state dynamics (ADR-0014).
//!
//! The frame `F` rotates with the rate `ω` relative to inertial space.
//! The velocity is relative to `F`. The specific force `f` and the
//! inertial angular rate `ω_ib` arrive in the body axes. The equations
//! are `ṗ = v`, `v̇ = C·f + g_app(p) − 2·ω × v`, and `Ċ = C·[ω_fb×]` with
//! `ω_fb = ω_ib − Cᵀ·ω`. The apparent gravity `g_app` includes the
//! centrifugal term.

use nalgebra::{Matrix3, Vector3};

use super::config::InertialConfig;
use super::discretization::van_loan;
use super::imu::ImuSample;
use super::nominal::{
    ACCEL_BIAS, ALT_BIAS, ATT, GYRO_BIAS, InertialNominal, MAG_BIAS, Mat17, POS, VEL, exp_rotation,
    renormalized,
};
use crate::environment::GravityField;

/// The mean rates of one IMU interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ImuRates {
    /// Specific force in m/s², in the body axes.
    pub(crate) specific_force: Vector3<f64>,
    /// Angular rate relative to inertial space in rad/s, in the body axes.
    pub(crate) angular_rate: Vector3<f64>,
}

impl ImuRates {
    pub(crate) fn of(sample: &ImuSample) -> Self {
        Self {
            specific_force: Vector3::from(sample.specific_force_body_mps2),
            angular_rate: Vector3::from(sample.angular_rate_body_rad_s),
        }
    }
}

/// The values of one step at the midpoint, where the dynamics are
/// linearized.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Midpoint {
    /// Bias-corrected specific force in the body axes.
    pub(crate) f_b: Vector3<f64>,
    /// Bias-corrected inertial angular rate in the body axes.
    pub(crate) w_ib_b: Vector3<f64>,
    /// Rotation from the body axes to the frame axes at the midpoint.
    pub(crate) c_mid: Matrix3<f64>,
    /// Position at the midpoint.
    pub(crate) p_mid: Vector3<f64>,
}

/// Advances the nominal state by `dt_s` seconds with midpoint RK2 for the
/// velocity, the trapezoidal rule for the position, and the exact
/// quaternion step for a constant rate. Each bias decays with its
/// first-order Gauss–Markov model.
pub(crate) fn mechanize(
    x: &InertialNominal,
    rates: &ImuRates,
    dt_s: f64,
    field: &impl GravityField,
    config: &InertialConfig,
) -> (InertialNominal, Midpoint) {
    let f_b = rates.specific_force - x.b_a;
    let w_ib_b = rates.angular_rate - x.b_g;
    let omega = field.frame_rotation();
    let c_0 = x.c_b_to_f();
    let w_fb_b = w_ib_b - c_0.transpose() * omega;
    let q_mid = x.q * exp_rotation(&(w_fb_b * (0.5 * dt_s)));
    let q_next = renormalized(x.q * exp_rotation(&(w_fb_b * dt_s)));
    let c_mid = q_mid.to_rotation_matrix().into_inner();
    let a_0 = c_0 * f_b + field.gravity(&x.p).0 - 2.0 * omega.cross(&x.v);
    let p_mid = x.p + x.v * (0.5 * dt_s);
    let v_mid = x.v + a_0 * (0.5 * dt_s);
    let a_mid = c_mid * f_b + field.gravity(&p_mid).0 - 2.0 * omega.cross(&v_mid);
    let v_next = x.v + a_mid * dt_s;
    let decay = |bias: &crate::filter::FogmBias, value: f64| bias.propagate(value, dt_s).0;
    let next = InertialNominal {
        p: x.p + (x.v + v_next) * (0.5 * dt_s),
        v: v_next,
        q: q_next,
        b_a: x.b_a.map(|b| decay(&config.accel_bias, b)),
        b_g: x.b_g.map(|b| decay(&config.gyro_bias, b)),
        b_alt: decay(&config.baro_bias, x.b_alt),
        b_mag: decay(&config.mag_bias, x.b_mag),
    };
    let midpoint = Midpoint {
        f_b,
        w_ib_b,
        c_mid,
        p_mid,
    };
    (next, midpoint)
}

/// The continuous Jacobian `F` of the error-state dynamics at the
/// midpoint. The right perturbation removes the frame rotation from the
/// attitude block: `δθ̇ = −[ω_ib×]·δθ − δb_g`.
pub(crate) fn continuous_jacobian(
    midpoint: &Midpoint,
    field: &impl GravityField,
    config: &InertialConfig,
) -> Mat17 {
    let mut f = Mat17::zeros();
    let identity = Matrix3::identity();
    let omega = field.frame_rotation();
    let gradient = field.gravity(&midpoint.p_mid).1;
    let c = midpoint.c_mid;
    f.fixed_view_mut::<3, 3>(POS, VEL).copy_from(&identity);
    f.fixed_view_mut::<3, 3>(VEL, POS).copy_from(&gradient);
    f.fixed_view_mut::<3, 3>(VEL, VEL)
        .copy_from(&(-2.0 * omega.cross_matrix()));
    f.fixed_view_mut::<3, 3>(VEL, ATT)
        .copy_from(&(-c * midpoint.f_b.cross_matrix()));
    f.fixed_view_mut::<3, 3>(VEL, ACCEL_BIAS).copy_from(&(-c));
    f.fixed_view_mut::<3, 3>(ATT, ATT)
        .copy_from(&(-midpoint.w_ib_b.cross_matrix()));
    f.fixed_view_mut::<3, 3>(ATT, GYRO_BIAS)
        .copy_from(&(-identity));
    f.fixed_view_mut::<3, 3>(ACCEL_BIAS, ACCEL_BIAS)
        .copy_from(&(identity * config.accel_bias.decay_rate()));
    f.fixed_view_mut::<3, 3>(GYRO_BIAS, GYRO_BIAS)
        .copy_from(&(identity * config.gyro_bias.decay_rate()));
    f[(ALT_BIAS, ALT_BIAS)] = config.baro_bias.decay_rate();
    f[(MAG_BIAS, MAG_BIAS)] = config.mag_bias.decay_rate();
    f
}

/// The continuous process noise density `Q_c`. The accelerometer and
/// gyroscope noise are isotropic, so the rotation into the frame axes
/// does not change the velocity block. A bias with a first-order
/// Gauss–Markov model has the density `2·σ²/τ`.
pub(crate) fn continuous_noise(config: &InertialConfig) -> Mat17 {
    let accel = config.accel_noise_sigma_mps2_sqrt_hz.powi(2);
    let gyro = config.gyro_noise_sigma_rad_s_sqrt_hz.powi(2);
    let mut q = Mat17::zeros();
    for axis in 0..3 {
        q[(VEL + axis, VEL + axis)] = accel;
        q[(ATT + axis, ATT + axis)] = gyro;
        q[(ACCEL_BIAS + axis, ACCEL_BIAS + axis)] = config.accel_bias.continuous_psd();
        q[(GYRO_BIAS + axis, GYRO_BIAS + axis)] = config.gyro_bias.continuous_psd();
    }
    q[(ALT_BIAS, ALT_BIAS)] = config.baro_bias.continuous_psd();
    q[(MAG_BIAS, MAG_BIAS)] = config.mag_bias.continuous_psd();
    q
}

/// One propagation step: the next nominal state, the transition matrix
/// `Φ`, and the discrete process noise `Q_d`. `None` when the
/// discretization fails.
pub(crate) fn transition(
    x: &InertialNominal,
    rates: &ImuRates,
    dt_s: f64,
    field: &impl GravityField,
    config: &InertialConfig,
) -> Option<(InertialNominal, Mat17, Mat17)> {
    let (next, midpoint) = mechanize(x, rates, dt_s, field, config);
    let f = continuous_jacobian(&midpoint, field, config);
    let (phi, q_d) = van_loan(&f, &continuous_noise(config), dt_s)?;
    Some((next, phi, q_d))
}
