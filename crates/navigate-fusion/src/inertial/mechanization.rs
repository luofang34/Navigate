//! Strapdown mechanization in a rotating or inertial Cartesian frame and
//! the continuous error-state dynamics (ADR-0014).
//!
//! The frame `F` rotates with the rate `ω` relative to inertial space.
//! The velocity is relative to `F`. The specific force `f` and the
//! inertial angular rate `ω_ib` arrive in the body axes. The equations
//! are `ṗ = v`, `v̇ = C·f + g_app(p) − 2·ω × v`, and `Ċ = C·[ω_fb×]` with
//! `ω_fb = ω_ib − Cᵀ·ω`. The apparent gravity `g_app` includes the
//! centrifugal term.
//!
//! An [`ImuIncrement`] has its own step. The frame rotation over the
//! step is exact: `q' = exp(−ω·Δt) ⊗ q ⊗ exp(Δθ̃)`. The velocity
//! increment `Δv_rot` rotates into the frame axes with the frame rotation
//! at the midpoint of the step.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::config::InertialConfig;
use super::discretization::van_loan;
use super::imu::{ImuIncrement, ImuSample};
use super::nominal::{
    ACCEL_BIAS, ATT, ErrorMat, GYRO_BIAS, InertialNominal, POS, VEL, exp_rotation, renormalized,
};
use super::stochastic;
use crate::float_math::powi;
use navigate_astrodynamics::GravityField;

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

    /// The mean rates of an increment: `Δv/Δt` and `Δθ/Δt`.
    pub(crate) fn mean_of(increment: &ImuIncrement) -> Self {
        let inverse_s = 1.0 / increment.dt_s;
        Self {
            specific_force: Vector3::from(increment.delta_velocity_body_mps) * inverse_s,
            angular_rate: Vector3::from(increment.delta_theta_body_rad) * inverse_s,
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
    let next = advanced(x, v_next, q_next, dt_s, config);
    let midpoint = Midpoint {
        f_b,
        w_ib_b,
        c_mid,
        p_mid,
    };
    (next, midpoint)
}

/// Advances the nominal state over one full increment of `dt_s`
/// seconds. The rotation correction gives `Δv_rot` in the body axes at
/// the start of the step. The step rotates it into the frame axes with
/// `exp(−½·ω·Δt)·C₀`, then adds the gravity and the Coriolis terms with
/// midpoint RK2. The position uses the trapezoidal rule. The midpoint of
/// the linearization holds the mean rates of the increment.
pub(crate) fn mechanize_increment(
    x: &InertialNominal,
    increment: &ImuIncrement,
    dt_s: f64,
    field: &impl GravityField,
    config: &InertialConfig,
) -> (InertialNominal, Midpoint) {
    let sensor_dt_s = increment.dt_s;
    let d_theta = Vector3::from(increment.delta_theta_body_rad) - x.b_g * sensor_dt_s;
    let d_velocity = Vector3::from(increment.delta_velocity_body_mps) - x.b_a * sensor_dt_s;
    let d_velocity_rot = d_velocity + 0.5 * d_theta.cross(&d_velocity);
    let omega = field.frame_rotation();
    let half_frame: UnitQuaternion<f64> = exp_rotation(&(omega * (-0.5 * dt_s)));
    let dv_frame = half_frame * (x.c_b_to_f() * d_velocity_rot);
    let q_next = renormalized(exp_rotation(&(omega * -dt_s)) * x.q * exp_rotation(&d_theta));
    let q_mid = half_frame * x.q * exp_rotation(&(d_theta * 0.5));
    let a_0 = field.gravity(&x.p).0 - 2.0 * omega.cross(&x.v);
    let p_mid = x.p + x.v * (0.5 * dt_s);
    let v_mid = x.v + dv_frame * 0.5 + a_0 * (0.5 * dt_s);
    let a_mid = field.gravity(&p_mid).0 - 2.0 * omega.cross(&v_mid);
    let v_next = x.v + dv_frame + a_mid * dt_s;
    let next = advanced(x, v_next, q_next, dt_s, config);
    let midpoint = Midpoint {
        f_b: d_velocity / sensor_dt_s,
        w_ib_b: d_theta / sensor_dt_s,
        c_mid: q_mid.to_rotation_matrix().into_inner(),
        p_mid,
    };
    (next, midpoint)
}

/// The nominal state after one step: the trapezoidal position, the new
/// velocity and attitude, and the bias, empirical acceleration, and
/// clock models of [`stochastic::advance`].
fn advanced(
    x: &InertialNominal,
    v_next: Vector3<f64>,
    q_next: UnitQuaternion<f64>,
    dt_s: f64,
    config: &InertialConfig,
) -> InertialNominal {
    InertialNominal {
        p: x.p + (x.v + v_next) * (0.5 * dt_s),
        v: v_next,
        q: q_next,
        ..stochastic::advance(x, dt_s, config)
    }
}

/// The continuous Jacobian `F` of the error-state dynamics at the
/// midpoint. The right perturbation removes the frame rotation from the
/// attitude block: `δθ̇ = −[ω_ib×]·δθ − δb_g`. The accelerometers sense
/// each non-gravitational force, so the empirical acceleration does not
/// enter the velocity.
pub(crate) fn continuous_jacobian(
    midpoint: &Midpoint,
    field: &impl GravityField,
    config: &InertialConfig,
) -> ErrorMat {
    let mut f = ErrorMat::zeros();
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
    stochastic::add_dynamics(&mut f, config);
    f
}

/// The continuous process noise density `Q_c`. The accelerometer and
/// gyroscope noise are isotropic, so the rotation into the frame axes
/// does not change the velocity block. [`stochastic::add_noise`] gives
/// the bias, empirical acceleration, and clock densities.
pub(crate) fn continuous_noise(config: &InertialConfig) -> ErrorMat {
    let accel = powi(config.accel_noise_sigma_mps2_sqrt_hz, 2);
    let gyro = powi(config.gyro_noise_sigma_rad_s_sqrt_hz, 2);
    let mut q = ErrorMat::zeros();
    for axis in 0..3 {
        q[(VEL + axis, VEL + axis)] = accel;
        q[(ATT + axis, ATT + axis)] = gyro;
    }
    stochastic::add_noise(&mut q, config);
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
) -> Option<(InertialNominal, ErrorMat, ErrorMat)> {
    let (next, midpoint) = mechanize(x, rates, dt_s, field, config);
    let (phi, q_d) = discretize(&midpoint, dt_s, field, config)?;
    Some((next, phi, q_d))
}

/// One propagation step over a full increment: the next nominal state,
/// `Φ`, and `Q_d`. `None` when the discretization fails.
pub(crate) fn increment_transition(
    x: &InertialNominal,
    increment: &ImuIncrement,
    dt_s: f64,
    field: &impl GravityField,
    config: &InertialConfig,
) -> Option<(InertialNominal, ErrorMat, ErrorMat)> {
    let (next, midpoint) = mechanize_increment(x, increment, dt_s, field, config);
    let (phi, q_d) = discretize(&midpoint, dt_s, field, config)?;
    Some((next, phi, q_d))
}

/// `Φ` and `Q_d` of the error dynamics linearized at `midpoint`.
fn discretize(
    midpoint: &Midpoint,
    dt_s: f64,
    field: &impl GravityField,
    config: &InertialConfig,
) -> Option<(ErrorMat, ErrorMat)> {
    let f = continuous_jacobian(midpoint, field, config);
    van_loan(&f, &continuous_noise(config), dt_s)
}
