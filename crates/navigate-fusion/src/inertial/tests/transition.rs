//! Finite-difference checks of the transition matrix, the reset
//! Jacobian, and the discretization.

use nalgebra::{Matrix3, SMatrix, Vector3};
use navigate_contract::GeodeticPosition;

use super::error_between;
use crate::inertial::config::InertialConfig;
use crate::inertial::discretization::{expm, van_loan};
use crate::inertial::local_level::body_fixed;
use crate::inertial::mechanization::{
    ImuRates, continuous_jacobian, continuous_noise, mechanize, transition,
};
use crate::inertial::nominal::{
    ALT_BIAS, InertialNominal, MAG_BIAS, Mat17, N, Vec17, exp_rotation, reset_jacobian,
    rotation_vector,
};
use crate::kalman::FogmBias;
use navigate_astrodynamics::{PlanetaryBody, SpaceEnvironment};

fn earth_frame() -> SpaceEnvironment {
    SpaceEnvironment::from_body(PlanetaryBody::EARTH_WGS84)
}

/// A body-fixed position 500 m above the WGS84 ellipsoid at 47°N, 8°E.
fn earth_position() -> Vector3<f64> {
    let origin = GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0);
    body_fixed(&origin, navigate_geodesy::BodyEllipsoid::WGS84)
}

fn short_bias_config() -> InertialConfig {
    InertialConfig {
        accel_bias: FogmBias::new(100.0, 1e-3),
        gyro_bias: FogmBias::new(200.0, 1e-5),
        baro_bias: FogmBias::new(50.0, 3.0),
        mag_bias: FogmBias::new(30.0, 3e-9),
        ..InertialConfig::default()
    }
}

fn nominal() -> InertialNominal {
    InertialNominal {
        p: earth_position() + Vector3::new(120.0, -40.0, -300.0),
        v: Vector3::new(60.0, 10.0, -2.0),
        q: nalgebra::UnitQuaternion::from_euler_angles(0.2, -0.1, 1.0),
        b_a: Vector3::new(0.01, -0.02, 0.005),
        b_g: Vector3::new(1e-4, -2e-4, 5e-5),
        b_alt: 2.0,
        b_mag: 1e-9,
    }
}

fn rates() -> ImuRates {
    ImuRates {
        specific_force: Vector3::new(0.5, -0.3, -9.7),
        angular_rate: Vector3::new(0.05, -0.2, 0.3),
    }
}

/// Step size per error component: position, velocity, attitude, and
/// each bias have their own scale. The body-fixed position is about
/// 6.4e6 m, so its rounding is about 1e-9 m. Each step is large enough
/// that this rounding stays below the tolerance of the check.
fn epsilon(index: usize) -> f64 {
    match index {
        0..=2 => 1.0,
        3..=5 => 1e-2,
        6..=8 => 1e-3,
        9..=11 => 1e-3,
        12..=14 => 1e-4,
        15 => 1e-2,
        _ => 1e-12,
    }
}

#[test]
fn transition_matrix_matches_finite_differences_of_the_mechanization() {
    let field = earth_frame();
    let config = short_bias_config();
    let x = nominal();
    let dt_s = 0.005;
    let (next, phi, _) = transition(&x, &rates(), dt_s, &field, &config).expect("finite");
    let mut numeric = Mat17::zeros();
    for j in 0..N {
        let eps = epsilon(j);
        let mut dx = Vec17::zeros();
        dx[j] = eps;
        let plus = mechanize(&x.inject(&dx), &rates(), dt_s, &field, &config).0;
        let minus = mechanize(&x.inject(&(-dx)), &rates(), dt_s, &field, &config).0;
        let column = (error_between(&plus, &next) - error_between(&minus, &next)) / (2.0 * eps);
        numeric.set_column(j, &column);
    }
    let error = (phi - numeric).abs().max();
    assert!(error < 1e-5, "{error}\n{}", phi - numeric);
    // The attitude couples into the velocity, so the check is not
    // trivial.
    assert!(phi[(3, 6)].abs() > 1e-3 || phi[(4, 6)].abs() > 1e-3);
}

#[test]
fn the_frame_rotation_cancels_in_the_attitude_block() {
    // Without a body rate, the attitude error of a vehicle at rest in a
    // rotating frame is constant: `F_θθ = −[ω_ib×]` with `ω_ib = Cᵀ·ω`
    // cancels the frame rotation in `δθ̇`.
    let field = earth_frame();
    let config = InertialConfig::default();
    let x = InertialNominal {
        v: Vector3::zeros(),
        b_a: Vector3::zeros(),
        b_g: Vector3::zeros(),
        ..nominal()
    };
    let omega_body =
        x.c_b_to_f().transpose() * navigate_astrodynamics::GravityField::frame_rotation(&field);
    let rates = ImuRates {
        specific_force: Vector3::new(0.0, 0.0, -9.8),
        angular_rate: omega_body,
    };
    let (_, midpoint) = mechanize(&x, &rates, 1.0, &field, &config);
    let f = continuous_jacobian(&midpoint, &field, &config);
    let block = f.fixed_view::<3, 3>(6, 6).into_owned();
    assert!((block + omega_body.cross_matrix()).norm() < 1e-18);
}

#[test]
fn the_reset_jacobian_matches_finite_differences_of_the_attitude_log() {
    let q_nom = nalgebra::UnitQuaternion::from_euler_angles(0.4, -0.3, 2.0);
    let dtheta_hat = Vector3::new(0.012, -0.018, 0.009);
    let reset = q_nom * exp_rotation(&dtheta_hat);
    let eps = 1e-6;
    let mut numeric = Matrix3::zeros();
    for j in 0..3 {
        let mut step = Vector3::zeros();
        step[j] = eps;
        let after = |e: Vector3<f64>| {
            rotation_vector(&(reset.inverse() * (q_nom * exp_rotation(&(dtheta_hat + e)))))
        };
        numeric.set_column(j, &((after(step) - after(-step)) / (2.0 * eps)));
    }
    let analytic = reset_jacobian(&dtheta_hat)
        .fixed_view::<3, 3>(6, 6)
        .into_owned();
    // The analytic form drops the second-order term `[δθ×]²/12`.
    assert!(
        (numeric - analytic).abs().max() < 1e-4,
        "{}",
        numeric - analytic
    );
    assert!((numeric - Matrix3::identity()).abs().max() > 5e-3);
}

#[test]
fn the_pade_exponential_matches_closed_forms() {
    let theta = Vector3::new(0.3, -1.2, 2.5);
    let rotation = expm(&theta.cross_matrix()).expect("finite");
    let rodrigues = exp_rotation(&theta).to_rotation_matrix().into_inner();
    assert!((rotation - rodrigues).abs().max() < 1e-13);
    let diagonal =
        SMatrix::<f64, 4, 4>::from_diagonal(&nalgebra::Vector4::new(-3.0, 0.0, 1.5, 7.0));
    let exponential = expm(&diagonal).expect("finite");
    for (i, value) in [-3.0_f64, 0.0, 1.5, 7.0].into_iter().enumerate() {
        assert!((exponential[(i, i)] / value.exp() - 1.0).abs() < 1e-13);
    }
    let mut bad = SMatrix::<f64, 4, 4>::zeros();
    bad[(1, 2)] = f64::NAN;
    assert!(expm(&bad).is_none());
}

#[test]
fn discrete_bias_noise_equals_the_gauss_markov_step() {
    let field = earth_frame();
    let config = short_bias_config();
    let x = nominal();
    let (_, midpoint) = mechanize(&x, &rates(), 0.5, &field, &config);
    let f = continuous_jacobian(&midpoint, &field, &config);
    for dt_s in [0.01, 0.5, 2.0] {
        let (phi, q_d) = van_loan(&f, &continuous_noise(&config), dt_s).expect("finite");
        for (index, bias) in [(ALT_BIAS, config.baro_bias), (MAG_BIAS, config.mag_bias)] {
            let (decay, variance) = bias.propagate(1.0, dt_s);
            assert!((phi[(index, index)] - decay).abs() < 1e-14);
            assert!(
                (q_d[(index, index)] / variance - 1.0).abs() < 1e-10,
                "{dt_s}"
            );
        }
        // The discrete noise is symmetric and positive semidefinite.
        assert!((q_d - q_d.transpose()).abs().max() == 0.0);
        let eigen = q_d.symmetric_eigenvalues();
        assert!(eigen.iter().all(|e| *e > -1e-24), "{eigen}");
    }
}
