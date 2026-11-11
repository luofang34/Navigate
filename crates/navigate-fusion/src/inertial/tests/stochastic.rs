//! The clock and the empirical acceleration states: the closed-form
//! clock step, and the zero blocks of a state without a model.

use nalgebra::Vector3;

use crate::inertial::ClockNoiseModel;
use crate::inertial::config::InertialConfig;
use crate::inertial::discretization::van_loan;
use crate::inertial::mechanization::{ImuRates, continuous_jacobian, continuous_noise, mechanize};
use crate::inertial::nominal::{CLOCK_BIAS, CLOCK_DRIFT, DMC, InertialNominal};
use crate::kalman::FogmBias;
use navigate_astrodynamics::{PlanetaryBody, SpaceEnvironment};

fn clock() -> ClockNoiseModel {
    ClockNoiseModel {
        white_frequency_psd_m2_per_s: 0.09,
        random_walk_frequency_psd_m2_per_s3: 4e-4,
        initial_bias_sigma_m: 300.0,
        initial_drift_sigma_mps: 2.0,
    }
}

fn jacobian_and_noise(config: &InertialConfig) -> (super::ErrorMat, super::ErrorMat) {
    let field = SpaceEnvironment::from_body(PlanetaryBody::EARTH_WGS84);
    let x = InertialNominal::at(
        Vector3::new(6.4e6, 0.0, 0.0),
        Vector3::new(0.0, 7_500.0, 0.0),
        nalgebra::UnitQuaternion::identity(),
    );
    let rates = ImuRates {
        specific_force: Vector3::new(0.1, 0.0, -0.2),
        angular_rate: Vector3::new(0.0, 1e-3, 0.0),
    };
    let (_, midpoint) = mechanize(&x, &rates, 1.0, &field, config);
    (
        continuous_jacobian(&midpoint, &field, config),
        continuous_noise(config),
    )
}

#[test]
fn the_van_loan_clock_block_equals_the_closed_form() {
    let config = InertialConfig {
        clock_model: Some(clock()),
        ..InertialConfig::default()
    };
    let (f, q_c) = jacobian_and_noise(&config);
    for dt_s in [0.1, 1.0, 30.0] {
        let (phi, q_d) = van_loan(&f, &q_c, dt_s).expect("finite");
        let (phi_clock, q_clock) = clock().discrete(dt_s);
        let rows = [CLOCK_BIAS, CLOCK_DRIFT];
        for (i, row) in rows.into_iter().enumerate() {
            for (j, column) in rows.into_iter().enumerate() {
                assert!((phi[(row, column)] - phi_clock[i][j]).abs() < 1e-12);
                let scale = q_clock[i][j].abs().max(1e-30);
                assert!(
                    (q_d[(row, column)] - q_clock[i][j]).abs() < 1e-9 * scale,
                    "{dt_s} {i} {j}: {} {}",
                    q_d[(row, column)],
                    q_clock[i][j]
                );
            }
        }
    }
}

#[test]
fn a_state_without_a_model_has_zero_blocks() {
    let (f, q_c) = jacobian_and_noise(&InertialConfig::default());
    let (phi, q_d) = van_loan(&f, &q_c, 10.0).expect("finite");
    for state in [DMC, DMC + 1, DMC + 2, CLOCK_BIAS, CLOCK_DRIFT] {
        for other in 0..super::super::nominal::FULL_STATES {
            let identity = if state == other { 1.0 } else { 0.0 };
            assert_eq!(phi[(state, other)], identity);
            assert_eq!(phi[(other, state)], identity);
            assert_eq!(q_d[(state, other)], 0.0);
        }
    }
}

#[test]
fn the_empirical_acceleration_is_a_gauss_markov_state_without_coupling_in_imu_mode() {
    let dmc = FogmBias::new(300.0, 2e-6);
    let config = InertialConfig {
        dmc: Some(dmc),
        ..InertialConfig::default()
    };
    let (f, q_c) = jacobian_and_noise(&config);
    let (phi, q_d) = van_loan(&f, &q_c, 20.0).expect("finite");
    let (decay, variance) = dmc.propagate(1.0, 20.0);
    for axis in 0..3 {
        let index = DMC + axis;
        assert!((phi[(index, index)] - decay).abs() < 1e-14);
        assert!((q_d[(index, index)] / variance - 1.0).abs() < 1e-10);
        // The accelerometers sense each force, so the velocity does not
        // integrate the empirical acceleration.
        for velocity in 3..6 {
            assert_eq!(f[(velocity, index)], 0.0);
        }
    }
}

#[test]
fn the_allan_coefficients_give_the_noise_densities() {
    let c = 299_792_458.0_f64;
    let model = ClockNoiseModel::from_allan_coefficients(2e-19, 2e-20, 10.0, 0.1);
    assert!((model.white_frequency_psd_m2_per_s / (c * c * 1e-19) - 1.0).abs() < 1e-12);
    let expected = 2.0 * core::f64::consts::PI.powi(2) * c * c * 2e-20;
    assert!((model.random_walk_frequency_psd_m2_per_s3 / expected - 1.0).abs() < 1e-12);
    assert!(model.is_plausible());
    assert!(
        !ClockNoiseModel {
            initial_bias_sigma_m: 0.0,
            ..model
        }
        .is_plausible()
    );
}
