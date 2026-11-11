//! A ground calibration before an in-air start: a declared start that
//! carries the gyroscope bias of the calibration as its prior.

use nalgebra::Vector3;

use super::declared::{CRAB_SIGMA_RAD, STATED_AT, cruise, declare, navigator, statement};
use super::moving::attitude_error;
use super::path::Vehicle;
use super::rig::{Noise, at};
use crate::{DeclaredStart, Navigator};

/// A declared start that carries the gyroscope bias of a ground
/// calibration keeps its heading as the bias drifts it without one.
#[test]
fn a_declared_start_takes_the_bias_of_a_ground_calibration() {
    // A seed whose turn-on bias about the vertical is at least 0.5 mrad/s.
    let seed = (0..64_u64)
        .find(|seed| {
            Vehicle::new(cruise(0.0, 0.0), [0.0, 0.0], *seed, 1.0)
                .gyro_bias()
                .z
                .abs()
                > 5e-4
        })
        .unwrap();
    // The residual of a 60 s rest: 0.0009 °/s.
    let residual = 0.0009_f64.to_radians();
    let drift = |with_prior: bool| {
        let mut vehicle = Vehicle::new(cruise(0.4, 0.0), [0.0, 0.0], seed, 1.0);
        let bias = vehicle.gyro_bias();
        let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
        let accel = 0.02_f64.powi(2);
        let gyro = residual * residual;
        let prior = crate::ImuBiasPrior::new(
            [0.0; 3],
            navigate_contract::SymmetricCov3::from_diagonal(accel, accel, accel),
            (bias + Vector3::new(0.0, 0.0, residual)).into(),
            navigate_contract::SymmetricCov3::from_diagonal(gyro, gyro, gyro),
        );
        let (truth, solution) = declare(
            &mut navigator,
            &mut vehicle,
            |truth| DeclaredStart {
                imu_bias_prior: with_prior.then_some(prior),
                ..statement(truth, 55.0, &mut Noise::new(9), 0.0)
            },
            0,
        );
        let start = attitude_error(&truth, &solution.unwrap()).z;
        let mut sink = Vec::new();
        while vehicle.step < STATED_AT + 3_000 {
            navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
        }
        let end = attitude_error(
            &vehicle.truth(),
            &navigator.solution(at(vehicle.step)).unwrap(),
        )
        .z;
        ((end - start).abs(), bias.z.abs())
    };
    let (with, bias) = drift(true);
    let (without, _) = drift(false);
    // 30 s: the bias turns the heading by its size times 30 s; the
    // residual by 0.5 mrad.
    assert!(without > 0.5 * bias * 30.0, "{without} {bias}");
    assert!(with < 0.2 * without, "{with} {without}");
}
