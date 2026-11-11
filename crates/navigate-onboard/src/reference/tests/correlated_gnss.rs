//! A GNSS receiver with a time-correlated error at 10 Hz on a turning
//! route: the 3-D position NEES of the reference navigator over many
//! seeds, with the correlated part declared in the budget and with the
//! whole error declared as white. The truth runs through an inertial
//! filter of Navigate fed with exact increments, so the truth and the
//! navigator integrate with the same mechanization.

use navigate_contract::{NedVelocity, SourceId};

use super::truth::{
    CLOCK, Draws, IMU, Imu, STEP_S, Truth, at, imu_budget, moved, ned_offset, nees,
};
use crate::sensor::SensorSpec;
use crate::{
    ErrorBudget, GnssCorrelatedError, Navigator, ReferenceNavigator, SensorData, SensorInput,
    SensorKind, SensorSuite,
};

const GNSS: SourceId = SourceId::new(2);
/// The turn rate of a 25° bank at 40 m/s.
const TURN_RAD_S: f64 = 0.114;

/// The heading rate of the route: 90° turns at the times of `TURNS_S`.
fn turn_rate(t_s: f64) -> f64 {
    TURNS_S
        .iter()
        .find(|(begin, _)| {
            (*begin..begin + core::f64::consts::FRAC_PI_2 / TURN_RAD_S).contains(&t_s)
        })
        .map_or(0.0, |(_, sign)| sign * TURN_RAD_S)
}
/// The receiver of the browser chain: 10 Hz fixes, white 0.6 m
/// horizontal and 1.0 m vertical, correlated 1.2 m and 2.0 m with a
/// correlation time of 300 s, and 0.05 m/s Doppler velocity.
const WHITE_M: [f64; 3] = [0.6, 0.6, 1.0];
const CORRELATED_M: [f64; 3] = [1.2, 1.2, 2.0];
const TAU_S: f64 = 300.0;
const VELOCITY_SIGMA: f64 = 0.05;

const SEEDS: u32 = if cfg!(debug_assertions) { 1 } else { 12 };
const DURATION_S: f64 = if cfg!(debug_assertions) {
    60.0
} else {
    1_200.0
};
const TURNS_S: &[(f64, f64)] = if cfg!(debug_assertions) {
    &[(20.0, 1.0)]
} else {
    &[(300.0, 1.0), (620.0, -1.0), (900.0, 1.0)]
};
const REPORTS_S: &[f64] = if cfg!(debug_assertions) {
    &[30.0, 60.0]
} else {
    &[300.0, 600.0, 900.0, 1_200.0]
};
/// The 0.1 % and 99.9 % points of χ²(3·SEEDS)/SEEDS.
const BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.024, 16.3)
} else {
    (1.28, 5.67)
};

fn suite(correlated: bool) -> SensorSuite {
    let (accel, gyro, accel_bias, gyro_bias) = imu_budget();
    let spec = |id, kind, budget| SensorSpec {
        id,
        kind,
        imager: None,
        mount: None,
        rate_hz: 10.0,
        budget,
    };
    let white = |i: usize| {
        if correlated {
            WHITE_M[i]
        } else {
            WHITE_M[i].hypot(CORRELATED_M[i])
        }
    };
    let gnss = ErrorBudget::Gnss {
        horizontal_sigma_m: white(0),
        vertical_sigma_m: white(2),
        velocity_sigma_m_s: VELOCITY_SIGMA,
        correlated: correlated.then_some(GnssCorrelatedError {
            tau_s: TAU_S,
            horizontal_sigma_m: CORRELATED_M[0],
            vertical_sigma_m: CORRELATED_M[2],
        }),
    };
    SensorSuite::new(vec![
        spec(
            IMU,
            SensorKind::Imu,
            ErrorBudget::Imu {
                accel_noise_mps2_sqrt_hz: accel,
                gyro_noise_rad_s_sqrt_hz: gyro,
                accel_bias,
                gyro_bias,
            },
        ),
        spec(GNSS, SensorKind::Gnss, gnss),
    ])
    .unwrap()
}

/// The 3-D position NEES of one run at each report time.
fn run(seed: u32, correlated: bool) -> Vec<f64> {
    let mut draws = Draws {
        seed: u64::from(seed),
        index: 0,
    };
    let mut truth = Truth::new(turn_rate);
    let mut navigator = ReferenceNavigator::new(suite(correlated), CLOCK, None).unwrap();
    let imu = Imu::align(&mut navigator, &mut draws);
    let mut correlated_error = draws.vec3(CORRELATED_M);
    let rho = (-STEP_S / TAU_S).exp();
    let innovation = CORRELATED_M.map(|s| s * (1.0 - rho * rho).sqrt());
    let mut out = Vec::new();
    let steps = (DURATION_S / STEP_S).round() as u32;
    for step in 1..=steps {
        imu.step(&mut truth, &mut navigator, &mut draws, step);
        correlated_error = correlated_error * rho + draws.vec3(innovation);
        let error = correlated_error + draws.vec3(WHITE_M);
        let v = truth.state.velocity;
        let dv = draws.vec3([VELOCITY_SIGMA; 3]);
        let fix = SensorInput {
            sensor: GNSS,
            acquired_at: at(step),
            delivered_at: at(step),
            data: SensorData::GnssFix {
                position: moved(&truth.state.position, &error),
                velocity: Some(NedVelocity::new(
                    v.north_mps + dv.x,
                    v.east_mps + dv.y,
                    v.down_mps + dv.z,
                )),
                position_cov: None,
                velocity_cov: None,
            },
        };
        navigator.ingest(&fix, &mut ()).unwrap();
        if REPORTS_S
            .iter()
            .any(|r| ((r / STEP_S).round() as u32) == step)
        {
            let solution = navigator.solution(at(step)).unwrap();
            let error = ned_offset(&truth.state.position, &solution.position);
            out.push(nees(&error, &solution.position_cov));
        }
    }
    out
}

fn mean_nees(correlated: bool) -> Vec<f64> {
    let mut sum = vec![0.0; REPORTS_S.len()];
    for seed in 0..SEEDS {
        for (total, value) in sum.iter_mut().zip(run(seed, correlated)) {
            *total += value;
        }
    }
    sum.into_iter()
        .map(|total| total / f64::from(SEEDS))
        .collect()
}

#[test]
fn a_correlated_gnss_budget_gives_the_navigator_a_correlated_block() {
    for (correlated, dimension) in [(true, 25), (false, 22)] {
        let navigator = Box::new(ReferenceNavigator::new(suite(correlated), CLOCK, None).unwrap());
        assert_eq!(navigator.filter().layout().dimension(), dimension);
    }
}

#[test]
fn an_implausible_correlated_gnss_budget_is_refused() {
    let mut sensors = suite(true).sensors().to_vec();
    for sensor in &mut sensors {
        if let ErrorBudget::Gnss { correlated, .. } = &mut sensor.budget {
            *correlated = Some(GnssCorrelatedError {
                tau_s: 0.0,
                horizontal_sigma_m: 1.0,
                vertical_sigma_m: 1.0,
            });
        }
    }
    assert!(SensorSuite::new(sensors).is_err());
}

#[test]
fn monte_carlo_a_declared_correlated_gnss_error_at_10_hz_keeps_the_position_nees_in_band() {
    let nees = mean_nees(true);
    assert!(
        nees.iter().all(|v| (BAND.0..=BAND.1).contains(v)),
        "NEES {nees:?}"
    );
}

#[test]
fn monte_carlo_a_correlated_gnss_error_declared_as_white_at_10_hz_is_over_confident() {
    let nees = mean_nees(false);
    // One debug run cannot show it.
    if !cfg!(debug_assertions) {
        assert!(nees.iter().skip(1).all(|v| *v > BAND.1), "NEES {nees:?}");
    }
}
