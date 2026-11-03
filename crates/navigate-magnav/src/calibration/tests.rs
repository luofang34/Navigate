use std::vec::Vec;

use nalgebra::{Rotation3, Vector3};

use super::*;

/// The core field in the local NED axes, in tesla.
const EARTH_FIELD_NED_T: [f64; 3] = [2.0e-5, 1.0e-6, 4.5e-5];
const DT_S: f64 = 0.1;

fn true_coefficients() -> [f64; TOLLES_LAWSON_TERMS] {
    [
        1.2e-7, -6.0e-8, 3.0e-8, // permanent, T
        2.0e-3, -1.0e-3, 5.0e-4, 1.5e-3, -2.0e-3, 2.5e-3, // induced
        3.0e-2, -1.0e-2, 2.0e-2, -1.5e-2, 1.0e-2, 5.0e-3, 2.0e-2, -2.5e-2, 1.0e-2, // eddy, s
    ]
}

fn interference(b: [f64; 3], b_dot: [f64; 3]) -> f64 {
    let row = tolles_lawson_basis(b, b_dot).expect("defined");
    row.iter()
        .zip(true_coefficients())
        .map(|(a, c)| a * c)
        .sum()
}

/// The body-axis field for roll, pitch, and heading in radians.
fn body_field(roll: f64, pitch: f64, heading: f64) -> Vector3<f64> {
    let body_to_ned = Rotation3::from_euler_angles(roll, pitch, heading);
    body_to_ned.inverse() * Vector3::from(EARTH_FIELD_NED_T)
}

/// Samples of a maneuver as `(B, Ḃ)` pairs. `phase` shifts the
/// oscillations, so a validation run differs from the calibration run.
fn maneuver(seconds: f64, phase: f64, amplitude: f64) -> Vec<([f64; 3], [f64; 3])> {
    let steps = (seconds / DT_S) as usize;
    let attitude = |time_s: f64| {
        let heading = core::f64::consts::FRAC_PI_2 * (time_s / 60.0).floor() + 0.3 * phase;
        body_field(
            amplitude * 0.17 * (0.9 * time_s + phase).sin(),
            amplitude * 0.09 * (0.6 * time_s + 2.0 * phase).sin(),
            heading + amplitude * 0.09 * (0.4 * time_s + 3.0 * phase).sin(),
        )
    };
    (0..steps)
        .map(|step| {
            let time_s = step as f64 * DT_S;
            let b = attitude(time_s);
            let b_dot = (attitude(time_s + 0.5 * DT_S) - attitude(time_s - 0.5 * DT_S)) / DT_S;
            (b.into(), b_dot.into())
        })
        .collect()
}

fn calibrated() -> OnlineTlCalibrator {
    let mut calibrator = OnlineTlCalibrator::new(TlCalibratorConfig::default()).expect("valid");
    for (b, b_dot) in maneuver(240.0, 0.0, 1.0) {
        let outcome = calibrator.observe(b, b_dot, interference(b, b_dot), DT_S);
        assert!(outcome.is_ok());
    }
    calibrator
}

fn demeaned(values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| v - mean).collect()
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

#[test]
fn the_basis_has_the_tolles_lawson_structure() {
    let b = [2.0e-5, -1.0e-5, 4.0e-5];
    let row = tolles_lawson_basis(b, [0.0; 3]).expect("defined");
    let magnitude = Vector3::from(b).norm();
    let permanent = Vector3::new(row[0], row[1], row[2]);
    assert!((permanent.norm() - 1.0).abs() < 1e-12);
    // The diagonal induced terms `|B|·cᵢ²` sum to `|B|`.
    assert!((row[3] + row[6] + row[8] - magnitude).abs() < 1e-18);
    assert!(row[9..].iter().all(|v| *v == 0.0));
    let eddy = tolles_lawson_basis(b, [1e-6, 0.0, 0.0]).expect("defined");
    assert!((eddy[9] - row[0] * 1e-6).abs() < 1e-20);
    assert!(tolles_lawson_basis([0.0; 3], [0.0; 3]).is_none());
    assert!(tolles_lawson_basis([f64::NAN, 0.0, 1e-5], [0.0; 3]).is_none());
}

#[test]
fn figure_of_merit_maneuvers_recover_the_aircraft_interference() {
    let calibrator = calibrated();
    let validation = maneuver(120.0, 1.3, 0.8);
    let truth: Vec<f64> = validation
        .iter()
        .map(|(b, b_dot)| interference(*b, *b_dot))
        .collect();
    let learned: Vec<f64> = validation
        .iter()
        .map(|(b, b_dot)| calibrator.interference_t(*b, *b_dot).expect("defined"))
        .collect();
    let errors: Vec<f64> = truth.iter().zip(&learned).map(|(t, l)| l - t).collect();
    // A constant interference is the same as the residual bias, so the
    // comparison removes the mean.
    let error_rms = rms(&demeaned(&errors));
    let signal_rms = rms(&demeaned(&truth));
    assert!(signal_rms > 5e-9, "{signal_rms}");
    assert!(error_rms < 0.05 * signal_rms, "{error_rms} vs {signal_rms}");
    assert!(error_rms < 1e-9, "{error_rms}");
    // The eddy-current terms are well excited and come back one by one.
    let coefficients = calibrator.coefficients();
    for (index, (learned, truth)) in coefficients
        .iter()
        .zip(true_coefficients())
        .enumerate()
        .skip(9)
    {
        assert!(
            (learned - truth).abs() < 0.2 * truth.abs().max(1e-2),
            "{index}: {learned} vs {truth}"
        );
    }
}

#[test]
fn straight_and_level_flight_freezes_the_coefficients() {
    let mut gated = calibrated();
    let mut ungated = gated.clone();
    ungated.config.min_b_dot_norm_t_per_s = 0.0;
    let learned = gated.coefficients();
    let b: [f64; 3] = body_field(0.0, 0.0, 0.4).into();
    let b_dot = [0.0; 3];
    for step in 0..600 {
        let time_s = f64::from(step) * DT_S;
        // The crustal anomaly changes along the track.
        let anomaly = 2e-8 * (0.05 * time_s).sin() + 5e-9 * (0.31 * time_s).cos();
        let residual = interference(b, b_dot) + anomaly;
        assert_eq!(
            gated.observe(b, b_dot, residual, DT_S),
            Ok(TlUpdate::Frozen)
        );
        assert_eq!(
            ungated.observe(b, b_dot, residual, DT_S),
            Ok(TlUpdate::Updated)
        );
    }
    assert_eq!(gated.coefficients(), learned);
    let drift: f64 = ungated
        .coefficients()
        .iter()
        .zip(learned)
        .take(9)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        drift > 1e-10,
        "the ungated estimator absorbs the anomaly: {drift}"
    );
    // The frozen calibrator still compensates with the learned terms.
    let measured = 5e-5 + interference(b, b_dot);
    let compensated = gated.compensate(b, b_dot, measured).expect("defined");
    assert!((compensated - 5e-5).abs() < 2e-9, "{compensated}");
}

#[test]
fn implausible_inputs_are_refused_without_a_change() {
    assert_eq!(
        OnlineTlCalibrator::new(TlCalibratorConfig {
            measurement_variance_t2: 0.0,
            ..TlCalibratorConfig::default()
        }),
        Err(TlError::InvalidConfig)
    );
    let mut calibrator = OnlineTlCalibrator::new(TlCalibratorConfig::default()).expect("valid");
    let before = calibrator.clone();
    assert_eq!(
        calibrator.observe([0.0; 3], [0.0; 3], 0.0, DT_S),
        Err(TlError::NonFiniteValue)
    );
    assert_eq!(
        calibrator.observe([1e-5, 0.0, 4e-5], [1e-6; 3], f64::NAN, DT_S),
        Err(TlError::NonFiniteValue)
    );
    assert_eq!(calibrator, before);
    assert_eq!(calibrator.residual_bias_t(), 0.0);
}
