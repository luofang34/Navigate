//! Finite-difference checks of the rows of the radiometric and
//! line-of-sight models, the Shapiro delay, and the refusals of the
//! one-way models without clock states.

use nalgebra::{SMatrix, UnitQuaternion, Vector3};

use super::*;
use crate::inertial::ClockNoiseModel;
use crate::inertial::PropagationMode;
use crate::inertial::estimate::InertialCore;
use crate::inertial::models::line_of_sight::{line_of_sight, tangent_basis};
use crate::inertial::models::radiometric::{linearize, shapiro_delay};
use crate::inertial::nominal::{ErrorVec, FULL_STATES as N, InertialNominal};
use crate::inertial::regime::Regime;
use crate::observation::{LineOfSightReference, ObservationValue};
use crate::rejection::RejectionReason;
use navigate_astrodynamics::{SpaceEnvironment, SpaceGravityModel};

const GM_SUN: f64 = 1.327_124_400_18e20;

fn clock_config() -> InertialConfig {
    InertialConfig {
        clock_model: Some(ClockNoiseModel {
            white_frequency_psd_m2_per_s: 0.01,
            random_walk_frequency_psd_m2_per_s3: 1e-4,
            initial_bias_sigma_m: 100.0,
            initial_drift_sigma_mps: 1.0,
        }),
        ..InertialConfig::default()
    }
}

fn test_core() -> InertialCore {
    let mut nominal = InertialNominal::at(
        Vector3::new(7.0e6, -2.0e6, 1.5e6),
        Vector3::new(1_000.0, 7_000.0, -500.0),
        UnitQuaternion::from_euler_angles(0.4, -0.7, 2.1),
    );
    nominal.clock_bias_m = 50.0;
    nominal.clock_drift_mps = 0.2;
    InertialCore {
        regime: Regime::space(SpaceEnvironment::inertial(
            FrameId::EARTH_J2000,
            SpaceGravityModel::Zero,
        )),
        nominal,
        p: crate::inertial::nominal::ErrorMat::<22>::identity(),
        layout: crate::inertial::StateLayout::FULL,
        state_at: t(0),
        last_rates: None,
        mag_key: None,
        correlated: crate::inertial::correlated::CorrelatedSources::default(),
        pending: None,
        mode: PropagationMode::SensedImu,
    }
}

/// Step size per error component. Each model is linear in the clock,
/// so a clock step of one unit keeps the rounding of a range of 2e7 m
/// below the tolerance.
fn epsilon(index: usize) -> f64 {
    match index {
        0..=2 => 1.0,
        3..=5 => 1e-3,
        6..=8 => 1e-6,
        20 | 21 => 1.0,
        _ => 1e-3,
    }
}

/// The finite-difference rows of `measure` at the nominal state of
/// `core`.
fn numeric_rows<const M: usize>(
    core: &InertialCore,
    measure: impl Fn(&InertialNominal) -> SMatrix<f64, M, 1>,
) -> SMatrix<f64, M, N> {
    let mut rows = SMatrix::<f64, M, N>::zeros();
    for j in 0..N {
        let mut dx = ErrorVec::<22>::zeros();
        dx[j] = epsilon(j);
        let plus = measure(
            &core
                .nominal
                .inject(&dx, &crate::inertial::StateLayout::FULL),
        );
        let minus = measure(
            &core
                .nominal
                .inject(&(-dx), &crate::inertial::StateLayout::FULL),
        );
        rows.set_column(j, &((plus - minus) / (2.0 * epsilon(j))));
    }
    rows
}

/// The predicted value of a scalar model: `z − residual`.
fn predicted(value: &ObservationValue, z: f64, nominal: &InertialNominal) -> SMatrix<f64, 1, 1> {
    let core = InertialCore {
        nominal: *nominal,
        ..test_core()
    };
    let model = linearize(&core, value, &clock_config()).expect("model");
    SMatrix::<f64, 1, 1>::new(z - model.residual[0])
}

fn radiometric_values() -> [(ObservationValue, f64); 4] {
    let station = [6.0e6, 1.0e6, 0.5e6];
    let other = [5.5e6, -2.0e6, 2.0e6];
    [
        (
            ObservationValue::Pseudorange {
                emitter_position_m: [1.5e7, 1.0e7, 2.0e7],
                pseudorange_m: 2.0e7,
                sagnac_and_media_correction_m: 3.0,
                variance_m2: 4.0,
            },
            2.0e7,
        ),
        (
            ObservationValue::PseudorangeRate {
                emitter_position_m: [1.5e7, 1.0e7, 2.0e7],
                emitter_velocity_mps: [-1_500.0, 2_000.0, 300.0],
                pseudorange_rate_mps: 100.0,
                variance_m2_per_s2: 1e-4,
            },
            100.0,
        ),
        (
            ObservationValue::TwoWayRange {
                tx_station_pos_m: station,
                rx_station_pos_m: other,
                gravitation_gm_m3_s2: Some((GM_SUN, [-1.0e10, 3.0e9, 0.0])),
                round_trip_range_m: 3.0e6,
                turnaround_delay_m: 12.0,
                variance_m2: 1.0,
            },
            3.0e6,
        ),
        (
            ObservationValue::TwoWayRangeRate {
                tx_station_pos_m: station,
                tx_station_vel_mps: [-70.0, 440.0, 0.0],
                rx_station_pos_m: other,
                rx_station_vel_mps: [150.0, 400.0, 0.0],
                round_trip_range_rate_mps: -2_000.0,
                variance_m2_per_s2: 1e-6,
            },
            -2_000.0,
        ),
    ]
}

#[test]
fn each_radiometric_row_matches_finite_differences() {
    let core = test_core();
    for (value, z) in radiometric_values() {
        let model = linearize(&core, &value, &clock_config()).expect("model");
        let numeric = numeric_rows(&core, |x| predicted(&value, z, x));
        let error = (model.h - numeric).abs().max();
        assert!(
            error < 1e-7,
            "{:?}: {error}\n{}",
            value.kind(),
            model.h - numeric
        );
    }
    // The one-way models read the clock, and the two-way models do not.
    let rows: [SMatrix<f64, 1, N>; 4] = radiometric_values()
        .map(|(value, _)| linearize(&core, &value, &clock_config()).expect("model").h);
    assert_eq!(rows[0][20], 1.0);
    assert_eq!(rows[1][21], 1.0);
    assert_eq!(
        (rows[2][20], rows[2][21], rows[3][20], rows[3][21]),
        (0.0, 0.0, 0.0, 0.0)
    );
}

#[test]
fn the_shapiro_gradient_matches_finite_differences() {
    let a = Vector3::new(1.2e11, 3.0e9, -1.0e9);
    let b = Vector3::new(-2.0e11, 1.0e10, 2.0e9);
    let sun = Vector3::zeros();
    let (delay, gradient) = shapiro_delay(&a, &b, GM_SUN, &sun).expect("finite");
    assert!(delay > 1.0e4, "{delay}");
    let mut numeric = Vector3::zeros();
    for axis in 0..3 {
        let mut d = Vector3::zeros();
        d[axis] = 1.0e4;
        let plus = shapiro_delay(&(a + d), &b, GM_SUN, &sun).expect("finite").0;
        let minus = shapiro_delay(&(a - d), &b, GM_SUN, &sun).expect("finite").0;
        numeric[axis] = (plus - minus) / 2.0e4;
    }
    assert!(
        (gradient - numeric).norm() < 1e-6 * gradient.norm(),
        "{gradient} {numeric}"
    );
    // A leg through the center of the body has no finite delay.
    let through = shapiro_delay(&a, &(-a), GM_SUN, &sun);
    assert_eq!(through, Err(RejectionReason::NonFiniteValue));
}

fn line_of_sight_values(core: &InertialCore) -> [ObservationValue; 2] {
    let star = Vector3::new(0.3, -0.5, 0.81).normalize();
    let target = Vector3::new(3.0e8, 1.0e8, -5.0e7);
    let c = core.nominal.c_b_to_f();
    let toward_target = (target - core.nominal.p).normalize();
    // Each measurement is 0.2 mrad from the prediction.
    let tilt = UnitQuaternion::from_scaled_axis(Vector3::new(2e-4, 0.0, 0.0));
    [
        ObservationValue::LineOfSight {
            measured_body_unit: (tilt * (c.transpose() * star)).into(),
            reference: LineOfSightReference::InertialDirection {
                direction_in_frame_unit: star.into(),
            },
            angular_variance_rad2: 1e-10,
        },
        ObservationValue::LineOfSight {
            measured_body_unit: (tilt * (c.transpose() * toward_target)).into(),
            reference: LineOfSightReference::TargetPosition {
                position_in_frame_m: target.into(),
            },
            angular_variance_rad2: 1e-10,
        },
    ]
}

/// The predicted body direction of `value` at `nominal`.
fn body_direction(value: &ObservationValue, nominal: &InertialNominal) -> Vector3<f64> {
    let ObservationValue::LineOfSight { reference, .. } = value else {
        return Vector3::zeros();
    };
    let frame = match reference {
        LineOfSightReference::InertialDirection {
            direction_in_frame_unit,
        } => Vector3::from(*direction_in_frame_unit),
        LineOfSightReference::TargetPosition {
            position_in_frame_m,
        } => (Vector3::from(*position_in_frame_m) - nominal.p).normalize(),
    };
    nominal.c_b_to_f().transpose() * frame
}

#[test]
fn each_line_of_sight_row_matches_finite_differences_with_the_sign_of_the_attitude() {
    let core = test_core();
    for value in line_of_sight_values(&core) {
        let (model, cosine) = line_of_sight(&core, &value).expect("model");
        assert!(cosine > 0.999_999);
        let basis = tangent_basis(&body_direction(&value, &core.nominal));
        assert!((basis.transpose() * basis - nalgebra::Matrix2::identity()).norm() < 1e-14);
        let numeric = numeric_rows(&core, |x| basis.transpose() * body_direction(&value, x));
        let error = (model.h - numeric).abs().max();
        assert!(error < 1e-8, "{error}\n{}", model.h - numeric);
        // The residual norm is the sine of the angle between the measured
        // and the predicted directions.
        let ObservationValue::LineOfSight {
            measured_body_unit, ..
        } = value
        else {
            continue;
        };
        let sine = Vector3::from(measured_body_unit)
            .cross(&body_direction(&value, &core.nominal))
            .norm();
        assert!(sine > 1e-5);
        assert!((model.residual.norm() - sine).abs() < 1e-12);
    }
    // A star direction updates the attitude only. A target direction
    // also updates the position across the line of sight.
    let [star, target] =
        line_of_sight_values(&core).map(|value| line_of_sight(&core, &value).expect("model").0.h);
    assert_eq!(star.fixed_view::<2, 3>(0, 0).norm(), 0.0);
    assert!(target.fixed_view::<2, 3>(0, 0).norm() > 1e-9);
}

#[test]
fn the_one_way_models_need_the_clock_states() {
    let core = test_core();
    for (value, _) in radiometric_values().into_iter().take(2) {
        assert_eq!(
            linearize(&core, &value, &InertialConfig::default()).map(|m| m.residual),
            Err(RejectionReason::ClockStateDisabled { kind: value.kind() })
        );
    }
    for (value, _) in radiometric_values().into_iter().skip(2) {
        assert!(linearize(&core, &value, &InertialConfig::default()).is_ok());
    }
}
