//! IMU increments: coning motion against the exact rigid-body rotation,
//! the rotation correction of the velocity increment, and the split of
//! an increment by a delayed fix.

use core::f64::consts::TAU;

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{CartesianPosition, CartesianVelocity, DurationNanos, SymmetricCov3};

use super::*;
use crate::inertial::ImuIncrement;
use crate::inertial::nominal::{rotation_vector, to_contract};
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;
use navigate_astrodynamics::{SpaceEnvironment, SpaceGravityModel};

/// A filter at rest at the origin of an inertial frame without gravity.
fn free_space_filter(attitude: &UnitQuaternion<f64>) -> InertialNavigationFilter {
    let env = SpaceEnvironment::inertial(FrameId::EARTH_J2000, SpaceGravityModel::Zero);
    let mut filter = InertialNavigationFilter::new_space(InertialConfig::default(), CLOCK, env);
    filter
        .initialize_space(
            CartesianPosition::new(0.0, 0.0, 0.0),
            CartesianVelocity::new(0.0, 0.0, 0.0),
            to_contract(attitude),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(0),
            inertial(),
        )
        .expect("plausible");
    filter
}

fn increment(
    sequence: u32,
    at_ms: u64,
    dt_s: f64,
    d_theta: Vector3<f64>,
    d_v: Vector3<f64>,
) -> ImuIncrement {
    ImuIncrement::new(
        stamp(IMU_SOURCE, sequence, at_ms),
        inertial(),
        dt_s,
        d_theta.into(),
        d_v.into(),
    )
    .with_frame(FrameId::EARTH_J2000)
}

const CONE_RATE: f64 = TAU;
const CONE_HALF_ANGLE: f64 = 0.02;
const CONING_DT_S: f64 = 0.01;
const CONING_STEPS: u32 = 500;

/// The classical coning motion `C(t) = R_z(Ωt)·R_x(β)·R_z(−Ωt)`.
fn cone(time_s: f64) -> UnitQuaternion<f64> {
    let precession = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), CONE_RATE * time_s);
    let tilt = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), CONE_HALF_ANGLE);
    precession * tilt * precession.inverse()
}

/// The integral of the body rate `ω_b = Ω·(Cᵀ·z − z)` over one step, as
/// a gyroscope without the coning correction gives it.
fn raw_rotation(start_s: f64, end_s: f64) -> Vector3<f64> {
    let (sin_b, cos_b) = CONE_HALF_ANGLE.sin_cos();
    let angle = |time_s: f64| CONE_RATE * time_s;
    Vector3::new(
        sin_b * (angle(end_s).cos() - angle(start_s).cos()),
        sin_b * (angle(end_s).sin() - angle(start_s).sin()),
        CONE_RATE * (cos_b - 1.0) * (end_s - start_s),
    )
}

#[test]
fn coning_increments_follow_the_exact_rotation() {
    let mut with_increments = free_space_filter(&cone(0.0));
    let mut with_rates = free_space_filter(&cone(0.0));
    for k in 1..=CONING_STEPS {
        let (start_s, end_s) = (f64::from(k - 1) * CONING_DT_S, f64::from(k) * CONING_DT_S);
        let at_ms = u64::from(k) * 10;
        let exact = rotation_vector(&(cone(start_s).inverse() * cone(end_s)));
        let step = increment(k, at_ms, CONING_DT_S, exact, Vector3::zeros());
        assert_accepted(with_increments.ingest_imu_increment(&step, t(at_ms)));
        let mean_rate = raw_rotation(start_s, end_s) / CONING_DT_S;
        let sample = imu_in(
            k,
            at_ms,
            ([0.0; 3], mean_rate.into()),
            CONING_DT_S,
            FrameId::EARTH_J2000,
        );
        assert_accepted(with_rates.ingest_imu(&sample, t(at_ms)));
    }
    let truth = cone(f64::from(CONING_STEPS) * CONING_DT_S);
    let increment_error = core(&with_increments).nominal.q.angle_to(&truth);
    let rate_error = core(&with_rates).nominal.q.angle_to(&truth);
    assert!(increment_error < 1e-11, "{increment_error}");
    // Without the coning correction, the attitude drifts about the cone
    // axis.
    assert!(rate_error > 1e-7, "{rate_error}");
    assert!(
        rate_error > 1e3 * increment_error,
        "{rate_error} vs {increment_error}"
    );
}

const SPIN_RATE: f64 = 0.5;
const THRUST_MPS2: f64 = 0.3;
const SPIN_DT_S: f64 = 0.02;
const SPIN_STEPS: u32 = 200;

/// The velocity increment of a body that spins about `z` at a constant
/// rate under a specific force fixed in the inertial axes along `x`:
/// `∫ R_z(−ωτ)·f dτ` in the body axes, without the rotation correction.
fn spin_velocity(start_s: f64, end_s: f64) -> Vector3<f64> {
    let angle = |time_s: f64| SPIN_RATE * time_s;
    THRUST_MPS2 / SPIN_RATE
        * Vector3::new(
            angle(end_s).sin() - angle(start_s).sin(),
            angle(end_s).cos() - angle(start_s).cos(),
            0.0,
        )
}

#[test]
fn the_rotation_correction_keeps_the_velocity_in_the_inertial_direction() {
    let mut filter = free_space_filter(&UnitQuaternion::identity());
    for k in 1..=SPIN_STEPS {
        let (start_s, end_s) = (f64::from(k - 1) * SPIN_DT_S, f64::from(k) * SPIN_DT_S);
        let at_ms = u64::from(k) * 20;
        let d_theta = Vector3::new(0.0, 0.0, SPIN_RATE * SPIN_DT_S);
        let step = increment(k, at_ms, SPIN_DT_S, d_theta, spin_velocity(start_s, end_s));
        assert_accepted(filter.ingest_imu_increment(&step, t(at_ms)));
    }
    let time_s = f64::from(SPIN_STEPS) * SPIN_DT_S;
    let state = core(&filter);
    let expected_v = Vector3::new(THRUST_MPS2 * time_s, 0.0, 0.0);
    let expected_p = 0.5 * expected_v * time_s;
    let spin = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), SPIN_RATE * time_s);
    assert!(state.nominal.q.angle_to(&spin) < 1e-12);
    // Without the correction the velocity turns by `½·ω·Δt` = 5 mrad.
    let direction = state.nominal.v.y.atan2(state.nominal.v.x);
    assert!(direction.abs() < 1e-4, "{direction}");
    assert!((state.nominal.v - expected_v).norm() < 1e-4 * expected_v.norm());
    assert!((state.nominal.p - expected_p).norm() < 1e-3 * expected_p.norm());
}

#[test]
fn increments_at_rest_in_the_rotating_frame_of_the_earth_stay_at_rest() {
    // The gyroscope measures the Earth rotation and the accelerometer
    // measures the reaction to gravity. The exact frame rotation and the
    // rotation correction cancel these terms, so the state stays at
    // rest. A wrong sign of either term turns the attitude at 2ω.
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    let start = *core(&filter);
    let (f, w) = rest_rates(&filter);
    let dt_s = 0.1;
    for k in 1..=1_000_u32 {
        let at_ms = u64::from(k) * 100;
        let step = ImuIncrement::new(
            stamp(IMU_SOURCE, k, at_ms),
            inertial(),
            dt_s,
            (Vector3::from(w) * dt_s).into(),
            (Vector3::from(f) * dt_s).into(),
        );
        assert_accepted(filter.ingest_imu_increment(&step, t(at_ms)));
    }
    let state = core(&filter);
    let speed = state.nominal.v.norm();
    let turn = state.nominal.q.angle_to(&start.nominal.q);
    let drift = (state.nominal.p - start.nominal.p).norm();
    assert!(speed < 1e-6, "{speed}");
    assert!(turn < 1e-9, "{turn}");
    assert!(drift < 1e-3, "{drift}");
}

/// Runs increments, mean-rate samples, and one fix on a WGS84 surface.
/// Each event is `(kind, at_ms, now_ms)`: 0 is an increment of one
/// second, 1 is a half-second sample with the mean rates of the second
/// increment, and 2 is the fix.
fn run_increments(events: &[(u8, u64, u64)]) -> InertialCore {
    let config = InertialConfig {
        fusion: crate::config::FusionConfig {
            staleness_bound: DurationNanos::from_millis(5_000),
            ..InertialConfig::default().fusion
        },
        ..InertialConfig::default()
    };
    let mut filter = surface_filter(config, PlanetaryBody::EARTH_WGS84, earth_origin());
    let (f, w) = rest_rates(&filter);
    let d_theta = Vector3::from(w) + Vector3::new(0.02, -0.01, 0.05);
    let d_v = Vector3::from(f) + Vector3::new(0.4, -0.2, 0.1);
    let fix = Observation::new(
        stamp(1, 1, 1_500),
        ObservationValue::PositionFix {
            position: earth_origin(),
            covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
        },
        gnss(),
    );
    for (sequence, &(kind, at_ms, now_ms)) in (1_u32..).zip(events) {
        let outcome = match kind {
            0 => {
                let step = ImuIncrement::new(
                    stamp(IMU_SOURCE, sequence, at_ms),
                    inertial(),
                    1.0,
                    d_theta.into(),
                    d_v.into(),
                );
                filter.ingest_imu_increment(&step, t(now_ms))
            }
            1 => filter.ingest_imu(
                &imu(sequence, at_ms, (d_v.into(), d_theta.into()), 0.5),
                t(now_ms),
            ),
            _ => filter.ingest(&fix, t(now_ms)),
        };
        assert_accepted(outcome);
    }
    *core(&filter)
}

#[test]
fn a_delayed_fix_inside_an_increment_splits_it_with_the_mean_rates() {
    let split = run_increments(&[
        (0, 1_000, 1_000),
        (1, 1_500, 1_500),
        (2, 1_500, 1_500),
        (1, 2_000, 2_000),
        (0, 3_000, 3_000),
    ]);
    let delayed = run_increments(&[
        (0, 1_000, 1_000),
        (0, 2_000, 2_000),
        (0, 3_000, 3_000),
        (2, 1_500, 3_000),
    ]);
    assert!((split.nominal.p - delayed.nominal.p).norm() < 1e-9);
    assert!((split.nominal.v - delayed.nominal.v).norm() < 1e-12);
    assert!(split.nominal.q.angle_to(&delayed.nominal.q) < 1e-12);
    assert!((split.p - delayed.p).abs().max() < 1e-12);
    // Without the fix, the full increment and the two mean-rate halves
    // differ only at second order.
    let full = run_increments(&[(0, 1_000, 1_000), (0, 2_000, 2_000)]);
    let halves = run_increments(&[(0, 1_000, 1_000), (1, 1_500, 1_500), (1, 2_000, 2_000)]);
    let gap = (full.nominal.v - halves.nominal.v).norm();
    assert!(gap > 0.0 && gap < 1e-2, "{gap}");
}

#[test]
fn implausible_or_undeclared_increments_are_refused() {
    let mut filter = free_space_filter(&UnitQuaternion::identity());
    let bad = increment(1, 10, 0.0, Vector3::zeros(), Vector3::zeros());
    assert_eq!(
        filter.ingest_imu_increment(&bad, t(10)),
        IngestOutcome::Rejected(RejectionReason::NonFiniteValue)
    );
    let undeclared = ImuIncrement::new(
        stamp(IMU_SOURCE, 2, 10),
        inertial(),
        0.01,
        [0.0; 3],
        [0.0; 3],
    );
    assert_eq!(
        filter.ingest_imu_increment(&undeclared, t(10)),
        IngestOutcome::Rejected(RejectionReason::MissingFrame {
            expected: FrameId::EARTH_J2000
        })
    );
    assert_eq!(filter.rejections().non_finite_value, 1);
    assert_eq!(filter.rejections().missing_frame, 1);
}
