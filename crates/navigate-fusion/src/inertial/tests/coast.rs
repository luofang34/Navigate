//! The force-model coast: the transition against finite differences,
//! the mode gates, the impulsive burn, and the replay of a delayed fix
//! through a burn.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{CartesianPosition, CartesianVelocity, DurationNanos, SymmetricCov3};

use super::*;
use crate::inertial::coast::{coast, coast_transition};
use crate::inertial::nominal::{ErrorMat, ErrorVec, N, VEL};
use crate::inertial::{ClockNoiseModel, ForceModelCoast, PropagationMode};
use crate::kalman::FogmBias;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;
use navigate_astrodynamics::{ExponentialDrag, SpaceEnvironment, SpaceGravityModel, ThirdBodySet};

const EARTH: PlanetaryBody = PlanetaryBody::EARTH_WGS84;

fn dense_drag(rotation: f64) -> ExponentialDrag {
    ExponentialDrag {
        cd: 2.2,
        area_to_mass_m2_per_kg: 0.02,
        reference_density_kg_m3: 1e-9,
        reference_altitude_m: 400e3,
        scale_height_m: 60e3,
        equatorial_radius_m: 6_378_137.0,
        atmosphere_rotation_rad_s: [0.0, 0.0, rotation],
    }
}

fn inertial_env() -> SpaceEnvironment {
    SpaceEnvironment::inertial(
        FrameId::EARTH_J2000,
        SpaceGravityModel::MultiBody {
            central: EARTH.central_gravity(),
            third_bodies: ThirdBodySet::EMPTY,
        },
    )
    .with_drag(dense_drag(EARTH.rotation_rate_rad_per_s))
}

fn full_config() -> InertialConfig {
    InertialConfig {
        dmc: Some(FogmBias::new(600.0, 1e-6)),
        clock_model: Some(ClockNoiseModel {
            white_frequency_psd_m2_per_s: 0.01,
            random_walk_frequency_psd_m2_per_s3: 1e-4,
            initial_bias_sigma_m: 100.0,
            initial_drift_sigma_mps: 1.0,
        }),
        ..InertialConfig::default()
    }
}

fn settings() -> ForceModelCoast {
    ForceModelCoast {
        body_rate_rad_s: [1e-3, -2e-3, 5e-4],
        acceleration_noise_sigma_mps2_sqrt_hz: 1e-5,
        body_rate_noise_sigma_rad_s_sqrt_hz: 1e-6,
        ..ForceModelCoast::new(10.0)
    }
}

fn leo_nominal() -> crate::inertial::nominal::InertialNominal {
    // About 390 km above the equatorial radius.
    let mut x = crate::inertial::nominal::InertialNominal::at(
        Vector3::new(4.2e6, -3.7e6, 3.8e6),
        Vector3::new(4_500.0, 5_200.0, -800.0),
        UnitQuaternion::from_euler_angles(0.3, -0.4, 1.2),
    );
    x.a_dmc = Vector3::new(3e-6, -1e-6, 2e-6);
    x.clock_bias_m = 40.0;
    x.clock_drift_mps = 0.3;
    x
}

fn epsilon(index: usize) -> f64 {
    match index {
        0..=2 => 1.0,
        3..=5 => 1e-3,
        6..=8 => 1e-4,
        9..=11 => 1e-3,
        12..=14 => 1e-4,
        15 => 1e-2,
        16 => 1e-12,
        // The empirical acceleration moves the position by `½·a·Δt²`.
        // The step keeps the rounding of the position below the
        // tolerance.
        17..=19 => 1e-2,
        _ => 1e-2,
    }
}

#[test]
fn the_coast_transition_matches_finite_differences_in_both_frames() {
    let rotating = SpaceEnvironment::from_body(EARTH).with_drag(dense_drag(0.0));
    let config = full_config();
    for env in [inertial_env(), rotating] {
        let model = env.force_model();
        let x = leo_nominal();
        let dt_s = 10.0;
        let (next, phi, q_d) =
            coast_transition(&x, dt_s, &model, &settings(), &config).expect("finite");
        let mut numeric = ErrorMat::zeros();
        for j in 0..N {
            let mut dx = ErrorVec::zeros();
            dx[j] = epsilon(j);
            let plus = coast(&x.inject(&dx), dt_s, &model, &settings(), &config).0;
            let minus = coast(&x.inject(&(-dx)), dt_s, &model, &settings(), &config).0;
            let column =
                (error_between(&plus, &next) - error_between(&minus, &next)) / (2.0 * epsilon(j));
            numeric.set_column(j, &column);
        }
        let error = (phi - numeric).abs().max();
        // The midpoint linearization drops the change of the gravity
        // gradient over the step, which is of the order of `Ġ·Δt⁵`.
        assert!(error < 1e-5, "{error}\n{}", phi - numeric);
        // The drag couples the velocity errors, and the empirical
        // acceleration enters the velocity.
        assert!((phi[(VEL, VEL)] - 1.0).abs() > 1e-6);
        assert!(phi[(VEL, 17)] > 9.0);
        let floor = -1e-12 * q_d.abs().max();
        assert!(q_d.symmetric_eigenvalues().iter().all(|e| *e > floor));
    }
}

fn space_filter(config: InertialConfig) -> InertialNavigationFilter {
    let config = InertialConfig {
        fusion: crate::config::FusionConfig {
            staleness_bound: DurationNanos::from_millis(600_000),
            ..config.fusion
        },
        ..config
    };
    let mut filter = InertialNavigationFilter::new_space(config, CLOCK, inertial_env());
    let x = leo_nominal();
    filter
        .initialize_space(
            CartesianPosition::new(x.p.x, x.p.y, x.p.z),
            CartesianVelocity::new(x.v.x, x.v.y, x.v.z),
            identity_attitude(),
            SymmetricCov3::from_diagonal(100.0, 100.0, 100.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(0),
            gnss(),
        )
        .expect("plausible");
    filter
}

#[test]
fn each_mode_refuses_the_events_of_the_other_mode() {
    let mut filter = space_filter(full_config());
    let burn_cov = SymmetricCov3::from_diagonal(1e-4, 1e-4, 1e-4);
    assert_eq!(
        filter.apply_impulsive_burn([1.0, 0.0, 0.0], burn_cov, t(0)),
        Err(RejectionReason::PropagationModeMismatch)
    );
    assert_eq!(
        filter.coast_to(t(1_000)),
        Err(RejectionReason::PropagationModeMismatch)
    );
    let mode = PropagationMode::ForceModel(settings());
    assert!(filter.set_propagation_mode(mode, t(0)).is_ok());
    assert_eq!(filter.propagation_mode(), mode);
    let sample = imu_in(1, 1_000, ([0.0; 3], [0.0; 3]), 1.0, FrameId::EARTH_J2000);
    assert_eq!(
        filter.ingest_imu(&sample, t(1_000)),
        IngestOutcome::Rejected(RejectionReason::PropagationModeMismatch)
    );
    assert_eq!(filter.rejections().propagation_mode_mismatch, 3);
    // The force model propagates without any IMU sample.
    assert!(filter.tick_space(t(60_000)).is_some());
    assert!(filter.coast_to(t(60_000)).is_ok());
    assert_eq!(
        filter.coast_to(t(30_000)),
        Err(RejectionReason::AcquisitionTimeRegression {
            last: t(60_000),
            got: t(30_000)
        })
    );
    let bad = PropagationMode::ForceModel(ForceModelCoast::new(0.0));
    assert_eq!(
        filter.set_propagation_mode(bad, t(60_000)),
        Err(RejectionReason::NonFiniteValue)
    );
    // A reset keeps the mode.
    filter.reset();
    assert_eq!(filter.propagation_mode(), mode);
}

#[test]
fn a_burn_adds_the_velocity_change_and_its_covariance() {
    let mut filter = space_filter(full_config());
    let mode = PropagationMode::ForceModel(settings());
    filter.set_propagation_mode(mode, t(0)).expect("plausible");
    filter.coast_to(t(30_000)).expect("coast");
    let before = *core(&filter);
    let burn_cov = SymmetricCov3::from_diagonal(4e-4, 1e-4, 9e-4);
    filter
        .apply_impulsive_burn([0.5, -1.0, 2.0], burn_cov, t(30_000))
        .expect("burn");
    let after = core(&filter);
    let dv = after.nominal.v - before.nominal.v;
    assert!((dv - Vector3::new(0.5, -1.0, 2.0)).norm() < 1e-12);
    let added = after.p.fixed_view::<3, 3>(VEL, VEL) - before.p.fixed_view::<3, 3>(VEL, VEL);
    assert!((added[(0, 0)] - 4e-4).abs() < 1e-15 && (added[(2, 2)] - 9e-4).abs() < 1e-15);
    assert_eq!(
        filter.apply_impulsive_burn([f64::NAN, 0.0, 0.0], burn_cov, t(30_000)),
        Err(RejectionReason::NonFiniteValue)
    );
}

fn fix(sequence: u32, at_ms: u64, position: Vector3<f64>) -> Observation {
    Observation::in_frame(
        stamp(5, sequence, at_ms),
        ObservationValue::CartesianPositionFix {
            position: CartesianPosition::new(position.x + 30.0, position.y - 20.0, position.z),
            covariance: SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
        },
        gnss(),
        FrameId::EARTH_J2000,
    )
}

/// A coast to 120 s with a burn at 60 s and a fix acquired at 30 s that
/// arrives at `fix_now_ms`.
fn run_with_burn(fix_now_ms: u64) -> InertialCore {
    let mut filter = space_filter(full_config());
    let mode = PropagationMode::ForceModel(settings());
    filter.set_propagation_mode(mode, t(0)).expect("plausible");
    let burn_cov = SymmetricCov3::from_diagonal(1e-4, 1e-4, 1e-4);
    // The fix is near the coast of the nominal state to 30 s.
    let model = inertial_env().force_model();
    let (mut position, mut velocity) = (leo_nominal().p, leo_nominal().v);
    for _ in 0..3 {
        (position, velocity) = model.rk4_step(&position, &velocity, 10.0);
    }
    let fix_position = position;
    for step in 1..=12_u64 {
        let at_ms = step * 10_000;
        if at_ms == 60_000 {
            filter
                .apply_impulsive_burn([0.0, 0.2, 0.1], burn_cov, t(at_ms))
                .expect("burn");
        }
        if at_ms == fix_now_ms {
            assert_accepted(filter.ingest(&fix(1, 30_000, fix_position), t(at_ms)));
        }
        filter.coast_to(t(at_ms)).expect("coast");
    }
    *core(&filter)
}

#[test]
fn a_delayed_fix_replays_through_the_burn() {
    let in_order = run_with_burn(30_000);
    let delayed = run_with_burn(90_000);
    let scale = in_order.nominal.p.norm();
    assert!((in_order.nominal.p - delayed.nominal.p).norm() < 1e-9 * scale);
    assert!((in_order.nominal.v - delayed.nominal.v).norm() < 1e-9);
    assert!((in_order.p - delayed.p).abs().max() < 1e-9 * in_order.p.abs().max());
    let without = run_with_burn(u64::MAX);
    assert!((without.nominal.p - in_order.nominal.p).norm() > 1.0);
}

#[test]
fn a_burn_after_the_state_time_in_the_sensed_imu_mode_names_the_mode() {
    // No IMU sample gives rates to reach 5 s, but the mode refusal comes
    // first.
    let mut filter = space_filter(full_config());
    let burn_cov = SymmetricCov3::from_diagonal(1e-4, 1e-4, 1e-4);
    assert_eq!(
        filter.apply_impulsive_burn([1.0, 0.0, 0.0], burn_cov, t(5_000)),
        Err(RejectionReason::PropagationModeMismatch)
    );
    assert_eq!(
        filter.coast_to(t(5_000)),
        Err(RejectionReason::PropagationModeMismatch)
    );
}

#[test]
fn a_change_of_mode_forgets_the_imu_rates() {
    let mut filter = space_filter(full_config());
    let thrust = ([0.0, 0.0, 2.0], [0.0; 3]);
    let sample = imu_in(1, 1_000, thrust, 1.0, FrameId::EARTH_J2000);
    assert_accepted(filter.ingest_imu(&sample, t(1_000)));
    let coast = PropagationMode::ForceModel(settings());
    filter
        .set_propagation_mode(coast, t(1_000))
        .expect("plausible");
    filter.coast_to(t(30_000)).expect("coast");
    filter
        .set_propagation_mode(PropagationMode::SensedImu, t(30_000))
        .expect("plausible");
    // The thrust before the coast does not drive the state after it.
    assert!(filter.tick_space(t(31_000)).is_none());
    assert!(filter.tick_space(t(30_000)).is_some());
    let next = imu_in(2, 31_000, ([0.0; 3], [0.0; 3]), 1.0, FrameId::EARTH_J2000);
    assert_accepted(filter.ingest_imu(&next, t(31_000)));
    assert!(filter.tick_space(t(32_000)).is_some());
}
