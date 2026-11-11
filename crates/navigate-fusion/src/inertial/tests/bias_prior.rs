//! The IMU bias prior of the initialization.

use navigate_contract::SymmetricCov3;

use super::*;
use crate::inertial::ImuBiasPrior;
use crate::inertial::nominal::{ACCEL_BIAS, GYRO_BIAS};
use crate::rejection::RejectionReason;

fn prior(gyro: [f64; 3]) -> ImuBiasPrior {
    ImuBiasPrior::new(
        [1e-3, -2e-3, 5e-4],
        SymmetricCov3::from_diagonal(1e-8, 1e-8, 1e-8),
        gyro,
        SymmetricCov3::from_diagonal(1e-14, 1e-14, 1e-14),
    )
}

fn initialized(bias_prior: Option<ImuBiasPrior>) -> InertialNavigationFilter {
    let mut filter = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        CLOCK,
        PlanetaryBody::EARTH_WGS84,
    );
    filter
        .set_imu_bias_prior(bias_prior)
        .expect("plausible prior");
    filter
        .initialize_surface(
            earth_origin(),
            NedVelocity::new(0.0, 0.0, 0.0),
            identity_attitude(),
            SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(0),
            gnss(),
        )
        .expect("plausible initial state");
    filter
}

#[test]
fn without_a_prior_the_biases_start_at_zero_as_before() {
    let plain = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    assert_eq!(core(&initialized(None)), core(&plain));
}

#[test]
fn a_prior_sets_the_biases_and_their_covariance_blocks() {
    let gyro = [2e-5, -1e-5, 3e-5];
    let seeded = initialized(Some(prior(gyro)));
    let plain = initialized(None);
    let (seeded, plain) = (core(&seeded), core(&plain));
    assert_eq!(
        seeded.nominal.b_a,
        nalgebra::Vector3::new(1e-3, -2e-3, 5e-4)
    );
    assert_eq!(seeded.nominal.b_g, nalgebra::Vector3::from(gyro));
    for axis in 0..3 {
        assert_eq!(seeded.p[(ACCEL_BIAS + axis, ACCEL_BIAS + axis)], 1e-8);
        assert_eq!(seeded.p[(GYRO_BIAS + axis, GYRO_BIAS + axis)], 1e-14);
    }
    // Every other block is the default one.
    let mut p = seeded.p;
    p.fixed_view_mut::<6, 6>(ACCEL_BIAS, ACCEL_BIAS)
        .copy_from(&plain.p.fixed_view::<6, 6>(ACCEL_BIAS, ACCEL_BIAS));
    assert_eq!(p, plain.p);
}

#[test]
fn a_known_gyro_bias_keeps_the_heading() {
    // A 3e-5 rad/s yaw-rate bias is 100 times the default stationary
    // 1-sigma. Without the prior the heading drifts with it.
    let gyro = [0.0, 0.0, 3e-5];
    let heading_after_60_s = |bias_prior: Option<ImuBiasPrior>| {
        let mut filter = initialized(bias_prior);
        let (force, rate) = rest_rates(&filter);
        let measured = (force, [rate[0], rate[1], rate[2] + gyro[2]]);
        for k in 1..=60_u32 {
            let at_ms = u64::from(k) * 1_000;
            assert_accepted(filter.ingest_imu(&imu(k, at_ms, measured, 1.0), t(at_ms)));
        }
        let q = filter
            .tick(t(60_000))
            .expect("published")
            .attitude
            .expect("attitude");
        // The sign of a quaternion is free: take the one with w ≥ 0.
        let (w, z) = if q.w < 0.0 { (-q.w, -q.z) } else { (q.w, q.z) };
        2.0 * z.atan2(w)
    };
    let seeded = heading_after_60_s(Some(prior(gyro)));
    let plain = heading_after_60_s(None);
    // The bias estimate decays with the one-hour correlation time, so
    // the heading keeps b·t²/(2τ) = 1.5e-5 rad of the bias.
    assert!((seeded - 1.5e-5).abs() < 2e-6, "{seeded}");
    assert!((plain - 1.8e-3).abs() < 1e-4, "{plain}");
}

#[test]
fn an_implausible_prior_is_refused_and_the_previous_one_stays() {
    let mut filter = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        CLOCK,
        PlanetaryBody::EARTH_WGS84,
    );
    let good = prior([0.0, 0.0, 1e-5]);
    filter.set_imu_bias_prior(Some(good)).expect("plausible");
    assert_eq!(
        filter.set_imu_bias_prior(Some(prior([f64::NAN, 0.0, 0.0]))),
        Err(RejectionReason::NonFiniteValue)
    );
    let negative = ImuBiasPrior::new(
        [0.0; 3],
        SymmetricCov3::from_diagonal(-1.0, 1.0, 1.0),
        [0.0; 3],
        SymmetricCov3::from_diagonal(1e-14, 1e-14, 1e-14),
    );
    assert_eq!(
        filter.set_imu_bias_prior(Some(negative)),
        Err(RejectionReason::ImplausibleCovariance)
    );
    assert_eq!(filter.bias_prior, Some(good));
    // The prior stays for an initialization after a reset.
    filter.reset();
    assert_eq!(filter.bias_prior, Some(good));
}
