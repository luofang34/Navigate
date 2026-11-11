//! A barometric altitude moves the horizontal estimate through the
//! altitude–horizontal cross-covariance. This is correct estimation when
//! the truth has the errors that the filter models. A Monte Carlo run
//! with a model-matched truth shows it: the horizontal NEES stays in its
//! band with and without the altimeter, and the altimeter decreases the
//! horizontal error.

use nalgebra::{Rotation3, Vector3};
use navigate_contract::SymmetricCov3;
use navigate_geodesy::{BodyEllipsoid, BodyFixedPosition, body_fixed_to_geodetic};

use super::*;
use crate::inertial::local_level::LocalLevel;
use crate::observation::{Observation, ObservationValue};

/// Independent runs, each with its own draw of every error. The debug
/// build of `cargo test` runs a small sample to keep the run short; the
/// release build that `ci.sh` runs takes the full sample.
const SEEDS: u32 = if cfg!(debug_assertions) { 4 } else { 24 };
/// IMU and altimeter interval, milliseconds. The vehicle is at rest, so
/// a long interval integrates exactly and keeps the test fast.
const STEP_MS: u32 = 1_000;
/// Duration of one run, seconds.
const DURATION_S: u32 = if cfg!(debug_assertions) { 240 } else { 600 };
/// The band of the mean horizontal NEES. The sum of the NEES of
/// [`SEEDS`] consistent runs is χ² with `2·SEEDS` degrees of freedom.
/// For 4 runs, the 0.1 % and 99.9 % points of χ²(8) are 0.86 and 26.1,
/// so the mean is in `[0.2, 6.5]`. For 24 runs, the band is the
/// `[0.5, 6]` of the trial acceptance, which is wider than the χ²(48)
/// interval.
const NEES_BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.2, 6.5)
} else {
    (0.5, 6.0)
};
/// Initial horizontal 1-sigma, meters: a position source lost long ago.
const HORIZONTAL_SIGMA_M: f64 = 800.0;
/// Initial attitude 1-sigma about each axis, radians: an in-flight
/// alignment.
const ATTITUDE_SIGMA_RAD: f64 = 1e-3;

/// SplitMix64 finalizer: independent bits for each input.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A standard normal draw for one seed and one index: Box–Muller over
/// two SplitMix64 uniforms.
fn gauss(seed: u32, index: u32) -> f64 {
    let uniform = |j: u64| {
        let bits = mix((u64::from(seed) << 40) ^ j) >> 11;
        (bits as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let j = u64::from(index) * 2;
    let (a, b) = (uniform(j), uniform(j + 1));
    (-2.0 * a.ln()).sqrt() * (core::f64::consts::TAU * b).cos()
}

/// Index offsets that keep the draws of one seed apart.
const POSITION: u32 = 1;
const VELOCITY: u32 = 4;
const TILT: u32 = 7;
const ACCEL_BIAS: u32 = 10;
const GYRO_BIAS: u32 = 13;
const BARO_BIAS: u32 = 16;
const BARO_NOISE: u32 = 1_000;
const IMU_NOISE: u32 = 100_000;

/// The horizontal NEES and the horizontal error in meters at the end of
/// one run of a vehicle at rest at `earth_origin`. The truth draws the
/// initial position, velocity, and attitude errors from the initial
/// covariance of the filter, and the IMU and altimeter errors from
/// [`InertialConfig::default`].
fn run(seed: u32, with_baro: bool) -> (f64, f64) {
    let config = InertialConfig::default();
    let truth_p = earth_body_fixed(&earth_origin());
    let level = LocalLevel::at(earth_origin());
    let draw3 = |first: u32, sigma: [f64; 3]| {
        Vector3::new(
            sigma[0] * gauss(seed, first),
            sigma[1] * gauss(seed, first + 1),
            sigma[2] * gauss(seed, first + 2),
        )
    };
    let attitude_sigma = [ATTITUDE_SIGMA_RAD; 3];
    let tilt = Rotation3::new(draw3(TILT, attitude_sigma));
    let at_rest = rest_rates(&surface_filter(
        config,
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    ));
    let rotate = |v: [f64; 3]| -> [f64; 3] { (tilt * Vector3::from(v)).into() };
    let rates = (rotate(at_rest.0), rotate(at_rest.1));
    let sigma_h = HORIZONTAL_SIGMA_M;
    let offset = level.ned_to_bf * draw3(POSITION, [sigma_h, sigma_h, 5.0]);
    let start_p = truth_p + offset;
    let start = body_fixed_to_geodetic(
        &BodyFixedPosition::new(start_p.x, start_p.y, start_p.z),
        BodyEllipsoid::WGS84,
    );
    let velocity = draw3(VELOCITY, [1.0, 1.0, 0.1]);
    let a2 = attitude_sigma.map(|s| s * s);
    let mut filter =
        InertialNavigationFilter::new_surface(config, CLOCK, PlanetaryBody::EARTH_WGS84);
    filter
        .initialize_surface(
            start,
            NedVelocity::new(velocity.x, velocity.y, velocity.z),
            identity_attitude(),
            SymmetricCov3::from_diagonal(sigma_h * sigma_h, sigma_h * sigma_h, 25.0),
            SymmetricCov3::from_diagonal(1.0, 1.0, 0.01),
            SymmetricCov3::from_diagonal(a2[0], a2[1], a2[2]),
            t(0),
            gnss(),
        )
        .expect("plausible initial state");
    let dt_s = f64::from(STEP_MS) / 1_000.0;
    let accel_white = config.accel_noise_sigma_mps2_sqrt_hz / dt_s.sqrt();
    let gyro_white = config.gyro_noise_sigma_rad_s_sqrt_hz / dt_s.sqrt();
    let accel_bias = draw3(ACCEL_BIAS, [config.accel_bias.sigma(); 3]);
    let gyro_bias = draw3(GYRO_BIAS, [config.gyro_bias.sigma(); 3]);
    let baro_bias = config.baro_bias.sigma() * gauss(seed, BARO_BIAS);
    let steps = DURATION_S * 1_000 / STEP_MS;
    let mut refused = 0_u32;
    let step_ms = u64::from(STEP_MS);
    for k in 1..=steps {
        let at_ms = u64::from(k) * step_ms;
        let accel = accel_bias + draw3(IMU_NOISE + 6 * k, [accel_white; 3]);
        let gyro = gyro_bias + draw3(IMU_NOISE + 6 * k + 3, [gyro_white; 3]);
        let sample = (
            (Vector3::from(rates.0) + accel).into(),
            (Vector3::from(rates.1) + gyro).into(),
        );
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, sample, dt_s), t(at_ms)));
        if with_baro {
            let altitude_m = earth_origin().altitude_m + baro_bias + gauss(seed, BARO_NOISE + k);
            let observation = Observation::new(
                stamp(3, k, at_ms),
                ObservationValue::Altitude {
                    altitude_m,
                    variance_m2: 1.0,
                },
                SourceComposition::of(SensorClass::Barometric),
            );
            // The 0.1 % innovation gate refuses a few true draws.
            if !filter.ingest(&observation, t(at_ms)).is_accepted() {
                refused = refused.wrapping_add(1);
            }
        }
    }
    assert!(refused * 100 < steps, "{refused} of {steps} refused");
    let end = t(u64::from(steps) * step_ms);
    let solution = filter.tick(end).expect("published");
    let error = level.ned_to_bf.transpose() * (core(&filter).nominal.p - truth_p);
    let [nn, ne, _, ee, _, _] = solution.position_cov.upper_triangle();
    let determinant = nn * ee - ne * ne;
    let nees = (ee * error.x * error.x - 2.0 * ne * error.x * error.y + nn * error.y * error.y)
        / determinant;
    (nees, error.x.hypot(error.y))
}

/// Mean horizontal NEES and RMS horizontal error over [`SEEDS`] runs.
fn monte_carlo(with_baro: bool) -> (f64, f64) {
    let (mut nees, mut squared) = (0.0, 0.0);
    for seed in 0..SEEDS {
        let (run_nees, error) = run(seed, with_baro);
        nees += run_nees;
        squared += error * error;
    }
    let n = f64::from(SEEDS);
    (nees / n, (squared / n).sqrt())
}

#[test]
fn monte_carlo_a_barometric_altitude_improves_the_horizontal_estimate_consistently() {
    let (free_nees, free_rms) = monte_carlo(false);
    let (baro_nees, baro_rms) = monte_carlo(true);
    // The horizontal NEES has two degrees of freedom: its mean is 2.
    let (low, high) = NEES_BAND;
    for nees in [free_nees, baro_nees] {
        assert!(
            (low..=high).contains(&nees),
            "free {free_nees} baro {baro_nees}"
        );
    }
    // Four runs are too few to compare two error distributions; the
    // release sample compares them.
    if !cfg!(debug_assertions) {
        assert!(
            baro_rms < free_rms,
            "baro {baro_rms} m vs free {free_rms} m"
        );
    }
}
