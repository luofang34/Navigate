//! Star lines of sight on a surface vehicle: the distribution of the
//! horizontal error with and without stars over many seeds, at rest and
//! in turns. The truth runs through the mechanization of the filter, so
//! the IMU, the alignment, and the star errors are exactly the errors
//! that the filter models. A correct filter that receives a correct
//! extra measurement must not get worse on average, and its horizontal
//! NEES must stay in its band.

use nalgebra::{Rotation3, Vector3};
use navigate_contract::{AttitudeQuaternion, SymmetricCov3};
use navigate_geodesy::{BodyEllipsoid, BodyFixedPosition, body_fixed_to_geodetic};

use super::*;
use crate::inertial::local_level::{LocalLevel, quaternion_of};
use crate::inertial::mechanization::{ImuRates, mechanize};
use crate::inertial::regime::Regime;
use crate::observation::{LineOfSightReference, Observation, ObservationValue};

/// Independent runs for each case. The debug build of `cargo test`
/// runs a small sample to keep the run short; the release build that
/// `ci.sh` runs takes the full sample.
const SEEDS: u32 = if cfg!(debug_assertions) { 3 } else { 20 };
/// Duration of one run, seconds: the ten minutes of the measured case
/// in the release build.
const DURATION_S: u32 = if cfg!(debug_assertions) { 90 } else { 600 };
/// The band of the mean horizontal NEES. The sum of the NEES of
/// [`SEEDS`] consistent runs is χ² with `2·SEEDS` degrees of freedom.
/// For 3 runs, the 0.1 % and 99.9 % points of χ²(6) give a mean in
/// `[0.13, 7.5]`. For 20 runs, the band is the `[0.5, 6]` of the trial
/// acceptance, which is wider than the χ²(40) interval.
const NEES_BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.13, 7.5)
} else {
    (0.5, 6.0)
};
/// Initial alignment 1-sigma: position in meters, velocity in m/s, and
/// attitude in radians about each axis, as an in-flight alignment.
const ALIGNMENT: (f64, f64, f64) = (2.0, 0.1, 1e-3);
/// Stars in each 1 Hz frame of a roof camera.
const STARS_PER_FRAME: u32 = 8;
/// Centroid direction 1-sigma, radians.
const STAR_SIGMA_RAD: f64 = 5e-6;
/// Camera mount alignment 1-sigma, radians: a constant error of each
/// run, declared in the variance of each line of sight.
const MOUNT_SIGMA_RAD: f64 = 2e-5;
/// Turn rate in rad/s for the first half of each 180 s period of a
/// turning run: four half turns in ten minutes.
const TURN_RATE_RAD_S: f64 = core::f64::consts::PI / 90.0;

/// A standard normal draw for one seed and one index.
fn gauss(seed: u32, index: u32) -> f64 {
    let uniform = |j: u32| {
        let hash = seed
            .wrapping_mul(7_919)
            .wrapping_add(j)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(0x9E37_79B9)
            >> 8;
        (f64::from(hash) + 0.5) / f64::from(1_u32 << 24)
    };
    let (a, b) = (uniform(2 * index), uniform(2 * index + 1));
    (-2.0 * a.ln()).sqrt() * (core::f64::consts::TAU * b).cos()
}

fn draw3(seed: u32, first: u32, sigma: f64) -> Vector3<f64> {
    Vector3::new(
        gauss(seed, first),
        gauss(seed, first + 1),
        gauss(seed, first + 2),
    ) * sigma
}

/// Index offsets that keep the draws of one seed apart.
const POSITION: u32 = 1;
const VELOCITY: u32 = 4;
const TILT: u32 = 7;
const ACCEL_BIAS: u32 = 10;
const GYRO_BIAS: u32 = 13;
const MOUNT: u32 = 16;
const IMU_NOISE: u32 = 100_000;
const STARS: u32 = 2_000_000;

/// One case of the distribution.
#[derive(Clone, Copy)]
struct Case {
    stars: bool,
    turning: bool,
}

/// The ideal IMU rates of the truth: at rest relative to the Earth,
/// except a turn about the local down axis at `turn_rate`.
fn ideal_rates(truth: &InertialNominal, regime: &Regime, turn_rate: f64) -> ImuRates {
    let (g, _) = regime.gravity(&truth.p);
    let c = truth.c_b_to_f();
    let down = -LocalLevel::at_body_fixed(&truth.p, BodyEllipsoid::WGS84).up();
    ImuRates {
        specific_force: c.transpose() * -g,
        angular_rate: c.transpose() * (regime.frame_rotation() + down * turn_rate),
    }
}

/// The horizontal NEES and the horizontal error in meters at the end of
/// one run.
fn run(seed: u32, case: Case) -> (f64, f64) {
    let config = InertialConfig::default();
    let (pos_sigma, vel_sigma, att_sigma) = ALIGNMENT;
    let reference = surface_filter(config, PlanetaryBody::EARTH_WGS84, earth_origin());
    let regime = core(&reference).regime;
    let level = LocalLevel::at(earth_origin());
    let tilt = Rotation3::new(draw3(seed, TILT, att_sigma));
    let mut truth = InertialNominal::at(
        earth_body_fixed(&earth_origin()),
        Vector3::zeros(),
        quaternion_of(&(level.ned_to_bf * tilt.matrix())),
    );
    let start_p = truth.p + level.ned_to_bf * draw3(seed, POSITION, pos_sigma);
    let start = body_fixed_to_geodetic(
        &BodyFixedPosition::new(start_p.x, start_p.y, start_p.z),
        BodyEllipsoid::WGS84,
    );
    let v0 = draw3(seed, VELOCITY, vel_sigma);
    let mut filter =
        InertialNavigationFilter::new_surface(config, CLOCK, PlanetaryBody::EARTH_WGS84);
    let (p2, v2, a2) = (pos_sigma.powi(2), vel_sigma.powi(2), att_sigma.powi(2));
    filter
        .initialize_surface(
            start,
            NedVelocity::new(v0.x, v0.y, v0.z),
            AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            SymmetricCov3::from_diagonal(p2, p2, p2),
            SymmetricCov3::from_diagonal(v2, v2, v2),
            SymmetricCov3::from_diagonal(a2, a2, a2),
            t(0),
            gnss(),
        )
        .expect("plausible initial state");
    let accel_bias = draw3(seed, ACCEL_BIAS, config.accel_bias.sigma());
    let gyro_bias = draw3(seed, GYRO_BIAS, config.gyro_bias.sigma());
    let mount = Rotation3::new(draw3(seed, MOUNT, MOUNT_SIGMA_RAD));
    let variance = STAR_SIGMA_RAD.powi(2) + MOUNT_SIGMA_RAD.powi(2);
    let mut star = 0_u32;
    for k in 1..=DURATION_S {
        let at_ms = u64::from(k) * 1_000;
        let turning = case.turning && (k - 1) % 180 < 90;
        let rates = ideal_rates(&truth, &regime, if turning { TURN_RATE_RAD_S } else { 0.0 });
        truth = mechanize(&truth, &rates, 1.0, &regime, &config).0;
        let measured = (
            (rates.specific_force
                + accel_bias
                + draw3(
                    seed,
                    IMU_NOISE + 6 * k,
                    config.accel_noise_sigma_mps2_sqrt_hz,
                ))
            .into(),
            (rates.angular_rate
                + gyro_bias
                + draw3(
                    seed,
                    IMU_NOISE + 6 * k + 3,
                    config.gyro_noise_sigma_rad_s_sqrt_hz,
                ))
            .into(),
        );
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, measured, 1.0), t(at_ms)));
        if !case.stars {
            continue;
        }
        let c = truth.c_b_to_f();
        for _ in 0..STARS_PER_FRAME {
            star = star.wrapping_add(1);
            let tilt_draw = draw3(seed, STARS + 6 * star, 0.25);
            let body_unit = Vector3::new(tilt_draw.x, tilt_draw.y, -1.0).normalize();
            let noise = draw3(seed, STARS + 6 * star + 3, STAR_SIGMA_RAD);
            let measured_unit = (mount * body_unit + noise).normalize();
            let observation = Observation::in_frame(
                stamp(7, star, at_ms),
                ObservationValue::LineOfSight {
                    measured_body_unit: measured_unit.into(),
                    reference: LineOfSightReference::InertialDirection {
                        direction_in_frame_unit: (c * body_unit).into(),
                    },
                    angular_variance_rad2: variance,
                },
                SourceComposition::of(SensorClass::Celestial),
                filter.frame(),
            );
            assert_accepted(filter.ingest(&observation, t(at_ms)));
        }
    }
    let solution = filter
        .tick(t(u64::from(DURATION_S) * 1_000))
        .expect("published");
    let truth_level = LocalLevel::at_body_fixed(&truth.p, BodyEllipsoid::WGS84);
    let error = truth_level.bf_to_ned() * (core(&filter).nominal.p - truth.p);
    let [nn, ne, _, ee, _, _] = solution.position_cov.upper_triangle();
    let nees = (ee * error.x * error.x - 2.0 * ne * error.x * error.y + nn * error.y * error.y)
        / (nn * ee - ne * ne);
    (nees, error.x.hypot(error.y))
}

/// Mean horizontal NEES, RMS horizontal error, and the errors of each
/// seed.
fn distribution(case: Case) -> (f64, f64, std::vec::Vec<f64>) {
    let runs: std::vec::Vec<(f64, f64)> = (0..SEEDS).map(|seed| run(seed, case)).collect();
    let n = f64::from(SEEDS);
    let nees = runs.iter().map(|r| r.0).sum::<f64>() / n;
    let rms = (runs.iter().map(|r| r.1 * r.1).sum::<f64>() / n).sqrt();
    (nees, rms, runs.into_iter().map(|r| r.1).collect())
}

#[test]
fn monte_carlo_stars_do_not_make_the_horizontal_estimate_worse() {
    for turning in [false, true] {
        let (free_nees, free_rms, free) = distribution(Case {
            stars: false,
            turning,
        });
        let (star_nees, star_rms, starred) = distribution(Case {
            stars: true,
            turning,
        });
        let label = if turning { "turning" } else { "at rest" };
        // The horizontal NEES has two degrees of freedom: its mean is 2.
        let (low, high) = NEES_BAND;
        for nees in [free_nees, star_nees] {
            assert!(
                (low..=high).contains(&nees),
                "{label}: NEES free {free_nees} stars {star_nees}"
            );
        }
        // Three runs are too few to compare two error distributions;
        // the release sample compares them.
        if cfg!(debug_assertions) {
            continue;
        }
        assert!(
            star_rms < free_rms,
            "{label}: RMS stars {star_rms} m vs free {free_rms} m"
        );
        let better = free.iter().zip(&starred).filter(|(f, s)| s < f).count();
        assert!(
            better * 4 >= free.len() * 3,
            "{label}: stars better on {better} of {} seeds",
            free.len()
        );
    }
}
