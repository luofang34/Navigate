//! A barometric altimeter on a route with coordinated turns and
//! kilometers of horizontal error (defect 5 of the simulator trial).
//!
//! The truth runs through the mechanization of the filter, so the IMU,
//! the alignment, and the altimeter errors are exactly the errors that
//! the filter models: a tactical IMU, an in-flight alignment, and an
//! altimeter with a bias and white noise. Without the second-order
//! height term and with a first-order update of the horizontal states,
//! the horizontal NEES of this route rose to 80 after the third turn.

use std::vec::Vec;

use nalgebra::{Rotation3, UnitQuaternion, Vector3};
use navigate_contract::SymmetricCov3;
use navigate_geodesy::{BodyEllipsoid, BodyFixedPosition, body_fixed_to_geodetic};

use super::*;
use crate::inertial::local_level::{LocalLevel, quaternion_of};
use crate::inertial::mechanization::{ImuRates, mechanize};
use crate::inertial::nominal::{POS, VEL, rotation_vector};
use crate::inertial::regime::Regime;
use crate::kalman::FogmBias;
use crate::observation::{Observation, ObservationValue};

const G: f64 = 9.80665;

/// A route: speed, bank of the turns, duration, step, and the start
/// time and direction of each 90° turn.
struct Route {
    speed_mps: f64,
    bank_deg: f64,
    duration_s: f64,
    step_s: f64,
    turns: &'static [(f64, f64)],
}

/// The route of the trial: 40 m/s, three turns at 25° bank, 1400 s.
const TRIAL_ROUTE: Route = Route {
    speed_mps: 40.0,
    bank_deg: 25.0,
    duration_s: 1_400.0,
    step_s: 0.1,
    turns: &[(507.5, 1.0), (784.0, -1.0), (1_203.9, 1.0)],
};

/// A short route for the debug build: 55 m/s, two turns at 45° bank.
const SHORT_ROUTE: Route = Route {
    speed_mps: 55.0,
    bank_deg: 45.0,
    duration_s: 300.0,
    step_s: 0.5,
    turns: &[(100.0, 1.0), (200.0, -1.0)],
};

/// The debug build of `cargo test` runs the short route with two seeds;
/// the release build that `ci.sh` runs takes the trial route with
/// twelve.
const ROUTE: Route = if cfg!(debug_assertions) {
    SHORT_ROUTE
} else {
    TRIAL_ROUTE
};
const SEEDS: u32 = if cfg!(debug_assertions) { 2 } else { 12 };

/// The band of the mean horizontal NEES: the 0.1 % and 99.9 % points of
/// χ²(4) over 2 for two runs, and the `[0.5, 6]` of the trial acceptance
/// for twelve.
const NEES_BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.045, 9.2)
} else {
    (0.5, 6.0)
};

/// SplitMix64 finalizer: independent bits for each input.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A standard normal draw for one seed and one index.
fn gauss(seed: u32, index: u32) -> f64 {
    let uniform = |j: u64| {
        let bits = mix((u64::from(seed) << 40) ^ j) >> 11;
        (bits as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let j = u64::from(index) * 2;
    let (a, b) = (uniform(j), uniform(j + 1));
    (-2.0 * a.ln()).sqrt() * (core::f64::consts::TAU * b).cos()
}

fn draw3(seed: u32, first: u32, sigma: f64) -> Vector3<f64> {
    Vector3::new(
        gauss(seed, first),
        gauss(seed, first + 1),
        gauss(seed, first + 2),
    ) * sigma
}

/// The tactical grade of the trial: 1 mg and 1°/h FOGM biases with a
/// one-hour correlation time, 0.05 m/s/√h and 0.1°/√h white noise.
fn tactical() -> InertialConfig {
    let per_sqrt_hour = 1.0 / 60.0;
    InertialConfig {
        accel_noise_sigma_mps2_sqrt_hz: 0.05 * per_sqrt_hour,
        gyro_noise_sigma_rad_s_sqrt_hz: 0.1_f64.to_radians() * per_sqrt_hour,
        accel_bias: FogmBias::new(3_600.0, 1e-3 * G),
        gyro_bias: FogmBias::new(3_600.0, 1.0_f64.to_radians() / 3_600.0),
        ..InertialConfig::default()
    }
}

/// The truth state and its commanded heading and bank.
struct Truth {
    x: InertialNominal,
    heading: f64,
    bank: f64,
    altitude_m: f64,
}

impl Truth {
    /// The commanded turn rate at `t_s`.
    fn turn_rate(t_s: f64) -> f64 {
        let rate = G * ROUTE.bank_deg.to_radians().tan() / ROUTE.speed_mps;
        let duration = core::f64::consts::FRAC_PI_2 / rate;
        ROUTE
            .turns
            .iter()
            .find(|(start, _)| (*start..start + duration).contains(&t_s))
            .map_or(0.0, |(_, sign)| sign * rate)
    }

    /// The IMU rates of one step of a coordinated turn that holds the
    /// altitude: the bank rolls toward the command at up to 0.5 rad/s,
    /// the heading turns at `g·tan φ / V`, and the specific force gives
    /// the centripetal acceleration.
    fn rates(&mut self, t_s: f64, regime: &Regime) -> ImuRates {
        let dt = ROUTE.step_s;
        let level = LocalLevel::at_body_fixed(&self.x.p, BodyEllipsoid::WGS84);
        let target = (Self::turn_rate(t_s) * ROUTE.speed_mps / G).atan();
        self.bank += ((target - self.bank) / 1.0).clamp(-0.5, 0.5) * dt;
        let rate = G * self.bank.tan() / ROUTE.speed_mps;
        self.heading += rate * dt;
        let body_to_ned = Rotation3::from_euler_angles(self.bank, 0.0, self.heading);
        let c_target = level.ned_to_bf * body_to_ned.matrix();
        let c = self.x.c_b_to_f();
        let omega = regime.frame_rotation();
        let turn = UnitQuaternion::from_matrix(&(c.transpose() * c_target));
        let v = level.bf_to_ned() * self.x.v;
        let height = level.position.altitude_m - self.altitude_m;
        let a_ned = Vector3::new(-rate * v.y, rate * v.x, 0.02 * height - 0.3 * v.z);
        let (g, _) = regime.gravity(&self.x.p);
        ImuRates {
            specific_force: c.transpose()
                * (level.ned_to_bf * a_ned - g + 2.0 * omega.cross(&self.x.v)),
            angular_rate: rotation_vector(&turn) / dt + c.transpose() * omega,
        }
    }
}

/// The horizontal NEES, the horizontal 1-sigma, and the horizontal
/// error every 100 s of one run.
fn run(seed: u32, baro: bool) -> Vec<(f64, f64, f64)> {
    let config = tactical();
    let regime = core(&surface_filter(
        config,
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    ))
    .regime;
    let level = LocalLevel::at(earth_origin());
    let speed = ROUTE.speed_mps;
    let mut truth = Truth {
        x: InertialNominal::at(
            earth_body_fixed(&earth_origin()),
            level.ned_to_bf * Vector3::new(speed, 0.0, 0.0),
            quaternion_of(&level.ned_to_bf),
        ),
        heading: 0.0,
        bank: 0.0,
        altitude_m: earth_origin().altitude_m,
    };
    let (pos_sigma, vel_sigma, att_sigma) = (20.0, 0.2, 2e-3);
    let start_p = truth.x.p + level.ned_to_bf * draw3(seed, 1, pos_sigma);
    let start = body_fixed_to_geodetic(
        &BodyFixedPosition::new(start_p.x, start_p.y, start_p.z),
        BodyEllipsoid::WGS84,
    );
    let dv = draw3(seed, 4, vel_sigma);
    let tilt = UnitQuaternion::from_rotation_matrix(&Rotation3::new(draw3(seed, 7, att_sigma)));
    let (p2, v2, a2) = (pos_sigma.powi(2), vel_sigma.powi(2), att_sigma.powi(2));
    let mut filter =
        InertialNavigationFilter::new_surface(config, CLOCK, PlanetaryBody::EARTH_WGS84);
    filter
        .initialize_surface(
            start,
            NedVelocity::new(speed + dv.x, dv.y, dv.z),
            AttitudeQuaternion::new(tilt.w, tilt.i, tilt.j, tilt.k),
            SymmetricCov3::from_diagonal(p2, p2, p2),
            SymmetricCov3::from_diagonal(v2, v2, v2),
            SymmetricCov3::from_diagonal(a2, a2, a2),
            t(0),
            gnss(),
        )
        .expect("plausible initial state");
    let dt = ROUTE.step_s;
    let accel_bias = draw3(seed, 10, config.accel_bias.sigma());
    let gyro_bias = draw3(seed, 13, config.gyro_bias.sigma());
    let baro_bias = config.baro_bias.sigma() * gauss(seed, 16);
    let accel_white = config.accel_noise_sigma_mps2_sqrt_hz / dt.sqrt();
    let gyro_white = config.gyro_noise_sigma_rad_s_sqrt_hz / dt.sqrt();
    let steps = (ROUTE.duration_s / dt).round() as u32;
    let step_ms = (dt * 1_000.0).round() as u64;
    let per_report = (100.0 / dt).round() as u32;
    let mut samples = Vec::new();
    let mut refused = 0_u32;
    for k in 1..=steps {
        let at_ms = u64::from(k) * step_ms;
        let rates = truth.rates(f64::from(k - 1) * dt, &regime);
        truth.x = mechanize(&truth.x, &rates, dt, &regime, &config).0;
        let force = rates.specific_force + accel_bias + draw3(seed, 100_000 + 6 * k, accel_white);
        let rate = rates.angular_rate + gyro_bias + draw3(seed, 100_003 + 6 * k, gyro_white);
        let sample = imu(k, at_ms, (force.into(), rate.into()), dt);
        assert_accepted(filter.ingest_imu(&sample, t(at_ms)));
        if baro {
            let height = LocalLevel::at_body_fixed(&truth.x.p, BodyEllipsoid::WGS84)
                .position
                .altitude_m;
            let obs = Observation::new(
                stamp(3, k, at_ms),
                ObservationValue::Altitude {
                    altitude_m: height + baro_bias + gauss(seed, 5_000_000 + k),
                    variance_m2: 1.0,
                },
                SourceComposition::of(SensorClass::Barometric),
            );
            if !filter.ingest(&obs, t(at_ms)).is_accepted() {
                refused = refused.wrapping_add(1);
            }
        }
        if k % per_report == 0 {
            let solution = filter.tick(t(at_ms)).expect("published");
            let local = LocalLevel::at_body_fixed(&truth.x.p, BodyEllipsoid::WGS84);
            let e = local.bf_to_ned() * (core(&filter).nominal.p - truth.x.p);
            let [nn, ne, _, ee, _, _] = solution.position_cov.upper_triangle();
            let nees =
                (ee * e.x * e.x - 2.0 * ne * e.x * e.y + nn * e.y * e.y) / (nn * ee - ne * ne);
            samples.push((nees, (nn + ee).sqrt(), e.x.hypot(e.y)));
        }
    }
    // The 0.1 % innovation gate refuses a few true draws.
    assert!(refused * 100 <= steps, "{refused} of {steps} refused");
    samples
}

/// Mean NEES, mean 1-sigma, and RMS error at each report over the
/// seeds.
fn statistics(baro: bool) -> Vec<(f64, f64, f64)> {
    let runs: Vec<Vec<(f64, f64, f64)>> = (0..SEEDS).map(|seed| run(seed, baro)).collect();
    let n = f64::from(SEEDS);
    let reports = runs.first().map_or(0, Vec::len);
    (0..reports)
        .map(|i| {
            let at = |run: &Vec<(f64, f64, f64)>| run.get(i).copied().unwrap_or_default();
            let nees = runs.iter().map(|r| at(r).0).sum::<f64>() / n;
            let sigma = runs.iter().map(|r| at(r).1).sum::<f64>() / n;
            let rms = (runs.iter().map(|r| at(r).2.powi(2)).sum::<f64>() / n).sqrt();
            (nees, sigma, rms)
        })
        .collect()
}

#[test]
fn monte_carlo_an_altimeter_in_turns_keeps_the_horizontal_estimate_consistent() {
    let free = statistics(false);
    let baro = statistics(true);
    let (low, high) = NEES_BAND;
    for (index, ((free_nees, _, free_rms), (nees, sigma, rms))) in
        free.iter().zip(&baro).enumerate()
    {
        let at_s = (index + 1) * 100;
        assert!(
            (low..=high).contains(nees),
            "{at_s} s: NEES {nees}, σ_h {sigma} m, RMS {rms} m"
        );
        assert!(
            (low..=high).contains(free_nees),
            "{at_s} s: INS NEES {free_nees}"
        );
        // The altimeter does not make the horizontal error worse.
        assert!(
            *rms <= 1.05 * free_rms,
            "{at_s} s: RMS {rms} vs INS {free_rms}"
        );
    }
}

#[test]
fn an_altitude_update_does_not_change_the_horizontal_covariance() {
    // After a turn the error state correlates the vertical with the
    // horizontal; an altitude update leaves the horizontal position and
    // velocity covariance as it was, and still reduces the vertical.
    let mut filter = surface_filter(tactical(), PlanetaryBody::EARTH_WGS84, earth_origin());
    let rates = rest_rates(&filter);
    for k in 1..=120_u32 {
        let at_ms = u64::from(k) * 1_000;
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, rates, 1.0), t(at_ms)));
    }
    let before = *core(&filter);
    let obs = Observation::new(
        stamp(3, 1, 120_000),
        ObservationValue::Altitude {
            altitude_m: earth_origin().altitude_m + 2.0,
            variance_m2: 1.0,
        },
        SourceComposition::of(SensorClass::Barometric),
    );
    assert_accepted(filter.ingest(&obs, t(120_000)));
    let after = *core(&filter);
    let local = LocalLevel::at_body_fixed(&before.nominal.p, BodyEllipsoid::WGS84);
    for offset in [POS, VEL] {
        let block = |c: &crate::inertial::estimate::InertialCore| {
            local.bf_to_ned_cov(&c.p.fixed_view::<3, 3>(offset, offset).into_owned())
        };
        let (b, a) = (block(&before), block(&after));
        let horizontal = (b.fixed_view::<2, 2>(0, 0) - a.fixed_view::<2, 2>(0, 0))
            .abs()
            .max();
        assert!(horizontal <= 1e-9 * b.abs().max(), "{offset}: {horizontal}");
        assert!(a[(2, 2)] < b[(2, 2)], "{offset}: vertical");
    }
    let shift = local.bf_to_ned() * (after.nominal.p - before.nominal.p);
    assert!(shift.x.hypot(shift.y) < 1e-6, "{shift}");
}
