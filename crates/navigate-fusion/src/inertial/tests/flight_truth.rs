//! A model-matched truth for Monte Carlo tests: a vehicle that flies a
//! route of straight legs and coordinated turns, and a filter that
//! receives its IMU data with exactly the errors that the filter
//! models.
//!
//! The truth runs through the mechanization of the filter, so the truth
//! and the filter integrate with the same method. Each run draws the
//! alignment error, the IMU biases, and the IMU white noise from the
//! filter's own initial covariance and noise model.

use nalgebra::{Const, DimMin, Matrix3, Rotation3, UnitQuaternion, Vector3};
use navigate_contract::SymmetricCov3;
use navigate_geodesy::{BodyEllipsoid, BodyFixedPosition, body_fixed_to_geodetic};

use super::*;
use crate::inertial::local_level::{LocalLevel, quaternion_of};
use crate::inertial::mechanization::{ImuRates, mechanize};
use crate::inertial::nominal::{ATT, FULL_STATES, POS, rotation_vector};
use crate::inertial::regime::Regime;
use crate::kalman::FogmBias;

/// Standard gravity in m/s², for the bank of a coordinated turn.
const G: f64 = 9.80665;

/// SplitMix64 finalizer: independent bits for each input.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A standard normal draw for one seed and one index: Box–Muller over
/// two SplitMix64 uniforms. Different indices give independent draws.
pub(super) fn gauss(seed: u32, index: u32) -> f64 {
    let uniform = |j: u64| {
        let bits = mix((u64::from(seed) << 40) ^ j) >> 11;
        (bits as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let j = u64::from(index) * 2;
    let (a, b) = (uniform(j), uniform(j + 1));
    (-2.0 * a.ln()).sqrt() * (core::f64::consts::TAU * b).cos()
}

/// Three independent normal draws with the 1-sigma `sigma`.
pub(super) fn draw3(seed: u32, first: u32, sigma: f64) -> Vector3<f64> {
    Vector3::new(
        gauss(seed, first),
        gauss(seed, first + 1),
        gauss(seed, first + 2),
    ) * sigma
}

/// The tactical grade of the simulator trial: 1 mg and 1°/h FOGM biases
/// with a one-hour correlation time, 0.05 m/s/√h and 0.1°/√h white
/// noise.
pub(super) fn tactical() -> InertialConfig {
    let per_sqrt_hour = 1.0 / 60.0;
    InertialConfig {
        accel_noise_sigma_mps2_sqrt_hz: 0.05 * per_sqrt_hour,
        gyro_noise_sigma_rad_s_sqrt_hz: 0.1_f64.to_radians() * per_sqrt_hour,
        accel_bias: FogmBias::new(3_600.0, 1e-3 * G),
        gyro_bias: FogmBias::new(3_600.0, 1.0_f64.to_radians() / 3_600.0),
        ..InertialConfig::default()
    }
}

/// A route: speed, bank of the turns, duration, IMU step, and the start
/// time and direction of each 90° turn. A route at zero speed has no
/// turns.
#[derive(Debug, Clone, Copy)]
pub(super) struct Route {
    pub(super) speed_mps: f64,
    pub(super) bank_deg: f64,
    pub(super) duration_s: f64,
    pub(super) step_s: f64,
    pub(super) turns: &'static [(f64, f64)],
}

impl Route {
    /// The number of IMU steps of the route.
    pub(super) fn steps(&self) -> u32 {
        (self.duration_s / self.step_s).round() as u32
    }

    /// The commanded turn rate at `t_s`.
    fn turn_rate(&self, t_s: f64) -> f64 {
        if self.speed_mps <= 0.0 {
            return 0.0;
        }
        let rate = G * self.bank_deg.to_radians().tan() / self.speed_mps;
        let duration = core::f64::consts::FRAC_PI_2 / rate;
        self.turns
            .iter()
            .find(|(start, _)| (*start..start + duration).contains(&t_s))
            .map_or(0.0, |(_, sign)| sign * rate)
    }
}

/// The 1-sigma of the in-flight alignment: position in meters, velocity
/// in m/s, and attitude in radians about each axis.
#[derive(Debug, Clone, Copy)]
pub(super) struct Alignment {
    pub(super) position_m: f64,
    pub(super) velocity_mps: f64,
    pub(super) attitude_rad: f64,
}

/// Index offsets that keep the draws of one seed apart. Each user of a
/// flight draws above [`FREE_INDEX`].
const POSITION: u32 = 1;
const VELOCITY: u32 = 4;
const TILT: u32 = 7;
const ACCEL_BIAS: u32 = 10;
const GYRO_BIAS: u32 = 13;
const IMU_NOISE: u32 = 100_000;
/// The first index that the flight does not use: the IMU noise uses six
/// indices per step, and no route here has a million steps.
pub(super) const FREE_INDEX: u32 = 10_000_000;

/// One run: the truth, the filter, and the IMU errors of one seed. The
/// filter has the layout of its configuration and the dimension `N`.
pub(super) struct Flight<const N: usize = FULL_STATES> {
    pub(super) route: Route,
    pub(super) seed: u32,
    pub(super) filter: InertialNavigationFilter<N>,
    pub(super) truth: InertialNominal,
    config: InertialConfig,
    regime: Regime,
    heading: f64,
    bank: f64,
    altitude_m: f64,
    accel_bias: Vector3<f64>,
    gyro_bias: Vector3<f64>,
    step: u32,
}

impl<const N: usize> Flight<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    /// A flight of `route` that starts north at `earth_origin`, and a
    /// filter aligned with the errors of `alignment`.
    pub(super) fn new(
        route: Route,
        seed: u32,
        config: InertialConfig,
        alignment: Alignment,
    ) -> Self {
        let regime = Regime::surface_of(&PlanetaryBody::EARTH_WGS84);
        let level = LocalLevel::at(earth_origin());
        let speed = route.speed_mps;
        let truth = InertialNominal::at(
            earth_body_fixed(&earth_origin()),
            level.ned_to_bf * Vector3::new(speed, 0.0, 0.0),
            quaternion_of(&level.ned_to_bf),
        );
        let Alignment {
            position_m,
            velocity_mps,
            attitude_rad,
        } = alignment;
        let start_p = truth.p + level.ned_to_bf * draw3(seed, POSITION, position_m);
        let start = body_fixed_to_geodetic(
            &BodyFixedPosition::new(start_p.x, start_p.y, start_p.z),
            BodyEllipsoid::WGS84,
        );
        let dv = draw3(seed, VELOCITY, velocity_mps);
        let tilt =
            UnitQuaternion::from_rotation_matrix(&Rotation3::new(draw3(seed, TILT, attitude_rad)));
        let (p2, v2, a2) = (
            position_m.powi(2),
            velocity_mps.powi(2),
            attitude_rad.powi(2),
        );
        let mut filter = InertialNavigationFilter::<N>::surface_with_layout(
            config,
            CLOCK,
            PlanetaryBody::EARTH_WGS84,
        )
        .expect("a layout of dimension N");
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
        Self {
            route,
            seed,
            filter,
            truth,
            config,
            regime,
            heading: 0.0,
            bank: 0.0,
            altitude_m: earth_origin().altitude_m,
            accel_bias: draw3(seed, ACCEL_BIAS, config.accel_bias.sigma()),
            gyro_bias: draw3(seed, GYRO_BIAS, config.gyro_bias.sigma()),
            step: 0,
        }
    }

    /// The IMU rates of one step of a coordinated turn that holds the
    /// altitude: the bank rolls toward the command at up to 0.5 rad/s,
    /// the heading turns at `g·tan φ / V`, and the specific force gives
    /// the centripetal acceleration.
    fn rates(&mut self, t_s: f64) -> ImuRates {
        let route = self.route;
        let dt = route.step_s;
        let level = LocalLevel::at_body_fixed(&self.truth.p, BodyEllipsoid::WGS84);
        let target = (route.turn_rate(t_s) * route.speed_mps / G).atan();
        self.bank += (target - self.bank).clamp(-0.5, 0.5) * dt;
        let rate = if route.speed_mps > 0.0 {
            G * self.bank.tan() / route.speed_mps
        } else {
            0.0
        };
        self.heading += rate * dt;
        let body_to_ned = Rotation3::from_euler_angles(self.bank, 0.0, self.heading);
        let c_target = level.ned_to_bf * body_to_ned.matrix();
        let c = self.truth.c_b_to_f();
        let omega = self.regime.frame_rotation();
        let turn = UnitQuaternion::from_matrix(&(c.transpose() * c_target));
        let v = level.bf_to_ned() * self.truth.v;
        let height = level.position.altitude_m - self.altitude_m;
        let a_ned = Vector3::new(-rate * v.y, rate * v.x, 0.02 * height - 0.3 * v.z);
        let (g, _) = self.regime.gravity(&self.truth.p);
        ImuRates {
            specific_force: c.transpose()
                * (level.ned_to_bf * a_ned - g + 2.0 * omega.cross(&self.truth.v)),
            angular_rate: rotation_vector(&turn) / dt + c.transpose() * omega,
        }
    }

    /// Flies one IMU step and gives the filter the measured sample. The
    /// result is the time of the step in milliseconds.
    pub(super) fn step(&mut self) -> u64 {
        let dt = self.route.step_s;
        let start_s = f64::from(self.step) * dt;
        self.step = self.step.wrapping_add(1);
        let k = self.step;
        let at_ms = u64::from(k) * (dt * 1_000.0).round() as u64;
        let rates = self.rates(start_s);
        self.truth = mechanize(&self.truth, &rates, dt, &self.regime, &self.config).0;
        let config = self.config;
        let accel_white = config.accel_noise_sigma_mps2_sqrt_hz / dt.sqrt();
        let gyro_white = config.gyro_noise_sigma_rad_s_sqrt_hz / dt.sqrt();
        let force = rates.specific_force
            + self.accel_bias
            + draw3(self.seed, IMU_NOISE + 6 * k, accel_white);
        let rate = rates.angular_rate
            + self.gyro_bias
            + draw3(self.seed, IMU_NOISE + 6 * k + 3, gyro_white);
        let sample = imu(k, at_ms, (force.into(), rate.into()), dt);
        assert_accepted(self.filter.ingest_imu(&sample, t(at_ms)));
        at_ms
    }

    /// The rotation from the truth body axes to the frame axes.
    pub(super) fn truth_body_to_frame(&self) -> Matrix3<f64> {
        self.truth.c_b_to_f()
    }

    /// The ellipsoidal height of the truth in meters.
    pub(super) fn truth_height_m(&self) -> f64 {
        LocalLevel::at_body_fixed(&self.truth.p, BodyEllipsoid::WGS84)
            .position
            .altitude_m
    }

    /// The horizontal NEES, the horizontal 1-sigma in meters, and the
    /// horizontal error in meters at `at_ms`.
    pub(super) fn horizontal(&mut self, at_ms: u64) -> (f64, f64, f64) {
        let solution = self.filter.tick(t(at_ms)).expect("published");
        let local = LocalLevel::at_body_fixed(&self.truth.p, BodyEllipsoid::WGS84);
        let e = local.bf_to_ned() * (core(&self.filter).nominal.p - self.truth.p);
        let [nn, ne, _, ee, _, _] = solution.position_cov.upper_triangle();
        let nees = (ee * e.x * e.x - 2.0 * ne * e.x * e.y + nn * e.y * e.y) / (nn * ee - ne * ne);
        (nees, (nn + ee).sqrt(), e.x.hypot(e.y))
    }

    /// The NEES of the geodetic height error: the height of the filter
    /// nominal less the height of the truth, against the vertical
    /// variance of the published solution.
    pub(super) fn height_nees(&mut self) -> f64 {
        let at = self.filter.tick(t(
            u64::from(self.step) * (self.route.step_s * 1_000.0).round() as u64
        ));
        let vertical = at.map_or(f64::NAN, |s| s.position_cov.upper_triangle()[5]);
        let height = |p: &Vector3<f64>| {
            LocalLevel::at_body_fixed(p, BodyEllipsoid::WGS84)
                .position
                .altitude_m
        };
        let error = height(&core(&self.filter).nominal.p) - height(&self.truth.p);
        error * error / vertical
    }

    /// The NEES of the 3-D position error in the frame axes against the
    /// position block of the error covariance.
    pub(super) fn position_nees(&self) -> f64 {
        let core = core(&self.filter);
        let error = core.nominal.p - self.truth.p;
        let p = core.p.fixed_view::<3, 3>(POS, POS).into_owned();
        p.try_inverse()
            .map_or(f64::INFINITY, |inverse| error.dot(&(inverse * error)))
    }

    /// The attitude NEES: the rotation from the filter attitude to the
    /// truth, in the body axes, against the attitude block of the error
    /// covariance. It has three degrees of freedom.
    pub(super) fn attitude_nees(&self) -> f64 {
        let core = core(&self.filter);
        let error = rotation_vector(&(core.nominal.q.inverse() * self.truth.q));
        let p = core.p.fixed_view::<3, 3>(ATT, ATT).into_owned();
        p.try_inverse()
            .map_or(f64::INFINITY, |inverse| error.dot(&(inverse * error)))
    }
}

#[test]
fn the_noise_is_standard_normal_white_and_independent_across_seeds() {
    // 100 000 draws of one seed: mean, variance, and lag-1 and lag-6
    // autocorrelation (the IMU draws six values per step), and the
    // correlation of two seeds over the same indices.
    let n = 100_000_u32;
    let draws = |seed: u32| (0..n).map(move |i| gauss(seed, i));
    let count = f64::from(n);
    let mean = draws(1).sum::<f64>() / count;
    let variance = draws(1).map(|z| (z - mean).powi(2)).sum::<f64>() / count;
    // 5-sigma bounds of the estimators for n independent N(0, 1) draws.
    let bound = 5.0 / count.sqrt();
    assert!(mean.abs() < bound, "mean {mean}");
    assert!(
        (variance - 1.0).abs() < 5.0 * (2.0 / count).sqrt(),
        "variance {variance}"
    );
    let values: std::vec::Vec<f64> = draws(1).collect();
    for lag in [1_usize, 6, 12] {
        let product = values
            .iter()
            .zip(values.iter().skip(lag))
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let rho = product / (count - lag as f64);
        assert!(rho.abs() < bound, "lag {lag}: {rho}");
    }
    let cross = draws(0).zip(draws(19)).map(|(a, b)| a * b).sum::<f64>() / count;
    assert!(cross.abs() < bound, "seeds 0 and 19: {cross}");
}
