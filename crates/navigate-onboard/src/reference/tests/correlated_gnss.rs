//! A GNSS receiver with a time-correlated error at 10 Hz on a turning
//! route: the 3-D position NEES of the reference navigator over many
//! seeds, with the correlated part declared in the budget and with the
//! whole error declared as white. The truth runs through an inertial
//! filter of Navigate fed with exact increments, so the truth and the
//! navigator integrate with the same mechanization.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{
    AttitudeQuaternion, ClockDomainId, GeodeticPosition, MonotonicNanos, NavigationSolution,
    NedVelocity, ObservationStamp, SensorClass, SourceComposition, SourceEpoch, SourceId,
    SymmetricCov3, WrappingSequence,
};
use navigate_fusion::{
    FogmBias, ImuIncrement, InertialConfig, InertialNavigationFilter, PlanetaryBody,
};

use crate::sensor::SensorSpec;
use crate::{
    Alignment, ErrorBudget, GnssCorrelatedError, Navigator, ReferenceNavigator, SensorData,
    SensorInput, SensorKind, SensorSuite,
};

const CLOCK: ClockDomainId = ClockDomainId::new(7);
const IMU: SourceId = SourceId::new(1);
const GNSS: SourceId = SourceId::new(2);
const LAT_DEG: f64 = 41.7;
const LON_DEG: f64 = -1.0;
const HEIGHT_M: f64 = 1_000.0;
const SPEED_MPS: f64 = 40.0;
/// The turn rate of a 25° bank at 40 m/s.
const TURN_RAD_S: f64 = 0.114;
const STEP_S: f64 = 0.1;
const EARTH_RATE: f64 = 7.292_115e-5;
const EARTH_RADIUS_M: f64 = 6_378_137.0;
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

/// SplitMix64 finalizer.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Independent standard normal draws for one seed.
struct Draws {
    seed: u64,
    index: u64,
}

impl Draws {
    fn normal(&mut self) -> f64 {
        let mut uniform = || {
            self.index = self.index.wrapping_add(1);
            let bits = mix((self.seed << 40) ^ self.index) >> 11;
            (bits as f64 + 0.5) / (1_u64 << 53) as f64
        };
        let (a, b) = (uniform(), uniform());
        (-2.0 * a.ln()).sqrt() * (core::f64::consts::TAU * b).cos()
    }

    fn vec3(&mut self, sigma: [f64; 3]) -> Vector3<f64> {
        Vector3::new(
            self.normal() * sigma[0],
            self.normal() * sigma[1],
            self.normal() * sigma[2],
        )
    }
}

fn at(step: u32) -> MonotonicNanos {
    MonotonicNanos::from_nanos(1_000_000_000 + u64::from(step) * 100_000_000)
}

/// The tactical IMU of the simulator trial.
fn imu_budget() -> (f64, f64, FogmBias, FogmBias) {
    let per_sqrt_hour = 1.0 / 60.0;
    (
        0.05 * per_sqrt_hour,
        0.1_f64.to_radians() * per_sqrt_hour,
        FogmBias::new(3_600.0, 1e-3 * 9.806_65),
        FogmBias::new(3_600.0, 1.0_f64.to_radians() / 3_600.0),
    )
}

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

/// The truth: an inertial filter of Navigate that integrates exact
/// increments of a level flight with coordinated heading changes.
struct Truth {
    filter: Box<InertialNavigationFilter>,
    sequence: WrappingSequence,
    state: NavigationSolution,
}

fn start() -> GeodeticPosition {
    GeodeticPosition::new(LAT_DEG.to_radians(), LON_DEG.to_radians(), HEIGHT_M)
}

impl Truth {
    fn new() -> Self {
        let mut filter = Box::new(InertialNavigationFilter::new_surface(
            InertialConfig::default(),
            CLOCK,
            PlanetaryBody::EARTH_WGS84,
        ));
        let tiny = SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6);
        filter
            .initialize_surface(
                start(),
                NedVelocity::new(SPEED_MPS, 0.0, 0.0),
                AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
                tiny,
                tiny,
                tiny,
                at(0),
                SourceComposition::of(SensorClass::Inertial),
            )
            .unwrap();
        let state = filter.tick(at(0)).unwrap();
        Self {
            filter,
            sequence: WrappingSequence::new(0),
            state,
        }
    }

    /// The exact increments of the step that starts at `t_s`, and the
    /// truth after it.
    fn step(&mut self, step: u32, t_s: f64) -> ([f64; 3], [f64; 3]) {
        let s = &self.state;
        let q = s.attitude.unwrap();
        let c = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q.w, q.x, q.y, q.z));
        let v = Vector3::new(
            s.velocity.north_mps,
            s.velocity.east_mps,
            s.velocity.down_mps,
        );
        let (sin, cos) = s.position.latitude_rad.sin_cos();
        let r = EARTH_RADIUS_M + s.position.altitude_m;
        let earth = Vector3::new(EARTH_RATE * cos, 0.0, -EARTH_RATE * sin);
        let transport = Vector3::new(v.y / r, -v.x / r, -v.y * sin / cos / r);
        let turn = TURNS_S
            .iter()
            .find(|(begin, _)| {
                (*begin..begin + core::f64::consts::FRAC_PI_2 / TURN_RAD_S).contains(&t_s)
            })
            .map_or(0.0, |(_, sign)| sign * TURN_RAD_S);
        // The centripetal acceleration of the turn, and a hold of the
        // start height.
        let height_error = s.position.altitude_m - HEIGHT_M;
        let acceleration = Vector3::new(-turn * v.y, turn * v.x, 0.02 * height_error - 0.3 * v.z);
        let g = Vector3::new(0.0, 0.0, 9.803);
        let force = acceleration - g + (2.0 * earth + transport).cross(&v);
        let rate = c.inverse() * (earth + transport) + Vector3::new(0.0, 0.0, turn);
        let theta = (rate * STEP_S).into();
        let dv = (c.inverse() * force * STEP_S).into();
        let stamp = ObservationStamp::new(IMU, SourceEpoch::new(0), self.sequence, at(step), CLOCK);
        self.sequence = self.sequence.next();
        let increment = ImuIncrement::new(
            stamp,
            SourceComposition::of(SensorClass::Inertial),
            STEP_S,
            theta,
            dv,
        );
        assert!(
            self.filter
                .ingest_imu_increment(&increment, at(step))
                .is_accepted()
        );
        self.state = self.filter.tick(at(step)).unwrap();
        (theta, dv)
    }
}

/// The NED offset of `b` from `a`, metres.
fn ned_offset(a: &GeodeticPosition, b: &GeodeticPosition) -> Vector3<f64> {
    let r = EARTH_RADIUS_M + a.altitude_m;
    Vector3::new(
        (b.latitude_rad - a.latitude_rad) * r,
        (b.longitude_rad - a.longitude_rad) * r * a.latitude_rad.cos(),
        a.altitude_m - b.altitude_m,
    )
}

/// `p` moved by the NED offset `d`, metres.
fn moved(p: &GeodeticPosition, d: &Vector3<f64>) -> GeodeticPosition {
    let r = EARTH_RADIUS_M + p.altitude_m;
    GeodeticPosition::new(
        p.latitude_rad + d.x / r,
        p.longitude_rad + d.y / (r * p.latitude_rad.cos()),
        p.altitude_m - d.z,
    )
}

fn nees(error: &Vector3<f64>, cov: &SymmetricCov3) -> f64 {
    let [xx, xy, xz, yy, yz, zz] = cov.upper_triangle();
    let p = nalgebra::Matrix3::new(xx, xy, xz, xy, yy, yz, xz, yz, zz);
    p.try_inverse()
        .map_or(f64::INFINITY, |i| error.dot(&(i * error)))
}

/// The 3-D position NEES of one run at each report time.
fn run(seed: u32, correlated: bool) -> Vec<f64> {
    let mut draws = Draws {
        seed: u64::from(seed),
        index: 0,
    };
    let mut truth = Truth::new();
    let mut navigator = ReferenceNavigator::new(suite(correlated), CLOCK, None).unwrap();
    let (accel, gyro, accel_bias, gyro_bias) = imu_budget();
    let start_error = draws.vec3([2.0; 3]);
    let dv = draws.vec3([0.1; 3]);
    let tilt = UnitQuaternion::from_scaled_axis(draws.vec3([1e-3; 3]));
    navigator
        .align(&Alignment {
            position: moved(&start(), &start_error),
            velocity: NedVelocity::new(SPEED_MPS + dv.x, dv.y, dv.z),
            attitude: AttitudeQuaternion::new(tilt.w, tilt.i, tilt.j, tilt.k),
            position_cov: SymmetricCov3::from_diagonal(4.0, 4.0, 4.0),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            at: at(0),
            imu_bias_prior: None,
        })
        .unwrap();
    let b_a = draws.vec3([accel_bias.sigma(); 3]);
    let b_g = draws.vec3([gyro_bias.sigma(); 3]);
    let mut correlated_error = draws.vec3(CORRELATED_M);
    let rho = (-STEP_S / TAU_S).exp();
    let innovation = CORRELATED_M.map(|s| s * (1.0 - rho * rho).sqrt());
    let mut out = Vec::new();
    let steps = (DURATION_S / STEP_S).round() as u32;
    for step in 1..=steps {
        let t_s = f64::from(step - 1) * STEP_S;
        let (theta, dv) = truth.step(step, t_s);
        let theta = Vector3::from(theta) + b_g * STEP_S + draws.vec3([gyro * STEP_S.sqrt(); 3]);
        let dv = Vector3::from(dv) + b_a * STEP_S + draws.vec3([accel * STEP_S.sqrt(); 3]);
        let imu = SensorInput {
            sensor: IMU,
            acquired_at: at(step),
            delivered_at: at(step),
            data: SensorData::ImuIncrement {
                dt_s: STEP_S,
                delta_theta_rad: theta.into(),
                delta_velocity_m_s: dv.into(),
            },
        };
        navigator.ingest(&imu, &mut ()).unwrap();
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
