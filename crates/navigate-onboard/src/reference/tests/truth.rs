//! A model-matched truth for Monte Carlo runs of the reference
//! navigator: an inertial filter of Navigate that integrates exact
//! increments of a level flight, an IMU with the errors of its budget,
//! and the draws of one seed.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{
    AttitudeQuaternion, ClockDomainId, GeodeticPosition, MonotonicNanos, NavigationSolution,
    NedVelocity, ObservationStamp, SensorClass, SourceComposition, SourceEpoch, SourceId,
    SymmetricCov3, WrappingSequence,
};
use navigate_fusion::{
    FogmBias, ImuIncrement, InertialConfig, InertialNavigationFilter, PlanetaryBody,
};

use crate::{Alignment, Navigator, ReferenceNavigator, SensorData, SensorInput};

pub(crate) const CLOCK: ClockDomainId = ClockDomainId::new(7);
pub(crate) const IMU: SourceId = SourceId::new(1);
const LAT_DEG: f64 = 41.7;
const LON_DEG: f64 = -1.0;
const HEIGHT_M: f64 = 1_000.0;
pub(crate) const SPEED_MPS: f64 = 40.0;
pub(crate) const STEP_S: f64 = 0.1;
const EARTH_RATE: f64 = 7.292_115e-5;
const EARTH_RADIUS_M: f64 = 6_378_137.0;

/// SplitMix64 finalizer.
pub(crate) fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Independent standard normal draws for one seed.
pub(crate) struct Draws {
    pub(crate) seed: u64,
    pub(crate) index: u64,
}

impl Draws {
    pub(crate) fn normal(&mut self) -> f64 {
        let mut uniform = || {
            self.index = self.index.wrapping_add(1);
            let bits = mix((self.seed << 40) ^ self.index) >> 11;
            (bits as f64 + 0.5) / (1_u64 << 53) as f64
        };
        let (a, b) = (uniform(), uniform());
        (-2.0 * a.ln()).sqrt() * (core::f64::consts::TAU * b).cos()
    }

    pub(crate) fn vec3(&mut self, sigma: [f64; 3]) -> Vector3<f64> {
        Vector3::new(
            self.normal() * sigma[0],
            self.normal() * sigma[1],
            self.normal() * sigma[2],
        )
    }
}

pub(crate) fn at(step: u32) -> MonotonicNanos {
    MonotonicNanos::from_nanos(1_000_000_000 + u64::from(step) * 100_000_000)
}

/// The tactical IMU of the simulator trial.
pub(crate) fn imu_budget() -> (f64, f64, FogmBias, FogmBias) {
    let per_sqrt_hour = 1.0 / 60.0;
    (
        0.05 * per_sqrt_hour,
        0.1_f64.to_radians() * per_sqrt_hour,
        FogmBias::new(3_600.0, 1e-3 * 9.806_65),
        FogmBias::new(3_600.0, 1.0_f64.to_radians() / 3_600.0),
    )
}

/// The truth: an inertial filter of Navigate that integrates exact
/// increments of a level flight with coordinated heading changes.
pub(crate) struct Truth {
    filter: Box<InertialNavigationFilter>,
    sequence: WrappingSequence,
    pub(crate) state: NavigationSolution,
    /// The heading rate at a time from the start, rad/s.
    turn: fn(f64) -> f64,
}

pub(crate) fn start() -> GeodeticPosition {
    GeodeticPosition::new(LAT_DEG.to_radians(), LON_DEG.to_radians(), HEIGHT_M)
}

impl Truth {
    pub(crate) fn new(turn: fn(f64) -> f64) -> Self {
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
            turn,
        }
    }

    /// The exact increments of the step that starts at `t_s`, and the
    /// truth after it.
    pub(crate) fn step(&mut self, step: u32, t_s: f64) -> ([f64; 3], [f64; 3]) {
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
        let turn = (self.turn)(t_s);
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
pub(crate) fn ned_offset(a: &GeodeticPosition, b: &GeodeticPosition) -> Vector3<f64> {
    let r = EARTH_RADIUS_M + a.altitude_m;
    Vector3::new(
        (b.latitude_rad - a.latitude_rad) * r,
        (b.longitude_rad - a.longitude_rad) * r * a.latitude_rad.cos(),
        a.altitude_m - b.altitude_m,
    )
}

/// `p` moved by the NED offset `d`, metres.
pub(crate) fn moved(p: &GeodeticPosition, d: &Vector3<f64>) -> GeodeticPosition {
    let r = EARTH_RADIUS_M + p.altitude_m;
    GeodeticPosition::new(
        p.latitude_rad + d.x / r,
        p.longitude_rad + d.y / (r * p.latitude_rad.cos()),
        p.altitude_m - d.z,
    )
}

pub(crate) fn nees(error: &Vector3<f64>, cov: &SymmetricCov3) -> f64 {
    let [xx, xy, xz, yy, yz, zz] = cov.upper_triangle();
    let p = nalgebra::Matrix3::new(xx, xy, xz, xy, yy, yz, xz, yz, zz);
    p.try_inverse()
        .map_or(f64::INFINITY, |i| error.dot(&(i * error)))
}

/// The IMU of one run: its biases, drawn once, and its white noise.
pub(crate) struct Imu {
    b_a: Vector3<f64>,
    b_g: Vector3<f64>,
}

impl Imu {
    /// The navigator aligned with the errors of a level in-flight
    /// alignment, and the IMU of the run.
    pub(crate) fn align(navigator: &mut ReferenceNavigator, draws: &mut Draws) -> Self {
        let (_, _, accel_bias, gyro_bias) = imu_budget();
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
            })
            .unwrap();
        Self {
            b_a: draws.vec3([accel_bias.sigma(); 3]),
            b_g: draws.vec3([gyro_bias.sigma(); 3]),
        }
    }

    /// Flies the truth one step and gives the navigator the measured
    /// increments.
    pub(crate) fn step(
        &self,
        truth: &mut Truth,
        navigator: &mut ReferenceNavigator,
        draws: &mut Draws,
        step: u32,
    ) {
        let (accel, gyro, _, _) = imu_budget();
        let (theta, dv) = truth.step(step, f64::from(step - 1) * STEP_S);
        let theta =
            Vector3::from(theta) + self.b_g * STEP_S + draws.vec3([gyro * STEP_S.sqrt(); 3]);
        let dv = Vector3::from(dv) + self.b_a * STEP_S + draws.vec3([accel * STEP_S.sqrt(); 3]);
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
    }
}
