//! A vehicle and its sensors, for the alignment tests: at rest, in steady
//! straight motion, or from rest into a forward acceleration. The true
//! specific force, body rate, and magnetic field come from the true
//! attitude and motion with formulas of this file; no value of the truth
//! reaches the navigator except through a sensor.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{ClockDomainId, GeodeticPosition, MonotonicNanos, NedVelocity, SourceId};
use navigate_fusion::FogmBias;

use crate::{
    ErrorBudget, MagneticReference, Navigator, ReferenceNavigator, SensorData, SensorInput,
    SensorKind, SensorSpec, SensorSuite,
};

pub(crate) const CLOCK: ClockDomainId = ClockDomainId::new(9);
pub(crate) const IMU: SourceId = SourceId::new(1);
pub(crate) const GNSS: SourceId = SourceId::new(2);
pub(crate) const MAGNETOMETER: SourceId = SourceId::new(3);
pub(crate) const SURVEY: SourceId = SourceId::new(4);

/// One inertial step: 100 Hz.
pub(crate) const STEP_S: f64 = 0.01;
pub(crate) const LAT_DEG: f64 = 41.7;
pub(crate) const LON_DEG: f64 = -1.0;
pub(crate) const HEIGHT_M: f64 = 300.0;
/// Horizontal intensity of the Earth's field at the place: 45 µT with a
/// dip of 57°.
pub(crate) const HORIZONTAL_FIELD_T: f64 = 45e-6 * 0.544_639_035;
/// Total intensity of the Earth's field at the place, tesla.
pub(crate) const FIELD_T: f64 = 45e-6;

/// The declared errors of the sensors of the rig.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Budgets {
    pub accel_noise: f64,
    pub accel_bias: f64,
    pub gyro_noise: f64,
    pub gyro_bias: f64,
    pub mag_noise_rad: f64,
    pub mag_disturbance_rad: f64,
    pub declination_rad: f64,
    pub declination_sigma_rad: f64,
    pub gnss_horizontal_m: f64,
    pub gnss_vertical_m: f64,
    pub gnss_velocity_m_s: f64,
}

impl Default for Budgets {
    /// A small tactical unit: 2 mg of accelerometer bias, 0.06 °/s of
    /// gyroscope bias, a magnetometer with 2° of body disturbance, a
    /// declination known to 0.5°, and GNSS of 1.5 m and 3 m.
    fn default() -> Self {
        Self {
            accel_noise: 2e-3,
            accel_bias: 0.02,
            gyro_noise: 1e-4,
            gyro_bias: 1e-3,
            mag_noise_rad: 0.02,
            mag_disturbance_rad: 0.035,
            declination_rad: 0.026,
            declination_sigma_rad: 0.009,
            gnss_horizontal_m: 1.5,
            gnss_vertical_m: 3.0,
            gnss_velocity_m_s: 0.1,
        }
    }
}

/// Normal numbers from a seed: a xorshift generator and the Box–Muller
/// transform.
pub(crate) struct Noise(u64);

impl Noise {
    /// The state is the seed through one round of a mixing function, so
    /// that a small seed does not give a state of few bits, whose first
    /// numbers are far in the tail.
    pub(crate) fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    pub(crate) fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
    }

    pub(crate) fn vector(&mut self, sigma: f64) -> Vector3<f64> {
        Vector3::new(self.normal(), self.normal(), self.normal()) * sigma
    }
}

/// A vehicle at rest with a true attitude and the errors of its sensors,
/// each drawn once from its declared budget.
pub(crate) struct Rig {
    pub budgets: Budgets,
    /// Rotation from the body axes into north, east, down.
    pub body_to_ned: UnitQuaternion<f64>,
    pub accel_bias: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,
    /// Error of the declared declination, radians.
    pub declination_error_rad: f64,
    /// The field of the body after calibration, tesla, body axes: hard
    /// iron, which turns with the vehicle.
    pub hard_iron_t: Vector3<f64>,
    /// The steady velocity of the vehicle, north, east, down, m/s.
    pub velocity_ned: Vector3<f64>,
    /// The step from which the vehicle accelerates, and the acceleration,
    /// north, east, down, m/s².
    pub roll: Option<(u64, Vector3<f64>)>,
    noise: Noise,
}

/// Shows the `tracing` events of a test with `--nocapture`.
pub(crate) fn show_events() {
    tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .try_init()
        .ok();
}

/// The time of an inertial step.
pub(crate) fn at(step: u64) -> MonotonicNanos {
    MonotonicNanos::from_nanos(1_000_000_000 + step * 10_000_000)
}

impl Rig {
    /// A vehicle with roll, pitch, and heading in degrees, and sensor
    /// errors of `seed`. `declination_error` scales the drawn error of the
    /// declared declination: one for a declaration that is honest.
    pub(crate) fn new(attitude_deg: [f64; 3], seed: u64, declination_error: f64) -> Self {
        let budgets = Budgets::default();
        let mut noise = Noise::new(seed);
        let [roll, pitch, heading] = attitude_deg.map(f64::to_radians);
        // Heading about down, then pitch about the right axis, then roll
        // about the forward axis.
        let body_to_ned = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), heading)
            * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), pitch)
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), roll);
        let accel_bias = noise.vector(budgets.accel_bias);
        let gyro_bias = noise.vector(budgets.gyro_bias);
        // A hard-iron component across the horizontal field of one sigma
        // of the declared disturbance turns the heading by that sigma.
        let hard_iron_t = noise.vector(budgets.mag_disturbance_rad * HORIZONTAL_FIELD_T);
        let declination_error_rad =
            noise.normal() * budgets.declination_sigma_rad * declination_error;
        Self {
            budgets,
            body_to_ned,
            accel_bias,
            gyro_bias,
            declination_error_rad,
            hard_iron_t,
            velocity_ned: Vector3::zeros(),
            roll: None,
            noise,
        }
    }

    /// The acceleration of the vehicle at `step`, north, east, down.
    fn acceleration(&self, step: u64) -> Vector3<f64> {
        match self.roll {
            Some((from, acceleration)) if step > from => acceleration,
            _ => Vector3::zeros(),
        }
    }

    /// The velocity of the vehicle at `step`, north, east, down.
    pub(crate) fn velocity(&self, step: u64) -> Vector3<f64> {
        let rolled = match self.roll {
            Some((from, acceleration)) if step > from => {
                acceleration * ((step - from) as f64 * STEP_S)
            }
            _ => Vector3::zeros(),
        };
        self.velocity_ned + rolled
    }

    fn input(sensor: SourceId, step: u64, data: SensorData) -> SensorInput {
        SensorInput {
            sensor,
            acquired_at: at(step),
            delivered_at: at(step),
            data,
        }
    }

    /// The inertial increment of the step that ends at `step`: the
    /// reaction to gravity and the Earth rate in the body axes, with the
    /// bias and the noise of the unit.
    pub(crate) fn increment(&mut self, step: u64) -> SensorInput {
        let (s, c) = LAT_DEG.to_radians().sin_cos();
        let gravity = 9.780_325_335_9 * (1.0 + 0.001_931_852_652_41 * s * s)
            / (1.0 - 0.006_694_379_990_13 * s * s).sqrt()
            - 3.086e-6 * HEIGHT_M;
        let ned_to_body = self.body_to_ned.inverse();
        let per_sample = 1.0 / STEP_S.sqrt();
        let force = ned_to_body * (self.acceleration(step) - Vector3::new(0.0, 0.0, gravity))
            + self.accel_bias
            + self.noise.vector(self.budgets.accel_noise * per_sample);
        let rate = ned_to_body * (Vector3::new(c, 0.0, -s) * 7.292_115e-5)
            + self.gyro_bias
            + self.noise.vector(self.budgets.gyro_noise * per_sample);
        Self::input(
            IMU,
            step,
            SensorData::ImuIncrement {
                dt_s: STEP_S,
                delta_theta_rad: (rate * STEP_S).into(),
                delta_velocity_m_s: (force * STEP_S).into(),
            },
        )
    }

    /// The field of the Earth at the place, 45 µT with a dip of 57°, as
    /// the magnetometer gives it: the declination error and the noise turn
    /// the field about the vertical, and the hard iron adds in the body
    /// axes.
    pub(crate) fn magnetic(&mut self, step: u64) -> SensorInput {
        let declination = self.budgets.declination_rad + self.declination_error_rad;
        let dip = 57.0_f64.to_radians();
        let field_ned = Vector3::new(
            dip.cos() * declination.cos(),
            dip.cos() * declination.sin(),
            dip.sin(),
        ) * FIELD_T;
        let noise = self
            .noise
            .vector(self.budgets.mag_noise_rad * HORIZONTAL_FIELD_T);
        let field_body_t: [f64; 3] =
            (self.body_to_ned.inverse() * field_ned + self.hard_iron_t + noise).into();
        Self::input(
            MAGNETOMETER,
            step,
            SensorData::MagneticField { field_body_t },
        )
    }

    /// A GNSS fix of the vehicle with the noise of the receiver, and the
    /// position error that it has, north, east, down, metres. The vehicle
    /// is at the reference point at step zero and moves with its
    /// velocity; the error is from the reference point.
    pub(crate) fn gnss(&mut self, step: u64, with_velocity: bool) -> (SensorInput, Vector3<f64>) {
        let b = self.budgets;
        let travelled = self.velocity_ned * (step as f64 * STEP_S);
        let error = travelled
            + Vector3::new(
                self.noise.normal() * b.gnss_horizontal_m,
                self.noise.normal() * b.gnss_horizontal_m,
                self.noise.normal() * b.gnss_vertical_m,
            );
        let lat = LAT_DEG.to_radians();
        let position = GeodeticPosition::new(
            lat + error.x / 6_364_000.0,
            LON_DEG.to_radians() + error.y / (6_378_137.0 * lat.cos()),
            HEIGHT_M - error.z,
        );
        let v = self.velocity(step) + self.noise.vector(b.gnss_velocity_m_s);
        let fix = SensorData::GnssFix {
            position,
            velocity: with_velocity.then(|| NedVelocity::new(v.x, v.y, v.z)),
            position_cov: None,
            velocity_cov: None,
        };
        (Self::input(GNSS, step, fix), error)
    }

    /// The statement that the vehicle is at rest on the surveyed point.
    pub(crate) fn surveyed(step: u64) -> SensorInput {
        let position = GeodeticPosition::new(LAT_DEG.to_radians(), LON_DEG.to_radians(), HEIGHT_M);
        Self::input(
            SURVEY,
            step,
            SensorData::SurveyedPosition {
                position,
                horizontal_sigma_m: 0.5,
                vertical_sigma_m: 0.5,
            },
        )
    }

    /// The suite of the rig, with a magnetometer if `magnetometer` gives
    /// its body disturbance.
    pub(crate) fn suite(&self, magnetometer: Option<f64>) -> SensorSuite {
        let b = self.budgets;
        self.suite_with(
            magnetometer,
            ErrorBudget::Gnss {
                horizontal_sigma_m: b.gnss_horizontal_m,
                vertical_sigma_m: b.gnss_vertical_m,
                velocity_sigma_m_s: b.gnss_velocity_m_s,
            },
        )
    }

    /// The suite of the rig with the GNSS budget `gnss`.
    pub(crate) fn suite_with(&self, magnetometer: Option<f64>, gnss: ErrorBudget) -> SensorSuite {
        let b = self.budgets;
        let spec = |id, kind, budget| SensorSpec {
            id,
            kind,
            imager: None,
            mount: None,
            rate_hz: 1.0,
            budget,
        };
        let mut sensors = vec![
            spec(
                IMU,
                SensorKind::Imu,
                ErrorBudget::Imu {
                    accel_noise_mps2_sqrt_hz: b.accel_noise,
                    gyro_noise_rad_s_sqrt_hz: b.gyro_noise,
                    accel_bias: FogmBias::new(3_600.0, b.accel_bias),
                    gyro_bias: FogmBias::new(3_600.0, b.gyro_bias),
                },
            ),
            spec(GNSS, SensorKind::Gnss, gnss),
            spec(SURVEY, SensorKind::SurveyedPoint, ErrorBudget::Declared),
        ];
        if let Some(disturbance_sigma_rad) = magnetometer {
            sensors.push(spec(
                MAGNETOMETER,
                SensorKind::Magnetometer,
                ErrorBudget::Magnetometer {
                    noise_sigma_t: b.mag_noise_rad * HORIZONTAL_FIELD_T,
                    disturbance_sigma_t: disturbance_sigma_rad * HORIZONTAL_FIELD_T,
                },
            ));
        }
        SensorSuite::new(sensors).unwrap()
    }

    /// A navigator of the suite of the rig that no one aligned, told the
    /// field of the place.
    pub(crate) fn navigator(&self) -> ReferenceNavigator {
        let suite = self.suite(Some(self.budgets.mag_disturbance_rad));
        self.told(ReferenceNavigator::new(suite, CLOCK, None).unwrap())
    }

    /// The field of the place as the host states it: the declared
    /// declination and its one-sigma, and the intensity to half a
    /// microtesla.
    pub(crate) fn reference(&self) -> MagneticReference {
        MagneticReference {
            declination_rad: self.budgets.declination_rad,
            declination_sigma_rad: self.budgets.declination_sigma_rad,
            intensity_t: FIELD_T,
            intensity_sigma_t: 0.5e-6,
        }
    }

    /// `navigator` told the field of the place.
    pub(crate) fn told(&self, mut navigator: ReferenceNavigator) -> ReferenceNavigator {
        navigator.set_magnetic_reference(self.reference()).unwrap();
        navigator
    }

    /// A navigator of the suite of the rig with the GNSS budget `gnss`.
    pub(crate) fn navigator_with_gnss(&self, gnss: ErrorBudget) -> ReferenceNavigator {
        let suite = self.suite_with(Some(self.budgets.mag_disturbance_rad), gnss);
        self.told(ReferenceNavigator::new(suite, CLOCK, None).unwrap())
    }
}
