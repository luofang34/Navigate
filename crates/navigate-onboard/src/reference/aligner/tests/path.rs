//! A moving vehicle and its sensors, for the tests of the alignment in
//! motion. The speed, the heading, and the attitude are functions of the
//! time. The specific force and the body rate come from the derivatives
//! of that motion on the rotating ellipsoid with formulas of this file,
//! so no test assumes what the accelerometers of a manoeuvre show.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{GeodeticPosition, NedVelocity};

use super::rig::{Budgets, GNSS, HEIGHT_M, IMU, LAT_DEG, LON_DEG, Noise, STEP_S, at};
use crate::{SensorData, SensorInput};

const EARTH_RATE: f64 = 7.292_115e-5;
const SEMI_MAJOR_M: f64 = 6_378_137.0;
const ECCENTRICITY_SQUARED: f64 = 6.694_379_990_141_316e-3;
/// Parts of one inertial step in the integration of the truth.
const PARTS: u32 = 4;
/// Half of the time span of a numeric derivative, seconds.
const DERIVATIVE_S: f64 = 1e-3;
/// One-sigma of the GNSS velocity of the suite of the rig, m/s.
const VELOCITY_SIGMA_M_S: f64 = 0.1;

/// What the vehicle does at one time.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Profile {
    /// Speed along the heading, m/s: over the ground on wheels, through
    /// the air in flight.
    pub speed_m_s: f64,
    pub heading_rad: f64,
    pub roll_rad: f64,
    pub pitch_rad: f64,
    pub climb_m_s: f64,
}

/// A motion that a test picks at run time.
pub(crate) type Manoeuvre = Box<dyn Fn(f64) -> Profile>;

/// The true state at the end of an inertial step.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Truth {
    /// Latitude and longitude in radians, height in metres.
    pub position: [f64; 3],
    /// North, east, down, m/s.
    pub velocity: Vector3<f64>,
    pub body_to_ned: UnitQuaternion<f64>,
}

/// A value that goes smoothly from zero at `from` to one at `to`, with a
/// continuous first and second derivative.
pub(crate) fn ramp(t: f64, from: f64, to: f64) -> f64 {
    let x = ((t - from) / (to - from)).clamp(0.0, 1.0);
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

/// The meridian and the prime-vertical radius of curvature, metres.
fn radii_m(lat: f64) -> (f64, f64) {
    let w = (1.0 - ECCENTRICITY_SQUARED * lat.sin().powi(2)).sqrt();
    (
        SEMI_MAJOR_M * (1.0 - ECCENTRICITY_SQUARED) / (w * w * w),
        SEMI_MAJOR_M / w,
    )
}

/// A vehicle that moves as `profile` says in an air mass that moves with
/// `wind`, and the errors of its sensors.
pub(crate) struct Vehicle<F> {
    profile: F,
    /// North, east, down, m/s.
    wind: Vector3<f64>,
    position: [f64; 3],
    /// Inertial steps since the start.
    pub step: u64,
    budgets: Budgets,
    accel_bias: Vector3<f64>,
    gyro_bias: Vector3<f64>,
    noise: Noise,
    /// One for sensors with the errors of the budgets, zero for sensors
    /// without error.
    errors: f64,
}

impl<F: Fn(f64) -> Profile> Vehicle<F> {
    /// A vehicle with sensor errors of `seed`, each drawn from its
    /// budget and scaled by `errors`.
    pub(crate) fn new(profile: F, wind_north_east: [f64; 2], seed: u64, errors: f64) -> Self {
        let budgets = Budgets::default();
        let mut noise = Noise::new(seed ^ 0x5EED);
        Self {
            profile,
            wind: Vector3::new(wind_north_east[0], wind_north_east[1], 0.0),
            position: [LAT_DEG.to_radians(), LON_DEG.to_radians(), HEIGHT_M],
            step: 0,
            budgets,
            accel_bias: noise.vector(budgets.accel_bias) * errors,
            gyro_bias: noise.vector(budgets.gyro_bias) * errors,
            noise,
            errors,
        }
    }

    fn velocity(&self, t: f64) -> Vector3<f64> {
        let p = (self.profile)(t);
        let (sin, cos) = p.heading_rad.sin_cos();
        Vector3::new(p.speed_m_s * cos, p.speed_m_s * sin, -p.climb_m_s) + self.wind
    }

    fn attitude(&self, t: f64) -> UnitQuaternion<f64> {
        let p = (self.profile)(t);
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), p.heading_rad)
            * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), p.pitch_rad)
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), p.roll_rad)
    }

    /// The rate of latitude, longitude, and height for a velocity.
    fn position_rate(position: [f64; 3], v: Vector3<f64>) -> [f64; 3] {
        let [lat, _, h] = position;
        let (meridian, vertical) = radii_m(lat);
        [
            v.x / (meridian + h),
            v.y / ((vertical + h) * lat.cos()),
            -v.z,
        ]
    }

    /// The specific force and the rate relative to inertial space, in the
    /// body axes, at a time and a position: the derivative of the
    /// velocity, the Coriolis and transport terms, and the reaction to
    /// normal gravity (Somigliana, with the free-air gradient).
    fn sensed(&self, t: f64, position: [f64; 3]) -> (Vector3<f64>, Vector3<f64>) {
        let [lat, _, h] = position;
        let (sin, cos) = lat.sin_cos();
        let (meridian, vertical) = radii_m(lat);
        let e = DERIVATIVE_S;
        let v = self.velocity(t);
        let acceleration = (self.velocity(t + e) - self.velocity(t - e)) / (2.0 * e);
        let earth = Vector3::new(cos, 0.0, -sin) * EARTH_RATE;
        let transport = Vector3::new(
            v.y / (vertical + h),
            -v.x / (meridian + h),
            -v.y * lat.tan() / (vertical + h),
        );
        let gravity = 9.780_325_335_9 * (1.0 + 0.001_931_852_652_41 * sin * sin)
            / (1.0 - 0.006_694_379_990_13 * sin * sin).sqrt()
            - 3.086e-6 * h;
        let force_ned =
            acceleration + (earth * 2.0 + transport).cross(&v) - Vector3::new(0.0, 0.0, gravity);
        let q = self.attitude(t);
        let turn = (self.attitude(t - e).inverse() * self.attitude(t + e)).scaled_axis();
        let ned_to_body = q.inverse();
        (
            ned_to_body * force_ned,
            ned_to_body * (earth + transport) + turn / (2.0 * e),
        )
    }

    /// The gyroscope bias that the vehicle's unit has, rad/s, body axes.
    pub(crate) const fn gyro_bias(&self) -> Vector3<f64> {
        self.gyro_bias
    }

    /// The true state at the end of the last step.
    pub(crate) fn truth(&self) -> Truth {
        let t = self.step as f64 * STEP_S;
        Truth {
            position: self.position,
            velocity: self.velocity(t),
            body_to_ned: self.attitude(t),
        }
    }

    /// Moves the vehicle by one inertial step and gives the increment of
    /// that step.
    pub(crate) fn increment(&mut self) -> SensorInput {
        self.increment_over(1)
    }

    /// Moves the vehicle by `steps` inertial steps and gives one
    /// increment for them as a unit of that rate gives it: the rotation
    /// vector, and the velocity increment without the rotation
    /// correction.
    pub(crate) fn increment_over(&mut self, steps: u32) -> SensorInput {
        let dt = STEP_S / f64::from(PARTS);
        let start = self.step as f64 * STEP_S;
        let span_s = STEP_S * f64::from(steps);
        // The rotation from the body axes at an instant into those at the
        // start of the step, and the velocity increment in those axes.
        let mut rotation = UnitQuaternion::identity();
        let mut in_start_axes = Vector3::zeros();
        for part in 0..PARTS * steps {
            let t = start + (f64::from(part) + 0.5) * dt;
            let rate = Self::position_rate(self.position, self.velocity(t));
            let middle = [0, 1, 2].map(|i| self.position[i] + rate[i] * 0.5 * dt);
            let (force, turn) = self.sensed(t, middle);
            in_start_axes +=
                rotation * UnitQuaternion::from_scaled_axis(turn * (0.5 * dt)) * force * dt;
            rotation *= UnitQuaternion::from_scaled_axis(turn * dt);
            let rate = Self::position_rate(middle, self.velocity(t));
            self.position = [0, 1, 2].map(|i| self.position[i] + rate[i] * dt);
        }
        self.step += u64::from(steps);
        let theta = rotation.scaled_axis();
        let velocity = in_start_axes - theta.cross(&in_start_axes) * 0.5;
        let per_sample = self.errors / span_s.sqrt();
        let b = self.budgets;
        let theta =
            theta + (self.gyro_bias + self.noise.vector(b.gyro_noise * per_sample)) * span_s;
        let velocity =
            velocity + (self.accel_bias + self.noise.vector(b.accel_noise * per_sample)) * span_s;
        SensorInput {
            sensor: IMU,
            acquired_at: at(self.step),
            delivered_at: at(self.step),
            data: SensorData::ImuIncrement {
                dt_s: span_s,
                delta_theta_rad: theta.into(),
                delta_velocity_m_s: velocity.into(),
            },
        }
    }

    /// A GNSS fix of the vehicle now, with the noise of the receiver,
    /// that reaches the navigator `lag` steps later.
    pub(crate) fn gnss(&mut self, lag: u64) -> SensorInput {
        let b = self.budgets;
        let truth = self.truth();
        let [lat, lon, h] = truth.position;
        let (meridian, vertical) = radii_m(lat);
        let error = Vector3::new(
            self.noise.normal() * b.gnss_horizontal_m,
            self.noise.normal() * b.gnss_horizontal_m,
            self.noise.normal() * b.gnss_vertical_m,
        ) * self.errors;
        let v = truth.velocity + self.noise.vector(VELOCITY_SIGMA_M_S) * self.errors;
        SensorInput {
            sensor: GNSS,
            acquired_at: at(self.step),
            delivered_at: at(self.step + lag),
            data: SensorData::GnssFix {
                position: GeodeticPosition::new(
                    lat + error.x / (meridian + h),
                    lon + error.y / ((vertical + h) * lat.cos()),
                    h - error.z,
                ),
                velocity: Some(NedVelocity::new(v.x, v.y, v.z)),
                position_cov: None,
                velocity_cov: None,
            },
        }
    }
}

impl Truth {
    /// The error of a position against this truth: north, east, down,
    /// metres.
    pub(crate) fn position_error_m(&self, position: &GeodeticPosition) -> Vector3<f64> {
        let [lat, lon, h] = self.position;
        let (meridian, vertical) = radii_m(lat);
        Vector3::new(
            (position.latitude_rad - lat) * (meridian + h),
            (position.longitude_rad - lon) * (vertical + h) * lat.cos(),
            h - position.altitude_m,
        )
    }
}
