//! Self-alignment of the reference navigator at rest: roll and pitch from
//! the specific force, heading from the magnetometer, and position from
//! GNSS or a surveyed point.
//!
//! The alignment works in rest windows. A window gives a state when, in
//! that one window, the vehicle is at rest by its inertial data and by
//! the evidence of another sensor, a heading source is good enough, and a
//! position came. Each window levels again, so no part of a state is
//! older than its window. The inertial data alone cannot show rest:
//! steady straight flight has the specific force and the body rate of
//! rest.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{
    AttitudeQuaternion, GeodeticPosition, MonotonicNanos, NedVelocity, SymmetricCov3,
};

use crate::{
    Alignment, AlignmentConfig, AlignmentPhase, AlignmentRecord, AlignmentSource, ErrorBudget,
    HeadingSource, PositionSource, SensorData, SensorKind, SensorSuite,
};

pub(crate) mod names;
mod state;
mod window;

use window::Window;

/// Normal gravity at the surface, m/s², for the test that a mean specific
/// force is that of rest. The tilt uses the measured magnitude.
const GRAVITY_MPS2: f64 = 9.806_65;
/// Largest difference of the mean specific force from gravity at rest,
/// m/s²: gravity changes by 0.05 m/s² over the Earth, and an
/// accelerometer bias adds to it.
const REST_GRAVITY_BAND_MPS2: f64 = 0.5;
/// Rotation rate of the Earth, rad/s.
const EARTH_RATE_RAD_S: f64 = 7.292_115e-5;

/// The declared errors of the magnetometer.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Magnetometer {
    noise_sigma_rad: f64,
    disturbance_sigma_rad: f64,
    declination_rad: f64,
    declination_sigma_rad: f64,
}

/// A state that one rest window gave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Solved {
    pub alignment: Alignment,
    pub record: AlignmentRecord,
}

/// The self-alignment of one navigator.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Aligner {
    pub(crate) config: AlignmentConfig,
    /// Accelerometer white noise density, (m/s²)/√Hz, and bias one-sigma,
    /// m/s².
    accelerometer: (f64, f64),
    magnetometer: Option<Magnetometer>,
    window: Window,
    waiting: (AlignmentPhase, &'static str),
}

impl Aligner {
    /// The aligner of the sensors of `suite`.
    pub(crate) fn new(suite: &SensorSuite) -> Self {
        let budget = |kind| suite.of_kind(kind).next().map(|spec| spec.budget);
        let accelerometer = match budget(SensorKind::Imu) {
            Some(ErrorBudget::Imu {
                accel_noise_mps2_sqrt_hz,
                accel_bias,
                ..
            }) => (accel_noise_mps2_sqrt_hz, accel_bias.sigma()),
            _ => (0.0, 0.0),
        };
        let magnetometer = match budget(SensorKind::Magnetometer) {
            Some(ErrorBudget::Magnetometer {
                noise_sigma_rad,
                disturbance_sigma_rad,
                declination_rad,
                declination_sigma_rad,
            }) => Some(Magnetometer {
                noise_sigma_rad,
                disturbance_sigma_rad,
                declination_rad,
                declination_sigma_rad,
            }),
            _ => None,
        };
        Self {
            config: AlignmentConfig::default(),
            accelerometer,
            magnetometer,
            window: Window::default(),
            waiting: (AlignmentPhase::Levelling, names::INERTIAL_INCREMENTS),
        }
    }

    /// Forgets the window.
    pub(crate) fn restart(&mut self) {
        self.window = Window::default();
        self.waiting = (AlignmentPhase::Levelling, names::INERTIAL_INCREMENTS);
    }

    /// What the alignment waits for.
    pub(crate) const fn waiting(&self) -> (AlignmentPhase, &'static str) {
        self.waiting
    }

    /// Takes one inertial increment that ends at `at`. At the end of a
    /// rest window it gives the state of that window, if the window has
    /// one.
    pub(crate) fn increment(
        &mut self,
        at: MonotonicNanos,
        dt_s: f64,
        delta_theta_rad: [f64; 3],
        delta_velocity_m_s: [f64; 3],
    ) -> Option<Solved> {
        if !(dt_s.is_finite() && dt_s > 0.0) {
            return None;
        }
        self.window.seconds += dt_s;
        self.window
            .force
            .add(Vector3::from(delta_velocity_m_s) / dt_s);
        self.window.rate.add(Vector3::from(delta_theta_rad) / dt_s);
        if self.window.seconds < self.config.window_s {
            if self.waiting.1 == names::INERTIAL_INCREMENTS {
                self.waiting = (AlignmentPhase::Levelling, names::REST_WINDOW);
            }
            return None;
        }
        let solved = self.solve(at);
        self.window = Window::default();
        match solved {
            Ok(solved) => Some(solved),
            Err(waiting) => {
                self.waiting = waiting;
                None
            }
        }
    }

    /// Takes a GNSS fix. A fix with a velocity says if the vehicle is at
    /// rest; a fix with a covariance gives the position.
    pub(crate) fn gnss(
        &mut self,
        position: GeodeticPosition,
        velocity: Option<NedVelocity>,
        covariance: Option<SymmetricCov3>,
    ) {
        if let Some(v) = velocity {
            let speed = Vector3::new(v.north_mps, v.east_mps, v.down_mps).norm();
            if speed <= self.config.rest_speed_m_s {
                self.window.rest_evidence = true;
            } else {
                self.window.moved = true;
            }
        }
        if let Some(covariance) = covariance {
            self.window.position = Some((position, covariance, PositionSource::Gnss));
        }
    }

    /// Takes the field of the magnetometer in the body axes.
    pub(crate) fn magnetic(&mut self, field_body_t: [f64; 3]) {
        let field = Vector3::from(field_body_t);
        if field.iter().all(|v| v.is_finite()) {
            self.window.field.add(field);
        }
    }

    /// Takes the statement that the vehicle is at rest on a surveyed point.
    pub(crate) fn surveyed(&mut self, position: GeodeticPosition, sigmas_m: (f64, f64)) {
        let (h, v) = sigmas_m;
        if h.is_finite() && v.is_finite() && h > 0.0 && v > 0.0 {
            self.window.rest_evidence = true;
            let covariance = SymmetricCov3::from_diagonal(h * h, h * h, v * v);
            self.window.position = Some((position, covariance, PositionSource::SurveyedPoint));
        }
    }

    /// The state of the complete window, or what it lacks.
    fn solve(&self, at: MonotonicNanos) -> Result<Solved, (AlignmentPhase, &'static str)> {
        let (w, c) = (&self.window, &self.config);
        let force = w.force.mean();
        let gravity = force.norm();
        let at_rest = !w.moved
            && w.force.largest_sigma() <= c.rest_specific_force_sigma_mps2
            && w.rate.mean().norm() <= c.rest_rate_rad_s
            && (gravity - GRAVITY_MPS2).abs() <= REST_GRAVITY_BAND_MPS2;
        if !at_rest {
            return Err((AlignmentPhase::Levelling, names::REST));
        }
        if !w.rest_evidence {
            return Err((AlignmentPhase::Levelling, names::REST_EVIDENCE));
        }
        // At rest the specific force is the reaction to gravity: up.
        let roll = (-force.y).atan2(-force.z);
        let pitch = force.x.atan2(force.y.hypot(force.z));
        // The noise averages over the window. The bias does not: a
        // levelled tilt holds the whole accelerometer bias.
        let (noise, bias) = self.accelerometer;
        let tilt_sigma_rad = (noise * noise / w.seconds + bias * bias).sqrt() / gravity;
        let (heading, heading_sigma_rad) = self.heading(roll, pitch, tilt_sigma_rad)?;
        let Some((position, position_cov, source)) = w.position else {
            return Err((AlignmentPhase::Position, names::POSITION_SOURCE));
        };
        let attitude = UnitQuaternion::from_euler_angles(roll, pitch, heading);
        let (sin_lat, cos_lat) = position.latitude_rad.sin_cos();
        let earth_rate_ned = Vector3::new(cos_lat, 0.0, -sin_lat) * EARTH_RATE_RAD_S;
        let gyro_bias = w.rate.mean() - attitude.inverse() * earth_rate_ned;
        let q = attitude.quaternion();
        let velocity_var = c.rest_velocity_sigma_m_s.powi(2);
        let tilt_var = tilt_sigma_rad * tilt_sigma_rad;
        Ok(Solved {
            alignment: Alignment {
                position,
                velocity: NedVelocity::new(0.0, 0.0, 0.0),
                attitude: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
                position_cov,
                velocity_cov: SymmetricCov3::from_diagonal(
                    velocity_var,
                    velocity_var,
                    velocity_var,
                ),
                attitude_cov: SymmetricCov3::from_diagonal(
                    tilt_var,
                    tilt_var,
                    heading_sigma_rad * heading_sigma_rad,
                ),
                at,
            },
            record: AlignmentRecord {
                at,
                source: AlignmentSource::Sensors,
                heading: Some(HeadingSource::Magnetometer),
                position: Some(source),
                tilt_sigma_rad,
                heading_sigma_rad,
                gyro_bias_rad_s: Some(gyro_bias.into()),
            },
        })
    }

    /// The heading of the window, clockwise from true north, and its
    /// one-sigma, from the mean field of the magnetometer.
    fn heading(
        &self,
        roll: f64,
        pitch: f64,
        tilt_sigma_rad: f64,
    ) -> Result<(f64, f64), (AlignmentPhase, &'static str)> {
        let no_source = (AlignmentPhase::Heading, names::HEADING_SOURCE);
        let magnetometer = self.magnetometer.ok_or(no_source)?;
        let samples = self.window.field.count();
        // The field in the axes of a level body with the same heading.
        let level = UnitQuaternion::from_euler_angles(roll, pitch, 0.0) * self.window.field.mean();
        let horizontal = level.x.hypot(level.y);
        if samples == 0 || !(horizontal.is_finite() && horizontal > 0.0) {
            return Err(no_source);
        }
        let heading = magnetometer.declination_rad + (-level.y).atan2(level.x);
        // A tilt error turns the vertical field into the horizontal
        // plane: the heading error is the tilt error times the tangent of
        // the dip. Noise averages over the samples; the disturbance of
        // the body and the declination are biases and do not.
        let dip = level.z / horizontal;
        let variance = magnetometer.noise_sigma_rad.powi(2) / f64::from(samples)
            + magnetometer.disturbance_sigma_rad.powi(2)
            + magnetometer.declination_sigma_rad.powi(2)
            + (tilt_sigma_rad * dip).powi(2);
        let sigma = variance.sqrt();
        if sigma.is_nan() || sigma > self.config.max_heading_sigma_rad {
            return Err((AlignmentPhase::Heading, names::HEADING_UNCERTAIN));
        }
        Ok((heading, sigma))
    }
}

/// What a transfer alignment states: the host gave each part.
pub(crate) fn transfer_record(a: &Alignment) -> AlignmentRecord {
    let (north, east, down) = a.attitude_cov.diagonal();
    AlignmentRecord {
        at: a.at,
        source: AlignmentSource::Transfer,
        heading: None,
        position: None,
        tilt_sigma_rad: ((north + east) / 2.0).max(0.0).sqrt(),
        heading_sigma_rad: down.max(0.0).sqrt(),
        gyro_bias_rad_s: None,
    }
}

/// Whether the alignment takes an input before the first state.
pub(crate) const fn takes(data: &SensorData) -> bool {
    matches!(
        data,
        SensorData::ImuIncrement { .. }
            | SensorData::GnssFix { .. }
            | SensorData::BaroAltitude { .. }
            | SensorData::MagneticField { .. }
            | SensorData::SurveyedPosition { .. }
    )
}

/// The measurement name of an input that the alignment takes.
pub(super) const fn measurement(data: &SensorData) -> &'static str {
    match data {
        SensorData::GnssFix { .. } => "gnss_position",
        SensorData::BaroAltitude { .. } => "baro_altitude",
        SensorData::MagneticField { .. } => names::MAGNETIC_FIELD,
        SensorData::SurveyedPosition { .. } => names::SURVEYED_POSITION,
        _ => "observation",
    }
}

#[cfg(test)]
mod tests;
