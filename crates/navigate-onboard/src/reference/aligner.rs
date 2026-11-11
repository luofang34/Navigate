//! Self-alignment of the reference navigator at rest: roll and pitch from
//! the specific force, heading from the magnetometer, and position from
//! GNSS or a surveyed point.
//!
//! The alignment works in rest windows. A window is the inertial
//! increments of one unbroken stretch of time, by their stamps: a gap or a
//! stamp that goes back starts a new window, and an input of another
//! sensor counts only in the window that holds its stamp. A window gives
//! a state when, in that one window, the vehicle is at rest by its
//! inertial data and by the evidence of another sensor, a heading source
//! is good enough, and a position came. Each window levels again, so no
//! part of a state is older than its window. The inertial data alone
//! cannot show rest: steady straight flight has the specific force and the
//! body rate of rest.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{
    AttitudeQuaternion, GeodeticPosition, MonotonicNanos, NedVelocity, SymmetricCov3,
};
use navigate_fusion::ImuBiasPrior;

use crate::{
    Alignment, AlignmentConfig, AlignmentPhase, AlignmentRecord, AlignmentSource, ErrorBudget,
    HeadingSource, PositionSource, SensorData, SensorKind, SensorSuite,
};

pub(crate) mod names;
mod rest;
mod state;
mod window;

use rest::{EARTH_RATE_RAD_S, Inertial};
use window::{Velocity, Window, seconds_between};

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
    /// The IMU biases of the window for the filter.
    pub biases: ImuBiasPrior,
}

/// What the alignment waits for.
type Waiting = (AlignmentPhase, &'static str);

/// The self-alignment of one navigator.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Aligner {
    pub(crate) config: AlignmentConfig,
    inertial: Option<Inertial>,
    magnetometer: Option<Magnetometer>,
    window: Window,
    waiting: Waiting,
}

impl Aligner {
    /// The aligner of the sensors of `suite`.
    pub(crate) fn new(suite: &SensorSuite) -> Self {
        let budget = |kind| suite.of_kind(kind).next().map(|spec| spec.budget);
        let inertial = match budget(SensorKind::Imu) {
            Some(ErrorBudget::Imu {
                accel_noise_mps2_sqrt_hz,
                gyro_noise_rad_s_sqrt_hz,
                accel_bias,
                gyro_bias,
            }) => Some(Inertial {
                accel_noise: accel_noise_mps2_sqrt_hz,
                accel_bias: accel_bias.sigma(),
                gyro_noise: gyro_noise_rad_s_sqrt_hz,
                gyro_bias: gyro_bias.sigma(),
            }),
            _ => None,
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
            inertial,
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
    pub(crate) const fn waiting(&self) -> Waiting {
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
        let begins =
            MonotonicNanos::from_nanos(at.as_nanos().saturating_sub((dt_s * 1e9).round() as u64));
        // An increment continues the window when it starts where the last
        // one ended, to half a step.
        let continues = self
            .window
            .span
            .is_some_and(|(_, end)| at > end && seconds_between(end, begins).abs() <= dt_s / 2.0);
        match self.window.span {
            Some((start, _)) if continues => self.window.span = Some((start, at)),
            _ => {
                self.window = self.window.from(begins);
                self.window.span = Some((begins, at));
            }
        }
        self.window
            .force
            .add(Vector3::from(delta_velocity_m_s) / dt_s);
        self.window.rate.add(Vector3::from(delta_theta_rad) / dt_s);
        if self.window.seconds() < self.config.window_s {
            if self.waiting.1 == names::INERTIAL_INCREMENTS {
                self.waiting = (AlignmentPhase::Levelling, names::REST_WINDOW);
            }
            return None;
        }
        let solved = self.solve(at);
        self.window = self
            .window
            .from(MonotonicNanos::from_nanos(at.as_nanos().saturating_add(1)));
        match solved {
            Ok(solved) => Some(solved),
            Err(waiting) => {
                self.waiting = waiting;
                None
            }
        }
    }

    /// Takes a GNSS fix acquired at `at`. A velocity with a one-sigma says
    /// whether the vehicle is at rest; a covariance gives the position.
    pub(crate) fn gnss(
        &mut self,
        at: MonotonicNanos,
        position: GeodeticPosition,
        velocity: Option<(NedVelocity, f64)>,
        covariance: Option<SymmetricCov3>,
    ) {
        if let Some((v, sigma_m_s)) = velocity
            && sigma_m_s.is_finite()
            && sigma_m_s > 0.0
        {
            let ned_m_s = Vector3::new(v.north_mps, v.east_mps, v.down_mps);
            if ned_m_s.iter().all(|c| c.is_finite()) {
                self.window
                    .velocities
                    .push((at, Velocity { ned_m_s, sigma_m_s }));
            }
        }
        if let Some(covariance) = covariance {
            self.window
                .positions
                .push((at, (position, covariance, PositionSource::Gnss)));
        }
    }

    /// Takes the field of the magnetometer in the body axes, measured at
    /// `at`.
    pub(crate) fn magnetic(&mut self, at: MonotonicNanos, field_body_t: [f64; 3]) {
        let field = Vector3::from(field_body_t);
        if field.iter().all(|v| v.is_finite()) {
            self.window.field.push((at, field));
        }
    }

    /// Takes the statement, made at `at`, that the vehicle is at rest on a
    /// surveyed point.
    pub(crate) fn surveyed(
        &mut self,
        at: MonotonicNanos,
        position: GeodeticPosition,
        sigmas_m: (f64, f64),
    ) {
        let (h, v) = sigmas_m;
        if h.is_finite() && v.is_finite() && h > 0.0 && v > 0.0 {
            let covariance = SymmetricCov3::from_diagonal(h * h, h * h, v * v);
            self.window.surveys.push(at);
            self.window
                .positions
                .push((at, (position, covariance, PositionSource::SurveyedPoint)));
        }
    }

    /// The state of the complete window, or what it lacks.
    fn solve(&self, at: MonotonicNanos) -> Result<Solved, Waiting> {
        let w = &self.window;
        let inertial = self
            .inertial
            .ok_or((AlignmentPhase::Levelling, names::INERTIAL_INCREMENTS))?;
        let rest = rest::rest(w, &inertial).map_err(|name| (AlignmentPhase::Levelling, name))?;
        let force = w.force.mean();
        let gravity = force.norm();
        // At rest the specific force is the reaction to gravity: up.
        let roll = (-force.y).atan2(-force.z);
        let pitch = force.x.atan2(force.y.hypot(force.z));
        // The noise averages over the window. The bias does not: a
        // levelled tilt holds the whole accelerometer bias.
        let tilt_sigma_rad =
            (inertial.accel_noise.powi(2) / w.seconds() + inertial.accel_bias.powi(2)).sqrt()
                / gravity;
        let (heading, heading_sigma_rad) = self.heading(roll, pitch, tilt_sigma_rad)?;
        let Some((position, position_cov, source)) = w
            .positions
            .iter()
            .filter(|(stamp, _)| w.holds(*stamp))
            .max_by_key(|(stamp, _)| *stamp)
            .map(|(_, position)| *position)
        else {
            return Err((AlignmentPhase::Position, names::POSITION_SOURCE));
        };
        let attitude = UnitQuaternion::from_euler_angles(roll, pitch, heading);
        let (sin_lat, cos_lat) = position.latitude_rad.sin_cos();
        let earth_rate_ned = Vector3::new(cos_lat, 0.0, -sin_lat) * EARTH_RATE_RAD_S;
        let gyro_bias = w.rate.mean() - attitude.inverse() * earth_rate_ned;
        // The mean rate holds the gyroscope noise over the window; the
        // Earth rate taken out of it is turned by the attitude error.
        let gyro_bias_var = inertial.gyro_noise.powi(2) / w.seconds()
            + (EARTH_RATE_RAD_S * tilt_sigma_rad.hypot(heading_sigma_rad)).powi(2);
        // A levelled tilt holds the accelerometer bias, so the window does
        // not measure it: zero with its declared variance.
        let accel_bias_var = inertial.accel_bias.powi(2);
        let biases = ImuBiasPrior::new(
            [0.0; 3],
            SymmetricCov3::from_diagonal(accel_bias_var, accel_bias_var, accel_bias_var),
            gyro_bias.into(),
            SymmetricCov3::from_diagonal(gyro_bias_var, gyro_bias_var, gyro_bias_var),
        );
        let q = attitude.quaternion();
        let velocity_var = rest.velocity_sigma_m_s.powi(2);
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
                gyro_bias_sigma_rad_s: Some(gyro_bias_var.sqrt()),
            },
            biases,
        })
    }

    /// The heading of the window, clockwise from true north, and its
    /// one-sigma, from the mean field of the magnetometer.
    fn heading(&self, roll: f64, pitch: f64, tilt_sigma_rad: f64) -> Result<(f64, f64), Waiting> {
        let no_source = (AlignmentPhase::Heading, names::HEADING_SOURCE);
        let magnetometer = self.magnetometer.ok_or(no_source)?;
        let w = &self.window;
        let (samples, sum) = w
            .field
            .iter()
            .filter(|(at, _)| w.holds(*at))
            .fold((0_u32, Vector3::zeros()), |(n, sum), (_, field)| {
                (n.saturating_add(1), sum + field)
            });
        if samples == 0 {
            return Err(no_source);
        }
        // The field in the axes of a level body with the same heading.
        let level =
            UnitQuaternion::from_euler_angles(roll, pitch, 0.0) * (sum / f64::from(samples));
        let horizontal = level.x.hypot(level.y);
        if !(horizontal.is_finite() && horizontal > 0.0) {
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
        gyro_bias_sigma_rad_s: None,
    }
}

/// The measurement name of an input that the alignment records.
pub(super) const fn measurement(data: &SensorData) -> &'static str {
    match data {
        SensorData::GnssFix { .. } => "gnss_position",
        SensorData::BaroAltitude { .. } => "baro_altitude",
        SensorData::MagneticField { .. } => names::MAGNETIC_FIELD,
        SensorData::SurveyedPosition { .. } => names::SURVEYED_POSITION,
        SensorData::StarCentroids { .. } => "star_line_of_sight",
        #[cfg(feature = "terrain-camera")]
        SensorData::TerrainFrame { .. } => super::terrain::MEASUREMENT,
        _ => "observation",
    }
}

#[cfg(test)]
mod tests;
