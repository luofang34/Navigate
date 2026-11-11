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
    Alignment, AlignmentConfig, AlignmentNeed, AlignmentPhase, AlignmentRecord, AlignmentSource,
    DeclaredStart, ErrorBudget, HeadingSource, MagneticReference, PositionSource, SensorData,
    SensorKind, SensorSuite,
};

mod declared;
mod heading;
mod motion;
pub(crate) mod names;
mod rest;
mod state;
mod window;

use heading::Magnetometer;
pub(crate) use motion::{Increment, Moving};
use motion::{Motion, VelocityFix};
use rest::{EARTH_RATE_RAD_S, Inertial};
use window::{Velocity, Window, seconds_between};

/// A state that one rest window gave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Solved {
    pub alignment: Alignment,
    pub record: AlignmentRecord,
    /// The IMU biases that the alignment measured, for the filter.
    pub biases: Option<ImuBiasPrior>,
}

/// What the alignment waits for.
type Waiting = (AlignmentPhase, AlignmentNeed);

/// The self-alignment of one navigator.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Aligner {
    pub(crate) config: AlignmentConfig,
    inertial: Option<Inertial>,
    magnetometer: Option<Magnetometer>,
    /// The field that the host states for the place.
    pub(crate) reference: Option<MagneticReference>,
    window: Window,
    motion: Motion,
    /// What the alignment in motion lacked at the last fix of a moving
    /// vehicle, or `None` when the last fix was of a vehicle near rest.
    motion_need: Option<Waiting>,
    /// A declared start whose stamp the increments have not passed.
    pending: Option<(MonotonicNanos, DeclaredStart)>,
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
                noise_sigma_t,
                disturbance_sigma_t,
            }) => Some(Magnetometer {
                noise_sigma_t,
                disturbance_sigma_t,
            }),
            _ => None,
        };
        Self {
            config: AlignmentConfig::default(),
            inertial,
            magnetometer,
            reference: None,
            window: Window::default(),
            motion: Motion::default(),
            motion_need: None,
            pending: None,
            waiting: (AlignmentPhase::Levelling, AlignmentNeed::InertialIncrements),
        }
    }

    /// Forgets the window.
    pub(crate) fn restart(&mut self) {
        self.window = Window::default();
        self.motion = Motion::default();
        self.motion_need = None;
        self.pending = None;
        self.waiting = (AlignmentPhase::Levelling, AlignmentNeed::InertialIncrements);
    }

    /// What the alignment waits for.
    pub(crate) const fn waiting(&self) -> Waiting {
        self.waiting
    }

    /// Takes one inertial increment. It gives the state of a declared start
    /// whose stamp it passes, or of a rest window that it completes.
    pub(crate) fn increment(&mut self, step: Increment) -> Option<Moving> {
        self.motion.increment(step);
        if let Some(moving) = self.try_declared() {
            return Some(moving);
        }
        self.rest_increment(step).map(|solved| Moving {
            solved,
            later: Vec::new(),
        })
    }

    /// Takes the start that an operator declared at `at`.
    pub(crate) fn declared(&mut self, at: MonotonicNanos, start: DeclaredStart) -> Option<Moving> {
        if !start.is_usable() {
            return None;
        }
        self.pending = Some((at, start));
        self.try_declared()
    }

    /// The state of the pending declared start, once the increments pass
    /// its stamp; what it lacks becomes what the alignment waits for.
    fn try_declared(&mut self) -> Option<Moving> {
        let (at, start) = self.pending?;
        if self.motion.latest().is_none_or(|latest| latest < at) {
            self.waiting = (AlignmentPhase::Levelling, AlignmentNeed::InertialIncrements);
            return None;
        }
        self.pending = None;
        let Some(inertial) = self.inertial else {
            self.waiting = (AlignmentPhase::Levelling, AlignmentNeed::InertialIncrements);
            return None;
        };
        match declared::solve(&start, at, &self.motion, &inertial, &self.config) {
            Ok(moving) => Some(moving),
            Err(need) => {
                let phase = match need {
                    AlignmentNeed::HeightDatum => AlignmentPhase::Position,
                    _ => AlignmentPhase::Levelling,
                };
                self.waiting = (phase, need);
                None
            }
        }
    }

    /// Takes one inertial increment into the rest window. At the end of the
    /// window it gives the state of that window, if the window has one.
    fn rest_increment(&mut self, step: Increment) -> Option<Solved> {
        let Increment {
            acquired_at: at,
            dt_s,
            delta_theta_rad,
            delta_velocity_m_s,
            ..
        } = step;
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
            if self.waiting.1 == AlignmentNeed::InertialIncrements {
                self.waiting = (AlignmentPhase::Levelling, AlignmentNeed::RestWindow);
            }
            return None;
        }
        let solved = self.solve(at);
        self.window = self
            .window
            .from(MonotonicNanos::from_nanos(at.as_nanos().saturating_add(1)));
        match solved {
            Ok(solved) => Some(solved),
            // A vehicle that moves is not at rest; what the alignment in
            // motion lacks says more.
            Err((_, AlignmentNeed::Rest)) if self.motion_need.is_some() => {
                self.waiting = self.motion_need.unwrap_or(self.waiting);
                None
            }
            Err(waiting) => {
                self.waiting = waiting;
                None
            }
        }
    }

    /// Takes a GNSS fix acquired at `at`. A velocity with a one-sigma says
    /// whether the vehicle is at rest; a covariance gives the position.
    /// For a vehicle whose course is a heading, a fix in motion gives the
    /// state at that fix when an earlier fix and the inertial increments
    /// between the two permit it.
    pub(crate) fn gnss(
        &mut self,
        at: MonotonicNanos,
        position: GeodeticPosition,
        velocity: Option<(NedVelocity, Option<f64>)>,
        covariance: Option<SymmetricCov3>,
    ) -> Option<Moving> {
        if let Some((v, Some(sigma_m_s))) = velocity
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
        let crab_sigma = self.config.course_heading_sigma_rad?;
        let outcome = self.moving(at, position, velocity, covariance, crab_sigma);
        match outcome {
            Ok(moving) => Some(moving),
            Err(need) => {
                self.motion_need = need;
                if let Some(need) = need {
                    self.waiting = need;
                }
                None
            }
        }
    }

    /// The state of a moving vehicle at a fix; or what the alignment in
    /// motion lacks, `None` for a vehicle near rest, which the rest window
    /// judges.
    fn moving(
        &mut self,
        at: MonotonicNanos,
        position: GeodeticPosition,
        velocity: Option<(NedVelocity, Option<f64>)>,
        covariance: Option<SymmetricCov3>,
        crab_sigma: f64,
    ) -> Result<Moving, Option<Waiting>> {
        let Some((velocity, sigma)) = velocity else {
            return Err(None);
        };
        let speed = Vector3::new(velocity.north_mps, velocity.east_mps, velocity.down_mps).norm();
        let sigma = sigma.filter(|s| s.is_finite() && *s > 0.0);
        // Near rest within the gate of the receiver noise, the rest window
        // judges the vehicle.
        if sigma.is_some_and(|s| speed <= rest::GATE_SIGMAS * s) {
            return Err(None);
        }
        let Some(position_cov) = covariance else {
            return Err(Some((
                AlignmentPhase::Position,
                AlignmentNeed::PositionSource,
            )));
        };
        let Some(sigma) = sigma else {
            return Err(Some((
                AlignmentPhase::Levelling,
                AlignmentNeed::VelocityAccuracy,
            )));
        };
        let inertial = self.inertial.ok_or(Some((
            AlignmentPhase::Levelling,
            AlignmentNeed::InertialIncrements,
        )))?;
        let fix = VelocityFix {
            at,
            position,
            position_cov,
            velocity,
            velocity_sigma_m_s: sigma,
        };
        self.motion
            .fix(
                &fix,
                crab_sigma,
                self.config.max_heading_sigma_rad,
                &inertial,
            )
            .map_err(Some)
    }

    /// Takes the field of the magnetometer in the body axes, measured at
    /// `at`.
    pub(crate) fn magnetic(&mut self, at: MonotonicNanos, field_body_t: [f64; 3]) {
        let field = Vector3::from(field_body_t);
        if field.iter().all(|v| v.is_finite()) {
            self.window.field.push((at, field));
        }
    }

    /// Takes the statement that the vehicle was at `position` at `at`. It
    /// says nothing of rest.
    pub(crate) fn known(
        &mut self,
        at: MonotonicNanos,
        position: GeodeticPosition,
        sigmas_m: (f64, f64),
    ) {
        let (h, v) = sigmas_m;
        if h.is_finite() && v.is_finite() && h > 0.0 && v > 0.0 {
            let covariance = SymmetricCov3::from_diagonal(h * h, h * h, v * v);
            self.window
                .positions
                .push((at, (position, covariance, PositionSource::KnownPosition)));
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
            .ok_or((AlignmentPhase::Levelling, AlignmentNeed::InertialIncrements))?;
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
        let (heading, heading_sigma_rad) = heading::heading(
            &heading::Inputs {
                magnetometer: self.magnetometer,
                reference: self.reference,
                window: w,
                max_sigma_rad: self.config.max_heading_sigma_rad,
            },
            roll,
            pitch,
            tilt_sigma_rad,
        )
        .map_err(|need| (AlignmentPhase::Heading, need))?;
        let Some((position, position_cov, source)) = w
            .positions
            .iter()
            .filter(|(stamp, _)| w.holds(*stamp))
            .max_by_key(|(stamp, _)| *stamp)
            .map(|(_, position)| *position)
        else {
            return Err((AlignmentPhase::Position, AlignmentNeed::PositionSource));
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
            biases: Some(biases),
        })
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
        SensorData::KnownPosition { .. } => names::KNOWN_POSITION,
        SensorData::DeclaredStart(_) => names::DECLARED_START,
        SensorData::StarCentroids { .. } => "star_line_of_sight",
        #[cfg(feature = "terrain-camera")]
        SensorData::TerrainFrame { .. } => super::terrain::MEASUREMENT,
        _ => "observation",
    }
}

#[cfg(test)]
mod tests;
