//! Alignment of a moving vehicle from two GNSS fixes with a velocity.
//!
//! The inertial data alone cannot level a moving vehicle: a takeoff roll
//! at 2.5 m/s² tilts the specific force by 14°, and a coordinated turn
//! keeps it along the body vertical. The change of the GNSS velocity
//! between two fixes is the mean acceleration of that span, so the mean
//! specific force of the span is known in the north, east, down axes:
//! the acceleration less gravity. The body turns in the span, so the
//! velocity increments are summed in the body axes of one instant with
//! the rotation increments. The two vectors give two of the three angles.
//! The third is the heading, which a vehicle that moves along its nose
//! has from its course over the ground, to the configured allowance for
//! its crab angle.
//!
//! Of the earlier fixes, the span is the one whose tilt one-sigma is the
//! smallest by the declared budgets: a short span makes the velocity
//! noise a large acceleration, and a long one lets the gyroscope bias turn
//! the sum.
//!
//! Each part of the state is for the instant of the second fix. A fix
//! can come late, so the alignment keeps the last increments: those
//! after the fix go into the filter after the state is set. A fix is not
//! at the end of an increment, so an increment that a fix divides counts
//! in the span by its share of the time, with its mean rates. The filter
//! does the same with an increment that holds the instant of its state.

use std::collections::VecDeque;

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{
    AttitudeQuaternion, GeodeticPosition, MonotonicNanos, NedVelocity, SourceId, SymmetricCov3,
};

use super::rest::{EARTH_RATE_RAD_S, GATE_SIGMAS, Inertial};
use super::window::seconds_between;
use super::{Solved, Waiting};
use crate::{
    Alignment, AlignmentNeed, AlignmentPhase, AlignmentRecord, AlignmentSource, HeadingSource,
    PositionSource,
};

/// Normal gravity, m/s², for the specific force of a known acceleration.
/// The solve scales the two vectors to one size, so the change of gravity
/// over the Earth does not tilt the result.
const GRAVITY_MPS2: f64 = 9.806_65;
/// Time of inertial increments and fixes that the alignment keeps,
/// seconds. It bounds the memory of the alignment and the longest span;
/// with the budgets of a tactical unit the best span is a few seconds.
const KEPT_S: f64 = 10.0;

/// One inertial increment as the navigator took it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Increment {
    pub sensor: SourceId,
    pub acquired_at: MonotonicNanos,
    pub delivered_at: MonotonicNanos,
    pub dt_s: f64,
    pub delta_theta_rad: [f64; 3],
    pub delta_velocity_m_s: [f64; 3],
}

impl Increment {
    /// The time of the increment that is after `instant`, seconds.
    fn seconds_after(&self, instant: MonotonicNanos) -> f64 {
        seconds_between(instant, self.acquired_at).clamp(0.0, self.dt_s)
    }
}

/// One GNSS fix with a velocity and its one-sigma.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VelocityFix {
    pub at: MonotonicNanos,
    pub position: GeodeticPosition,
    pub position_cov: SymmetricCov3,
    pub velocity: NedVelocity,
    /// One-sigma of the velocity, each axis, m/s.
    pub velocity_sigma_m_s: f64,
}

/// The state of a moving vehicle at a fix, and the increments after the
/// fix, which the state does not hold.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Moving {
    pub solved: Solved,
    pub later: Vec<Increment>,
}

/// The last inertial increments and GNSS velocities of a vehicle that
/// has no state.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Motion {
    increments: VecDeque<Increment>,
    /// Time and velocity of each fix of the last [`KEPT_S`].
    fixes: VecDeque<(MonotonicNanos, Vector3<f64>)>,
}

impl Motion {
    /// Keeps one inertial increment.
    pub(crate) fn increment(&mut self, increment: Increment) {
        let newest = increment.acquired_at;
        self.increments.push_back(increment);
        while self.increments.front().is_some_and(|old| {
            let age = seconds_between(old.acquired_at, newest);
            age < 0.0 || age > KEPT_S
        }) {
            self.increments.pop_front();
        }
    }

    /// The mean specific force between two instants, in the body axes at
    /// the second, m/s². `None` when the increments do not cover the span
    /// to one step.
    fn mean_force(&self, from: MonotonicNanos, to: MonotonicNanos) -> Option<Vector3<f64>> {
        let span_s = seconds_between(from, to);
        // The rotation from the body axes at an instant into the body axes
        // at `from`, and the sum of the velocity increments in those axes.
        let mut rotation = UnitQuaternion::identity();
        let mut sum = Vector3::zeros();
        let mut covered_s = 0.0;
        let mut longest_step_s: f64 = 0.0;
        for step in &self.increments {
            let inside_s = step.seconds_after(from) - step.seconds_after(to);
            if inside_s <= 0.0 {
                continue;
            }
            // A velocity increment is in the body axes of the middle of
            // its step.
            let part = inside_s / step.dt_s;
            let turn = Vector3::from(step.delta_theta_rad) * part;
            let half = UnitQuaternion::from_scaled_axis(turn * 0.5);
            sum += rotation * half * Vector3::from(step.delta_velocity_m_s) * part;
            rotation *= UnitQuaternion::from_scaled_axis(turn);
            covered_s += inside_s;
            longest_step_s = longest_step_s.max(step.dt_s);
        }
        (covered_s > 0.0 && (covered_s - span_s).abs() <= longest_step_s)
            .then(|| rotation.inverse() * sum / covered_s)
    }

    /// Takes a fix of a vehicle whose course is its heading to
    /// `crab_sigma` radians. With an earlier fix it gives the state at
    /// this fix; if not, what the alignment lacks. Of the earlier fixes,
    /// the spans are tried from the smallest tilt one-sigma up: a span
    /// that the increments do not cover or whose two specific forces do
    /// not agree gives no state, and the next span may.
    pub(crate) fn fix(
        &mut self,
        fix: &VelocityFix,
        crab_sigma: f64,
        max_heading_sigma_rad: f64,
        inertial: &Inertial,
    ) -> Result<Moving, Waiting> {
        let v = &fix.velocity;
        let velocity = Vector3::new(v.north_mps, v.east_mps, v.down_mps);
        while self.fixes.front().is_some_and(|(at, _)| {
            let age = seconds_between(*at, fix.at);
            age < 0.0 || age > KEPT_S
        }) {
            self.fixes.pop_front();
        }
        let mut earlier: Vec<_> = self
            .fixes
            .iter()
            .filter(|(at, _)| seconds_between(*at, fix.at) > 0.0)
            .copied()
            .collect();
        self.fixes.push_back((fix.at, velocity));
        let speed = velocity.xy().norm();
        let course_sigma = fix.velocity_sigma_m_s / speed;
        if course_sigma.is_nan() || course_sigma > max_heading_sigma_rad {
            return Err((AlignmentPhase::Heading, AlignmentNeed::CourseSpeed));
        }
        let heading_sigma_rad = crab_sigma.hypot(course_sigma);
        if heading_sigma_rad.is_nan() || heading_sigma_rad > max_heading_sigma_rad {
            return Err((AlignmentPhase::Heading, AlignmentNeed::HeadingUncertain));
        }
        let span_sigma = |at: MonotonicNanos| {
            tilt_sigma(
                inertial,
                fix.velocity_sigma_m_s,
                seconds_between(at, fix.at),
            )
        };
        earlier.sort_by(|(a, _), (b, _)| span_sigma(*a).total_cmp(&span_sigma(*b)));
        let mut first_need = None;
        for (earlier_at, earlier_velocity) in earlier {
            let span = Span {
                from: earlier_at,
                velocity_from: earlier_velocity,
                tilt_sigma_rad: span_sigma(earlier_at),
            };
            match self.solve_span(fix, &span, heading_sigma_rad) {
                Ok(moving) => return Ok(moving),
                Err(need) => {
                    first_need.get_or_insert(need);
                }
            }
        }
        Err((
            AlignmentPhase::Levelling,
            first_need.unwrap_or(AlignmentNeed::VelocityFixes),
        ))
    }

    /// The state at `fix` from the span that starts at an earlier fix, or
    /// why the span gives none.
    fn solve_span(
        &self,
        fix: &VelocityFix,
        span: &Span,
        heading_sigma_rad: f64,
    ) -> Result<Moving, AlignmentNeed> {
        let v = &fix.velocity;
        let velocity = Vector3::new(v.north_mps, v.east_mps, v.down_mps);
        let span_s = seconds_between(span.from, fix.at);
        let force_body = self
            .mean_force(span.from, fix.at)
            .ok_or(AlignmentNeed::InertialGap)?;
        let acceleration = (velocity - span.velocity_from) / span_s;
        // The Coriolis acceleration of the mean velocity: 0.5 mrad of tilt
        // at 55 m/s, and more for a faster vehicle.
        let (sin_lat, cos_lat) = fix.position.latitude_rad.sin_cos();
        let earth_rate = Vector3::new(cos_lat, 0.0, -sin_lat) * EARTH_RATE_RAD_S;
        let coriolis = earth_rate.cross(&(velocity + span.velocity_from));
        let force_ned = acceleration + coriolis - Vector3::new(0.0, 0.0, GRAVITY_MPS2);
        // The sizes of the two specific forces differ by the accelerometer
        // errors and the velocity noise; a rotation does not change them.
        let force_sigma = span.tilt_sigma_rad * GRAVITY_MPS2;
        if (force_body.norm() - force_ned.norm()).abs() > GATE_SIGMAS * force_sigma {
            return Err(AlignmentNeed::ForceDisagreement);
        }
        let course = velocity.y.atan2(velocity.x);
        // The specific force in the axes of a level body with the heading
        // of the course.
        let level = UnitQuaternion::from_euler_angles(0.0, 0.0, -course) * force_ned;
        let (roll, pitch) = tilt(level, force_body * (level.norm() / force_body.norm()));
        // A heading error turns the horizontal acceleration: that part of
        // the tilt error goes with the heading error. The state gives it
        // as a part of the tilt variance.
        let coupled = heading_sigma_rad * acceleration.xy().norm() / GRAVITY_MPS2;
        let state = MovingState {
            attitude: UnitQuaternion::from_euler_angles(roll, pitch, course),
            tilt_sigma_rad: span.tilt_sigma_rad.hypot(coupled),
            heading_sigma_rad,
        };
        Ok(Moving {
            solved: state.solved(fix),
            later: self
                .increments
                .iter()
                .filter(|step| step.acquired_at > fix.at)
                .copied()
                .collect(),
        })
    }
}

/// A span from an earlier fix to the fix of the state.
struct Span {
    from: MonotonicNanos,
    velocity_from: Vector3<f64>,
    tilt_sigma_rad: f64,
}

/// The tilt one-sigma of a span of `span_s` seconds, radians: the
/// accelerometer noise of the mean and its bias, the noise of the two GNSS
/// velocities of one-sigma `velocity_sigma` as an acceleration, and the
/// turn of the summed force by the gyroscope bias and noise, half the
/// span on the mean.
fn tilt_sigma(inertial: &Inertial, velocity_sigma: f64, span_s: f64) -> f64 {
    let acceleration_var = 2.0 * (velocity_sigma / span_s).powi(2);
    let accelerometer_var = inertial.accel_noise.powi(2) / span_s + inertial.accel_bias.powi(2);
    let turn_var =
        (inertial.gyro_bias * span_s / 2.0).powi(2) + inertial.gyro_noise.powi(2) * span_s / 3.0;
    ((acceleration_var + accelerometer_var) / GRAVITY_MPS2.powi(2) + turn_var).sqrt()
}

/// The attitude that a span gave and how well it is known.
struct MovingState {
    attitude: UnitQuaternion<f64>,
    tilt_sigma_rad: f64,
    heading_sigma_rad: f64,
}

impl MovingState {
    /// The state at `fix` and what the alignment states about it.
    fn solved(&self, fix: &VelocityFix) -> Solved {
        let q = self.attitude.quaternion();
        let velocity_var = fix.velocity_sigma_m_s * fix.velocity_sigma_m_s;
        let tilt_var = self.tilt_sigma_rad * self.tilt_sigma_rad;
        let heading_var = self.heading_sigma_rad * self.heading_sigma_rad;
        Solved {
            alignment: Alignment {
                position: fix.position,
                velocity: fix.velocity,
                attitude: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
                position_cov: fix.position_cov,
                velocity_cov: SymmetricCov3::from_diagonal(
                    velocity_var,
                    velocity_var,
                    velocity_var,
                ),
                attitude_cov: SymmetricCov3::from_diagonal(tilt_var, tilt_var, heading_var),
                at: fix.at,
            },
            record: AlignmentRecord {
                at: fix.at,
                source: AlignmentSource::Sensors,
                heading: Some(HeadingSource::GnssCourse),
                position: Some(PositionSource::Gnss),
                tilt_sigma_rad: self.tilt_sigma_rad,
                heading_sigma_rad: self.heading_sigma_rad,
                gyro_bias_rad_s: None,
                gyro_bias_sigma_rad_s: None,
            },
            biases: None,
        }
    }
}

/// Roll and pitch of a body whose specific force is `body` in its own
/// axes and `level` in the axes of a level body with the same heading.
/// The two have one size. Of the two attitudes that agree, this is the
/// one that is not on its back.
pub(super) fn tilt(level: Vector3<f64>, body: Vector3<f64>) -> (f64, f64) {
    // Forward: body.x = level.x·cos(pitch) − level.z·sin(pitch).
    let along = level.x.hypot(level.z);
    let pitch = (-level.z).atan2(level.x) - (body.x / along).clamp(-1.0, 1.0).acos();
    let (sin, cos) = pitch.sin_cos();
    // The downward part of the specific force after the pitch, before
    // the roll.
    let down = level.x * sin + level.z * cos;
    let roll = down.atan2(level.y) - body.z.atan2(body.y);
    let wrapped =
        (roll + core::f64::consts::PI).rem_euclid(core::f64::consts::TAU) - core::f64::consts::PI;
    (wrapped, pitch)
}
