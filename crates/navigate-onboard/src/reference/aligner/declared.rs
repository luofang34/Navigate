//! The state of a declared start: the position, height, heading and speed
//! that an operator states, and the roll and pitch that the navigator
//! levels from its own inertial data before the stamp.
//!
//! Levelling takes the vehicle as not accelerated, and the state says
//! what that costs. The size of the mean specific force tests the
//! acceleration along gravity well, and an acceleration across it hardly:
//! an acceleration `a` across gravity changes the size by `a²/2g` only.
//! The tilt one-sigma therefore holds every horizontal acceleration that
//! the size test cannot see, spread evenly up to the largest one. A turn
//! is seen: its body rate and the declared speed give the centripetal
//! acceleration, which a vehicle that moves along its nose takes out of
//! the specific force.
//!
//! The statement does not say the wind, the vertical speed, or the crab
//! angle; the configuration states each as an allowance.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::SymmetricCov3;
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, MonotonicNanos, NedVelocity};
use navigate_data::{UNKNOWN_UNDULATION_BOUND_M, VerticalDatum};

use super::Solved;
use super::motion::{Motion, Moving};
use super::rest::{GATE_SIGMAS, GRAVITY_MPS2, GRAVITY_SPAN_MPS2, Inertial};
use crate::{
    Alignment, AlignmentConfig, AlignmentNeed, AlignmentRecord, AlignmentSource, DeclaredStart,
    HeadingSource, PositionSource, SpeedKind,
};

/// The state of `start` at `at` from the increments of `motion` in the
/// window before it; or what the alignment lacks.
pub(crate) fn solve(
    start: &DeclaredStart,
    at: MonotonicNanos,
    motion: &Motion,
    inertial: &Inertial,
    config: &AlignmentConfig,
) -> Result<Moving, AlignmentNeed> {
    let undulation_m = match start.height.datum {
        VerticalDatum::Ellipsoidal => 0.0,
        VerticalDatum::Orthometric { .. } => UNKNOWN_UNDULATION_BOUND_M,
        _ => return Err(AlignmentNeed::HeightDatum),
    };
    let window_ns = (config.window_s * 1e9).round() as u64;
    let from = MonotonicNanos::from_nanos(at.as_nanos().saturating_sub(window_ns));
    let (Some(force), Some(rate)) = (motion.mean_force(from, at), motion.mean_rate(from, at))
    else {
        return Err(AlignmentNeed::InertialGap);
    };
    let speed = start.speed;
    let wind_var = match speed.kind {
        SpeedKind::Airspeed => config.declared_wind_sigma_m_s.powi(2),
        SpeedKind::GroundSpeed => 0.0,
    };
    let crab = config.course_heading_sigma_rad;
    // The gyroscope bias and noise in the mean rate make a centripetal
    // acceleration that is not there, at the declared speed.
    let rate_sigma =
        (inertial.gyro_bias.powi(2) + inertial.gyro_noise.powi(2) / config.window_s).sqrt();
    let false_turn = speed.speed_m_s * rate_sigma;
    // A vehicle that moves along its nose has its velocity along the body
    // forward axis: in a turn the centripetal acceleration is the body
    // rate across that velocity, and it comes out of the specific force.
    // The rest of it, and all of it for another vehicle, is an error.
    let (centripetal, turn_sigma) = match crab {
        Some(crab) => (
            rate.cross(&Vector3::new(speed.speed_m_s, 0.0, 0.0)),
            rate.norm()
                * (speed.sigma_m_s.powi(2) + (speed.speed_m_s * crab).powi(2) + wind_var).sqrt(),
        ),
        None => (Vector3::zeros(), rate.norm() * speed.speed_m_s),
    };
    let level_force = force - centripetal;
    let window_s = config.window_s;
    let force_var = inertial.accel_bias.powi(2) + inertial.accel_noise.powi(2) / window_s;
    let band = GRAVITY_SPAN_MPS2 + GATE_SIGMAS * force_var.sqrt().hypot(false_turn);
    if (level_force.norm() - GRAVITY_MPS2).abs() > band {
        return Err(AlignmentNeed::SteadyMotion);
    }
    // A size inside the band admits a horizontal acceleration up to
    // √(4 g band); spread evenly, a third of its square is its variance.
    let unseen_var = 4.0 * GRAVITY_MPS2 * band / 3.0;
    let gravity = level_force.norm();
    let roll = (-level_force.y).atan2(-level_force.z);
    let pitch = level_force.x.atan2(level_force.y.hypot(level_force.z));
    let tilt_sigma_rad =
        (force_var + unseen_var + turn_sigma.powi(2) + false_turn.powi(2)).sqrt() / gravity;
    let heading = start.true_heading_rad;
    let attitude = UnitQuaternion::from_euler_angles(roll, pitch, heading);
    let (velocity, velocity_cov) = velocity(start, crab, wind_var, config);
    let h = start.horizontal_sigma_m.powi(2);
    let v = start.height.sigma_m.powi(2) + undulation_m.powi(2);
    let q = attitude.quaternion();
    let tilt_var = tilt_sigma_rad.powi(2);
    let heading_var = start.heading_sigma_rad.powi(2);
    Ok(Moving {
        solved: Solved {
            alignment: Alignment {
                position: GeodeticPosition::new(
                    start.latitude_rad,
                    start.longitude_rad,
                    start.height.height_m,
                ),
                velocity,
                attitude: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
                position_cov: SymmetricCov3::from_diagonal(h, h, v),
                velocity_cov,
                attitude_cov: SymmetricCov3::from_diagonal(tilt_var, tilt_var, heading_var),
                at,
            },
            record: AlignmentRecord {
                at,
                source: AlignmentSource::Declared,
                heading: Some(HeadingSource::Declared),
                position: Some(PositionSource::Declared),
                tilt_sigma_rad,
                heading_sigma_rad: start.heading_sigma_rad,
                gyro_bias_rad_s: None,
                gyro_bias_sigma_rad_s: None,
            },
            biases: None,
        },
        later: motion.after(at),
    })
}

/// The velocity of a declared start and its covariance. A vehicle that
/// moves along its nose moves along its heading, to the heading and crab
/// one-sigmas across it; another moves in an unknown direction. An
/// airspeed adds the wind allowance on each horizontal axis; the vertical
/// speed is zero to its allowance.
fn velocity(
    start: &DeclaredStart,
    crab: Option<f64>,
    wind_var: f64,
    config: &AlignmentConfig,
) -> (NedVelocity, SymmetricCov3) {
    let speed = start.speed;
    let vertical_var = config.declared_vertical_speed_sigma_m_s.powi(2);
    // North, east, and the north-east covariance of the horizontal
    // velocity, with its mean.
    let ([north, east], [nn, ne, ee]) = match crab {
        Some(crab) => {
            let (sin, cos) = start.true_heading_rad.sin_cos();
            let along_var = speed.sigma_m_s.powi(2);
            let across_var =
                speed.speed_m_s.powi(2) * (start.heading_sigma_rad.powi(2) + crab.powi(2));
            (
                [speed.speed_m_s * cos, speed.speed_m_s * sin],
                [
                    cos * cos * along_var + sin * sin * across_var,
                    sin * cos * (along_var - across_var),
                    sin * sin * along_var + cos * cos * across_var,
                ],
            )
        }
        None => {
            let var = speed.speed_m_s.powi(2) / 2.0 + speed.sigma_m_s.powi(2);
            ([0.0, 0.0], [var, 0.0, var])
        }
    };
    (
        NedVelocity::new(north, east, 0.0),
        SymmetricCov3::from_upper_triangle([
            nn + wind_var,
            ne,
            0.0,
            ee + wind_var,
            0.0,
            vertical_var,
        ]),
    )
}
