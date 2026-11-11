//! Whether a window is one of rest, by tests that come from the declared
//! budgets of the sensors, and the velocity one-sigma that a state of
//! rest declares.
//!
//! Each test refuses at [`GATE_SIGMAS`] of what the budgets say an honest
//! window of rest scatters by. A velocity that the tests let through is
//! not zero; the declared velocity one-sigma covers it: the noise of the
//! evidence, and a velocity spread evenly up to the gate.

use nalgebra::Vector3;

use super::window::Window;
use crate::AlignmentNeed;

/// Multiple of a stated one-sigma at which a test of rest refuses. On one
/// axis an honest window passes a test of this width but about once in
/// fifteen thousand.
pub(crate) const GATE_SIGMAS: f64 = 4.0;
/// Normal gravity half way between that at the equator and at the poles,
/// m/s².
pub(super) const GRAVITY_MPS2: f64 = 9.806_2;
/// Half the span of normal gravity from the equator to the poles and from
/// sea level to 3 km, m/s².
pub(super) const GRAVITY_SPAN_MPS2: f64 = 0.036;
/// Rotation rate of the Earth, rad/s.
pub(crate) const EARTH_RATE_RAD_S: f64 = 7.292_115e-5;

/// The declared errors of the inertial unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Inertial {
    /// Accelerometer white noise density, (m/s²)/√Hz.
    pub accel_noise: f64,
    /// Accelerometer bias one-sigma, m/s².
    pub accel_bias: f64,
    /// Gyroscope white noise density, (rad/s)/√Hz.
    pub gyro_noise: f64,
    /// Gyroscope bias one-sigma, rad/s.
    pub gyro_bias: f64,
}

/// A window of rest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rest {
    /// The velocity one-sigma of the state, each axis, m/s.
    pub velocity_sigma_m_s: f64,
}

/// The velocity one-sigma of a state of rest from evidence with the noise
/// `sigma` on each axis, tested at the gate: the noise, and a velocity
/// spread evenly up to the gate, whose variance is a third of the square
/// of the gate.
fn spread(sigma: f64) -> f64 {
    sigma * (1.0 + GATE_SIGMAS * GATE_SIGMAS / 3.0).sqrt()
}

/// Whether `window` is one of rest by the inertial unit `inertial` and the
/// evidence of the other sensors, or the name of what it lacks.
pub(crate) fn rest(window: &Window, inertial: &Inertial) -> Result<Rest, AlignmentNeed> {
    let samples = window.force.count();
    // The sample variance of n increments scatters by √(2/(n−1)) of the
    // variance; the test needs the gate inside one variance.
    let variance_gate = (2.0 / f64::from(samples.saturating_sub(1).max(1))).sqrt() * GATE_SIGMAS;
    let Some(scatter) = window
        .force
        .largest_variance()
        .filter(|_| variance_gate <= 1.0)
    else {
        return Err(AlignmentNeed::RestSamples);
    };
    let seconds = window.seconds();
    let sample_variance =
        inertial.accel_noise * inertial.accel_noise * f64::from(samples) / seconds;
    let force = window.force.mean();
    let gravity_band = GRAVITY_SPAN_MPS2
        + GATE_SIGMAS * (inertial.accel_bias.powi(2) + sample_variance / f64::from(samples)).sqrt();
    let rate_bound = EARTH_RATE_RAD_S
        + GATE_SIGMAS
            * (3.0 * (inertial.gyro_bias.powi(2) + inertial.gyro_noise.powi(2) / seconds)).sqrt();
    let inertial_rest = scatter <= sample_variance * (1.0 + variance_gate)
        && (force.norm() - GRAVITY_MPS2).abs() <= gravity_band
        && window.rate.mean().norm() <= rate_bound;
    let velocity = gnss_velocity(window);
    let moving = velocity.is_some_and(|(mean, sigma)| mean.amax() > GATE_SIGMAS * sigma);
    if !inertial_rest || moving {
        return Err(AlignmentNeed::Rest);
    }
    if window.surveys.iter().any(|at| window.holds(*at)) {
        // No sensor measures the velocity on the point. The statement says
        // it is zero; a mean specific force at the gate of its noise moves
        // the vehicle by the noise density times the root of the time.
        let noise = inertial.accel_noise * seconds.sqrt();
        return Ok(Rest {
            velocity_sigma_m_s: spread(noise),
        });
    }
    velocity
        .map(|(_, sigma)| Rest {
            velocity_sigma_m_s: spread(sigma),
        })
        .ok_or(AlignmentNeed::RestEvidence)
}

/// The mean of the GNSS velocities of the window and its one-sigma on each
/// axis, or `None` without a velocity.
fn gnss_velocity(window: &Window) -> Option<(Vector3<f64>, f64)> {
    let inside: Vec<_> = window
        .velocities
        .iter()
        .filter(|(at, _)| window.holds(*at))
        .map(|(_, velocity)| velocity)
        .collect();
    let count = inside.len() as f64;
    if inside.is_empty() {
        return None;
    }
    let sum = inside
        .iter()
        .fold(Vector3::zeros(), |sum, v| sum + v.ned_m_s);
    let variance: f64 = inside.iter().map(|v| v.sigma_m_s * v.sigma_m_s).sum();
    Some((sum / count, variance.sqrt() / count))
}
