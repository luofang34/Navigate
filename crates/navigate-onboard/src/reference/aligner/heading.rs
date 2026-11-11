//! The heading of a rest window from the magnetometer and the field that
//! the host states for the place.
//!
//! The declared errors of the magnetometer are fields, in tesla: the noise
//! of one sample and the residual body field after calibration, each axis.
//! A field across the horizontal component of the Earth's field turns the
//! heading by its ratio to that component, so the heading one-sigma comes
//! from the measured horizontal intensity. The strength of the measured
//! field must be that of the stated reference: a field near steel or a
//! running motor has another strength and another direction.

use nalgebra::{UnitQuaternion, Vector3};

use super::rest::GATE_SIGMAS;
use super::window::Window;
use crate::{AlignmentNeed, MagneticReference};

/// The declared errors of the magnetometer, each axis, tesla.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Magnetometer {
    pub noise_sigma_t: f64,
    pub disturbance_sigma_t: f64,
}

/// What the heading of a window needs.
pub(crate) struct Inputs<'a> {
    pub magnetometer: Option<Magnetometer>,
    pub reference: Option<MagneticReference>,
    pub window: &'a Window,
    pub max_sigma_rad: f64,
}

/// The heading of the window, clockwise from true north, and its
/// one-sigma, for a body of `roll` and `pitch` known to `tilt_sigma_rad`;
/// or what the heading lacks.
pub(crate) fn heading(
    inputs: &Inputs<'_>,
    roll: f64,
    pitch: f64,
    tilt_sigma_rad: f64,
) -> Result<(f64, f64), AlignmentNeed> {
    let (Some(magnetometer), Some(reference)) = (inputs.magnetometer, inputs.reference) else {
        return Err(AlignmentNeed::HeadingSource);
    };
    let w = inputs.window;
    let (samples, sum) = w
        .field
        .iter()
        .filter(|(at, _)| w.holds(*at))
        .fold((0_u32, Vector3::zeros()), |(n, sum), (_, field)| {
            (n.saturating_add(1), sum + field)
        });
    if samples == 0 {
        return Err(AlignmentNeed::HeadingSource);
    }
    let mean = sum / f64::from(samples);
    let noise_var = magnetometer.noise_sigma_t.powi(2) / f64::from(samples);
    let body_var = magnetometer.disturbance_sigma_t.powi(2);
    // Along the field, the strength holds the residual body field, the
    // noise of the mean, and the error of the model.
    let strength_sigma = (body_var + noise_var + reference.intensity_sigma_t.powi(2)).sqrt();
    if (mean.norm() - reference.intensity_t).abs() > GATE_SIGMAS * strength_sigma {
        return Err(AlignmentNeed::FieldStrength);
    }
    // The field in the axes of a level body with the same heading.
    let level = UnitQuaternion::from_euler_angles(roll, pitch, 0.0) * mean;
    let horizontal = level.x.hypot(level.y);
    if !(horizontal.is_finite() && horizontal > 0.0) {
        return Err(AlignmentNeed::HeadingSource);
    }
    let heading = reference.declination_rad + (-level.y).atan2(level.x);
    // A field across the horizontal component turns the heading by its
    // ratio to that component. A tilt error turns the vertical field into
    // the horizontal plane: the heading error is the tilt error times the
    // tangent of the dip. Noise averages over the samples; the body field
    // and the declination are biases and do not.
    let dip = level.z / horizontal;
    let variance = (noise_var + body_var) / (horizontal * horizontal)
        + reference.declination_sigma_rad.powi(2)
        + (tilt_sigma_rad * dip).powi(2);
    let sigma = variance.sqrt();
    if sigma.is_nan() || sigma > inputs.max_sigma_rad {
        return Err(AlignmentNeed::HeadingUncertain);
    }
    Ok((heading, sigma))
}
