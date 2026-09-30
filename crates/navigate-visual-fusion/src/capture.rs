//! Camera capture alignment for host or flight-controller attitude samples.
//!
//! The host decodes the transport and supplies clock error bounds. Use a new
//! clock domain for each MCU boot. Extend wrapping boot counters before use.
//! Packet arrival time is not the attitude measurement time.
mod clock;
mod error;
use crate::FrameCapture;
pub use clock::{ClockAlignment, TimedCapture};
pub use error::CaptureError;
use nalgebra::{Quaternion, UnitQuaternion};
use navigate_contract::{AttitudeQuaternion, DurationNanos};

/// An attitude at its measurement time. The body axes are forward, right, down.
#[derive(Clone, Copy, Debug)]
pub struct AttitudeSample {
    /// Measurement time mapped to the host clock, with a known error bound.
    pub time: TimedCapture,
    /// Rotation from the body axes to local north, east, down. Scalar first.
    pub body_to_ned: AttitudeQuaternion,
    /// Conservative angular RMS error in radians. None means unknown.
    pub angular_rms_rad: Option<f64>,
}

/// Camera mounting and limits for interpolation between attitude samples.
#[derive(Clone, Copy, Debug)]
pub struct CameraAlignment {
    /// Rotation from camera right, up, back to body forward, right, down.
    /// A gimbal host supplies this rotation at capture time.
    pub eye_to_body: UnitQuaternion<f64>,
    /// Conservative mounting angular RMS error. None means unknown.
    pub mounting_rms_rad: Option<f64>,
    /// Conservative angular RMS error of the interpolation model.
    /// This includes unresolved motion between samples. None means unknown.
    pub interpolation_rms_rad: Option<f64>,
    /// Largest permitted interval between attitude measurements.
    pub maximum_gap: DurationNanos,
    /// Upper bound on angular speed over the interval, in radians per second.
    pub angular_speed_bound_rad_s: f64,
}

/// Interpolate body attitude at camera capture time and apply camera mounting.
///
/// Timestamp error intervals must fit between the attitude samples. Error terms
/// are added as RMS allowances; no independence is assumed. This result narrows
/// visual search. It is not a fused attitude or an integrity bound.
///
/// # Errors
/// Rejects foreign clocks, absent budgets, invalid rotations, gaps,
/// extrapolation, and motion above the supplied angular-speed bound.
pub fn camera_capture(
    capture: TimedCapture,
    before: AttitudeSample,
    after: AttitudeSample,
    alignment: CameraAlignment,
) -> Result<FrameCapture, CaptureError> {
    let gap = bracket(capture, before.time, after.time, alignment.maximum_gap)?;
    let first = quaternion(before.body_to_ned)?;
    let last = quaternion(after.body_to_ned)?;
    let rate = alignment.angular_speed_bound_rad_s;
    let sample_error = rms(before.angular_rms_rad, "first attitude")?
        .max(rms(after.angular_rms_rad, "last attitude")?);
    let mounting = rms(alignment.mounting_rms_rad, "camera mounting")?;
    let interpolation = rms(alignment.interpolation_rms_rad, "attitude interpolation")?;
    if !rate.is_finite() || rate < 0.0 || rate * gap >= std::f64::consts::PI {
        return Err(CaptureError::Invalid {
            field: "angular speed or ambiguous interval",
        });
    }
    let timing_error = capture.error_bound.as_nanos() as f64 * 1e-9
        + before
            .time
            .error_bound
            .as_nanos()
            .max(after.time.error_bound.as_nanos()) as f64
            * 1e-9;
    if first.angle_to(&last) > rate * (gap + 2.0 * timing_error) + 2.0 * sample_error {
        return Err(CaptureError::Invalid {
            field: "attitude exceeds angular speed bound",
        });
    }
    let fraction = (capture.at.as_nanos() - before.time.at.as_nanos()) as f64
        / (after.time.at.as_nanos() - before.time.at.as_nanos()) as f64;
    let attitude = first
        .try_slerp(&last, fraction, 1e-9)
        .ok_or(CaptureError::Invalid {
            field: "ambiguous attitude interpolation",
        })?;
    let q = alignment.eye_to_body.quaternion();
    quaternion(AttitudeQuaternion::new(q.w, q.i, q.j, q.k))?;
    let axis = std::f64::consts::FRAC_1_SQRT_2;
    let ned_to_enu = UnitQuaternion::new_normalize(Quaternion::new(0.0, axis, axis, 0.0));
    Ok(FrameCapture {
        at: capture.at,
        clock: capture.clock,
        time_error_bound: capture.error_bound,
        orientation: ned_to_enu * attitude * alignment.eye_to_body,
        attitude_sigma_rad: sample_error + mounting + interpolation + rate * timing_error,
    })
}

fn quaternion(value: AttitudeQuaternion) -> Result<UnitQuaternion<f64>, CaptureError> {
    let q = Quaternion::new(value.w, value.x, value.y, value.z);
    if q.coords.iter().any(|v| !v.is_finite()) || (q.norm_squared() - 1.0).abs() > 1e-5 {
        return Err(CaptureError::Invalid {
            field: "unit attitude quaternion",
        });
    }
    Ok(UnitQuaternion::new_normalize(q))
}
fn rms(value: Option<f64>, field: &'static str) -> Result<f64, CaptureError> {
    let value = value.ok_or(CaptureError::UnknownError { field })?;
    if !value.is_finite() || value < 0.0 {
        return Err(CaptureError::Invalid { field });
    }
    Ok(value)
}
fn bracket(
    capture: TimedCapture,
    before: TimedCapture,
    after: TimedCapture,
    maximum: DurationNanos,
) -> Result<f64, CaptureError> {
    if capture.clock != before.clock || capture.clock != after.clock {
        return Err(CaptureError::ClockDomain);
    }
    let gap = after
        .at
        .as_nanos()
        .checked_sub(before.at.as_nanos())
        .filter(|gap| *gap > 0 && *gap <= maximum.as_nanos())
        .ok_or(CaptureError::Invalid {
            field: "attitude sample interval",
        })?;
    let (first, last) = (before.at.as_nanos(), after.at.as_nanos());
    let c = capture.at.as_nanos();
    let earliest = c.checked_sub(capture.error_bound.as_nanos());
    let latest = c.checked_add(capture.error_bound.as_nanos());
    if earliest
        .zip(first.checked_add(before.error_bound.as_nanos()))
        .is_none_or(|(c, b)| c < b)
        || latest
            .zip(last.checked_sub(after.error_bound.as_nanos()))
            .is_none_or(|(c, a)| c > a)
    {
        return Err(CaptureError::NotBracketed {
            capture_ns: c,
            first_ns: first,
            last_ns: last,
        });
    }
    Ok(gap as f64 * 1e-9)
}

#[cfg(test)]
mod tests;
