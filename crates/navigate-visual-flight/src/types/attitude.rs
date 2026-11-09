//! The camera attitude prior from a body attitude and its mount.

use std::f64::consts::TAU;

use nalgebra::{UnitQuaternion, Vector3};

use super::{AttitudePrior, CameraMountProfile};

/// Below this horizontal part of the optical axis, the axis is vertical and
/// the heading comes from the image right axis.
const VERTICAL_AXIS: f64 = 1e-9;

impl AttitudePrior {
    /// The camera attitude prior of a camera on `mount`, from the rotation
    /// from the body axes (forward, right, down) into the local north, east,
    /// and down (NED) axes.
    ///
    /// The angles follow the camera rotation of the engine: `yaw_rad` is the
    /// heading of the optical axis clockwise from north, `pitch_rad` its
    /// angle below the horizon, and `roll_rad` the turn of the image right
    /// axis from the horizontal toward the image down axis. With a vertical
    /// optical axis the heading is that of the image up axis, and the roll
    /// is zero. `body_ang_vel_rad_s` stays in the body axes.
    #[must_use]
    pub fn from_body_attitude(
        ned_from_body: &UnitQuaternion<f64>,
        mount: &CameraMountProfile,
        tilt_sigma_rad: f64,
        yaw_sigma_rad: f64,
        body_ang_vel_rad_s: Option<[f64; 3]>,
    ) -> Self {
        let ned_from_cam = ned_from_body * mount.cam_from_body.inverse();
        let forward = ned_from_cam * Vector3::z();
        let right = ned_from_cam * Vector3::x();
        let heading = if forward.x.hypot(forward.y) > VERTICAL_AXIS {
            forward.y.atan2(forward.x)
        } else {
            (-right.x).atan2(right.y)
        };
        // The image right axis at zero roll, and the image down axis.
        let level_right = Vector3::new(-heading.sin(), heading.cos(), 0.0);
        let level_down = forward.cross(&level_right).normalize();
        Self {
            roll_rad: right.dot(&level_down).atan2(right.dot(&level_right)),
            pitch_rad: forward.z.clamp(-1.0, 1.0).asin(),
            yaw_rad: Some(heading.rem_euclid(TAU)),
            tilt_sigma_rad,
            yaw_sigma_rad,
            body_ang_vel_rad_s,
        }
    }
}
