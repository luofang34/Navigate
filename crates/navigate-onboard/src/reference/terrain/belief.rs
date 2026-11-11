//! What the navigator believes about the vehicle at the middle of an
//! exposure, and the camera poses that follow from it.

use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion, Vector3};
use navigate_camera::{CameraMount, OpticalPose};
use navigate_contract::NavigationSolution;
use navigate_visual::{CameraPose, LocalFrame, VisualError};

/// The vehicle as the navigator believes it at the middle of an exposure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Belief {
    /// `[lat, lon]` in degrees and the height above the ellipsoid in metres
    /// of the body reference point.
    pub position: [f64; 3],
    /// Rotation from the body axes into the north, east, down axes.
    pub body_to_ned: UnitQuaternion<f64>,
    /// Horizontal one-sigma of the position, metres.
    pub sigma_h_m: f64,
}

/// Rotation from the north, east, down axes into the east, north, up axes.
fn enu_from_ned() -> UnitQuaternion<f64> {
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(Matrix3::new(
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
    )))
}

impl Belief {
    /// The belief `lag_s` before `solution`. The filter gives a solution
    /// only at or after its state time, so the position goes back along
    /// the velocity of the solution, and the attitude is the attitude of
    /// the solution. The belief seeds and bounds the search; the fix does
    /// not contain it. `None` without an attitude.
    pub(crate) fn before(solution: &NavigationSolution, lag_s: f64) -> Option<Self> {
        let q = solution.attitude?;
        let p = &solution.position;
        let (lat, lon) = (p.latitude_rad.to_degrees(), p.longitude_rad.to_degrees());
        let frame = LocalFrame::wgs84_enu(lat, lon).ok()?;
        let v = &solution.velocity;
        let back = Vector3::new(
            -v.east_mps * lag_s,
            -v.north_mps * lag_s,
            p.altitude_m + v.down_mps * lag_s,
        );
        let (nn, ee, _) = solution.position_cov.diagonal();
        Some(Self {
            position: frame.geodetic(back),
            body_to_ned: UnitQuaternion::from_quaternion(Quaternion::new(q.w, q.x, q.y, q.z)),
            sigma_h_m: (nn + ee).max(0.0).sqrt(),
        })
    }

    /// The believed camera pose in `frame`.
    pub(crate) fn camera_in(
        &self,
        frame: LocalFrame,
        mount: &CameraMount,
    ) -> Result<CameraPose, VisualError> {
        let [lat, lon, _] = self.position;
        let frame_from_body = frame.local_rotation([lat, lon])? * enu_from_ned() * self.body_to_ned;
        let optical = mount.camera_pose(&frame.local(self.position)?, &frame_from_body);
        Ok(CameraPose {
            position: optical.eye,
            orientation: optical.world_from_gl_eye(),
        })
    }

    /// `[lat, lon]` in degrees and the height in metres of the believed
    /// optical centre, and the believed camera pose in the tangent frame at
    /// the vehicle.
    pub(crate) fn eye(&self, mount: &CameraMount) -> Result<([f64; 3], CameraPose), VisualError> {
        let [lat, lon, _] = self.position;
        let frame = LocalFrame::wgs84_enu(lat, lon)?;
        let pose = self.camera_in(frame, mount)?;
        Ok((frame.geodetic(pose.position), pose))
    }
}

/// The position of the body reference point for a solved camera pose: the
/// lever arm comes off with the attitude that the image gave, so the fix
/// does not contain the attitude belief of the navigator.
pub(crate) fn body_position(camera: &CameraPose, mount: &CameraMount) -> Vector3<f64> {
    let optical = OpticalPose::from_gl(camera.position, camera.orientation);
    let frame_from_body = optical.world_to_cam.inverse() * mount.cam_from_body;
    camera.position - frame_from_body * mount.lever_arm_body_m
}

#[cfg(test)]
mod tests;
