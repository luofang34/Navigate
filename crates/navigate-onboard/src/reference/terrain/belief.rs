//! What the navigator believes about the vehicle at the middle of an
//! exposure, and the camera poses that follow from it.

use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion, Vector3};
use navigate_camera::{CameraMount, OpticalPose};
use navigate_contract::{MonotonicNanos, NavigationSolution};
use navigate_fusion::InertialNavigationFilter;
use navigate_visual::{CameraPose, LocalFrame, VisualError};

use super::names;

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
    /// One-sigma of the attitude, radians, when the solution states it.
    pub sigma_attitude_rad: Option<f64>,
}

/// Where the belief of a frame came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefSource {
    /// The filter gave its state at the exposure: position and attitude
    /// are those of the middle of the exposure.
    FilterAtExposure,
    /// The exposure is older than the history of the filter. The belief
    /// is the solution on delivery with its position moved back along its
    /// velocity; its attitude is that of the delivery. The search bounds
    /// grow with the delay to hold the difference.
    MovedBackFromDelivery,
}

/// The belief of one frame and what the search must add to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Exposure {
    pub belief: Belief,
    pub source: BeliefSource,
    /// Time of vehicle motion that the belief does not hold, seconds: zero
    /// for the state of the filter at the exposure.
    pub unmodelled_lag_s: f64,
}

/// The belief of the navigator for a frame acquired at `acquired` and
/// delivered at `delivered`, or the name of the reason why the frame is
/// not matched.
pub(crate) fn at_exposure(
    filter: &mut InertialNavigationFilter,
    (acquired, delivered): (MonotonicNanos, MonotonicNanos),
    max_capture_lag_s: f64,
) -> Result<Exposure, &'static str> {
    let lag_ns = delivered
        .as_nanos()
        .checked_sub(acquired.as_nanos())
        .ok_or(names::ACQUIRED_AFTER_DELIVERY)?;
    let lag_s = lag_ns as f64 * 1e-9;
    if lag_s > max_capture_lag_s {
        return Err(names::CAPTURE_LAG);
    }
    let on_delivery = filter.tick(delivered);
    let exposure = |solution: &NavigationSolution, source, unmodelled_lag_s| {
        Belief::before(solution, unmodelled_lag_s).map(|belief| Exposure {
            belief,
            source,
            unmodelled_lag_s,
        })
    };
    let past = (lag_ns > 0)
        .then(|| filter.state_at(acquired).ok())
        .flatten()
        .and_then(|past| exposure(&past.solution, BeliefSource::FilterAtExposure, 0.0));
    let source = if lag_ns == 0 {
        BeliefSource::FilterAtExposure
    } else {
        BeliefSource::MovedBackFromDelivery
    };
    past.or_else(|| exposure(on_delivery.as_ref()?, source, lag_s))
        .ok_or(names::NO_SOLUTION)
}

/// Rotation from the north, east, down axes into the east, north, up axes.
fn enu_from_ned() -> UnitQuaternion<f64> {
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(Matrix3::new(
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
    )))
}

impl Belief {
    /// The belief `lag_s` before `solution`: the position goes back along
    /// the velocity of the solution, and the attitude is the attitude of
    /// the solution. A lag of zero is the solution itself. The belief
    /// seeds and bounds the search; the fix does not contain it. `None`
    /// without an attitude.
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
            sigma_attitude_rad: solution.attitude_cov.map(|c| {
                let (xx, yy, zz) = c.diagonal();
                (xx + yy + zz).max(0.0).sqrt()
            }),
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

/// Largest angle of the optical axis from the local down at which the
/// ground point of the axis is taken, radians.
const MAX_GROUND_ANGLE_RAD: f64 = 1.3;

impl Belief {
    /// Where the believed optical axis meets ground at the height
    /// `ground_height_m` above the ellipsoid: `[lat, lon]` in degrees and
    /// the range along the axis in metres. `None` for an axis near or
    /// above the horizon, a camera at or below that ground, and a range
    /// past `far_m`.
    pub(crate) fn ground(
        &self,
        mount: &CameraMount,
        ground_height_m: f64,
        far_m: f64,
    ) -> Option<([f64; 2], f64)> {
        let [lat, lon, _] = self.position;
        let frame = LocalFrame::wgs84_enu(lat, lon).ok()?;
        let pose = self.camera_in(frame, mount).ok()?;
        // The eye axes are right, up, and back: the camera looks along -Z.
        let axis = pose.orientation * -Vector3::z();
        let down = -axis.z;
        let height_m = frame.geodetic(pose.position)[2] - ground_height_m;
        if down < MAX_GROUND_ANGLE_RAD.cos() || height_m <= 0.0 {
            return None;
        }
        let range_m = height_m / down;
        let [lat, lon, _] = frame.geodetic(pose.position + axis * range_m);
        (range_m <= far_m).then_some(([lat, lon], range_m))
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
