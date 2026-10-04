//! Camera intrinsics, lens model, mounting, and the vertical reference.

use crate::pose::Pose;
use nalgebra::UnitQuaternion;
use navigate_visual::CameraModel;

/// The lens model of the source images.
///
/// The visual crate needs undistorted pinhole images. A host that does not
/// know the distortion declares `Unknown`. The session then keeps all poses,
/// but the error budget of every result names the unknown lens model.
#[derive(Clone, Debug, PartialEq)]
pub enum LensModel {
    /// The host removed the lens distortion before matching.
    Undistorted {
        /// Root mean square residual of the distortion calibration, in pixels.
        residual_rms_px: f64,
    },
    /// The images keep a lens distortion that nobody calibrated.
    Unknown,
}

/// Where the camera is on the vehicle body.
#[derive(Clone, Debug, PartialEq)]
pub enum CameraMount {
    /// A fixed mount. `body_from_camera` gives the camera position and axes in
    /// the body frame. Camera axes are right, up, and back.
    Fixed {
        /// Body-from-camera transform.
        body_from_camera: Pose,
    },
    /// A gimbal. Each frame needs a gimbal attitude for body results.
    Gimbal {
        /// Body-from-gimbal-base transform.
        body_from_base: Pose,
        /// Clock of the gimbal attitude samples. It must be the frame clock.
        clock: navigate_contract::ClockDomainId,
    },
    /// The mount is not known. Body height and body position are not available.
    Unknown,
}

impl CameraMount {
    /// Body-from-camera for one frame, or `None` when the mount is unknown or
    /// the gimbal attitude of the frame is not available.
    pub fn body_from_camera(&self, gimbal: Option<&UnitQuaternion<f64>>) -> Option<Pose> {
        match self {
            Self::Fixed { body_from_camera } => Some(*body_from_camera),
            Self::Gimbal { body_from_base, .. } => gimbal
                .map(|rotation| body_from_base * Pose::from_parts(Default::default(), *rotation)),
            Self::Unknown => None,
        }
    }
}

/// The vertical reference of heights and terrain elevations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerticalReference {
    /// Height above the WGS84 ellipsoid.
    Wgs84Ellipsoid,
    /// Height above a named geoid model.
    Orthometric {
        /// Geoid model name, for example `EGM2008`.
        geoid: String,
    },
    /// The data does not state its vertical reference.
    Unknown,
}

/// Calibration of the session camera.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    /// Pinhole intrinsics of the processed images.
    pub intrinsics: CameraModel,
    /// Lens model of the processed images.
    pub lens: LensModel,
    /// Camera mount on the vehicle.
    pub mount: CameraMount,
    /// Vertical reference of the map package and its rendered depth.
    pub map_vertical: VerticalReference,
}
