//! Frame inputs and map reference identity.

use crate::{CameraModel, CameraPose, VisualError};
use image::GrayImage;

/// Frame ordering within one capture stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStamp {
    /// Source sequence number. Producers use wrapping increments.
    pub sequence: u64,
    /// Monotonic acquisition time in the host's clock domain.
    pub capture_time_ns: u64,
}

/// One undistorted camera frame. Video decoders remain in the host.
pub struct Frame {
    /// Frame acquisition stamp.
    pub stamp: FrameStamp,
    /// Camera intrinsics for these pixels.
    pub camera: CameraModel,
    /// Grayscale image. No location metadata enters the matcher.
    pub image: GrayImage,
}

/// The immutable map selection shared with the display renderer.
///
/// Pin one installed source selection for display and localization. The host
/// verifies the source files and keeps them available while they are in use.
/// Switch both consumers together when a new selection is ready. This library
/// neither downloads files nor verifies their bytes. Derived feature caches
/// must also bind the camera calibration and matcher/model identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapRevision {
    /// Selected package release ID.
    pub release_id: String,
    /// SHA-256 of the manifest that identifies imagery, elevation, and the anchor.
    pub manifest_sha256: String,
}

/// Reference pixels and visible surface depth from a candidate camera pose.
///
/// Render with the query intrinsics and the declared camera pose. Depth measures
/// distance along the camera optical axis, not Euclidean range. Convert reversed
/// GPU depth before construction. Exclude sky, missing terrain, and missing
/// imagery. The reference and prior must use the same local frame and datum.
pub struct ReferenceView {
    /// Map release used to render this reference.
    pub map: MapRevision,
    /// Pose of the reference camera in the map's local frame.
    pub pose: CameraPose,
    /// Reference image in the query camera's calibrated projection.
    pub image: GrayImage,
    /// Optical-axis depth in metres, row-major. Zero marks sky or missing data.
    pub depth_m: Vec<f32>,
}

impl ReferenceView {
    pub(crate) fn validate(&self, frame: &Frame) -> Result<(), VisualError> {
        frame.camera.validate()?;
        self.pose.validate()?;
        if frame.image.dimensions() != self.image.dimensions() {
            return Err(VisualError::Dimensions {
                query: frame.image.dimensions(),
                reference: self.image.dimensions(),
            });
        }
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height)
            || self.depth_m.len() != self.image.as_raw().len()
        {
            return Err(VisualError::Invalid {
                field: "frame dimensions or depth length",
            });
        }
        if self.map.release_id.is_empty()
            || self.map.manifest_sha256.len() != 64
            || !self
                .map
                .manifest_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Err(VisualError::Invalid {
                field: "map revision",
            });
        }
        Ok(())
    }
}
