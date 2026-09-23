//! Frame inputs and map reference identity.

use crate::{CameraModel, CameraPose, LocalFrame, VisualError};
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

impl Frame {
    /// Bind exact pixels, calibration and capture stamp to a processing identity.
    ///
    /// This digest is not an independence claim. Different stamps can still
    /// share image or map evidence. The host must retain capture-stream identity.
    pub fn evidence_sha256(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"navigate-visual-frame-v1");
        hash.update(self.stamp.sequence.to_le_bytes());
        hash.update(self.stamp.capture_time_ns.to_le_bytes());
        hash.update(self.camera.width.to_le_bytes());
        hash.update(self.camera.height.to_le_bytes());
        for value in [
            self.camera.fx,
            self.camera.fy,
            self.camera.cx,
            self.camera.cy,
        ] {
            hash.update(value.to_le_bytes());
        }
        hash.update(self.image.as_raw());
        format!("{hash:x}", hash = hash.finalize())
    }
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
/// Depth is a supplied world-model surface, not independently verified scene
/// geometry. Terrain and richer surfaces can use the same optical-depth input.
/// Source errors, geographic registration and shared evidence remain unknown
/// unless the host has separate information about them.
pub struct ReferenceView {
    /// Map release used to render this reference.
    pub map: MapRevision,
    /// Frame of `pose`, of the depth surface, and of every pose solved from them.
    pub frame: LocalFrame,
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
        self.frame.validate()?;
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
