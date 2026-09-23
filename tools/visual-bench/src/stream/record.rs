//! Data transfer records. Camera poses use the selected map anchor.

use super::prior::PriorRecord;
use crate::{BenchError, read_blocking};
use navigate_visual::{CameraModel, Frame, FrameStamp};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CameraRecord {
    /// Calibrated undistorted image width in pixels.
    pub width: u32,
    /// Calibrated undistorted image height in pixels.
    pub height: u32,
    /// Horizontal focal length in pixels.
    pub fx: f64,
    /// Vertical focal length in pixels.
    pub fy: f64,
    /// Principal point column. Integer coordinates identify pixel centers.
    pub cx: f64,
    /// Principal point row. Image rows increase downward.
    pub cy: f64,
}

impl CameraRecord {
    pub fn model(self) -> CameraModel {
        CameraModel {
            width: self.width,
            height: self.height,
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
        }
    }
}

impl From<CameraModel> for CameraRecord {
    fn from(model: CameraModel) -> Self {
        Self {
            width: model.width,
            height: model.height,
            fx: model.fx,
            fy: model.fy,
            cx: model.cx,
            cy: model.cy,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FrameRecord {
    /// Sequence within one stream. Wrapping increments are permitted.
    pub sequence: u64,
    /// Acquisition time in one monotonic input clock. Video sidecars use relative PTS.
    pub capture_time_ns: u64,
    /// Decoded image file path. Video sidecars do not use this field.
    pub image: PathBuf,
    /// Intrinsics remain constant within a command invocation.
    pub camera: CameraRecord,
    /// Camera pose and acceptance bounds, in the selected map frame.
    pub prior: PriorRecord,
}

impl FrameRecord {
    pub fn load_blocking(&self, base: &Path) -> Result<Frame, BenchError> {
        let path = base.join(&self.image);
        let bytes = read_blocking(&path)?;
        let image = image::load_from_memory(&bytes)
            .map_err(|source| BenchError::Image { path, source })?
            .to_luma8();
        let camera = self.camera.model();
        camera.validate()?;
        Ok(Frame {
            stamp: FrameStamp {
                sequence: self.sequence,
                capture_time_ns: self.capture_time_ns,
            },
            camera,
            image,
        })
    }
}
