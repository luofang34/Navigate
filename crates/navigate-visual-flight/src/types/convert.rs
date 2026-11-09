//! Conversions from the `navigate-camera` model (ADR-0021).

use navigate_camera::{CameraMount, ImageView, Intrinsics, Pixels};

use super::{CalibrationMode, CameraIntrinsics, CameraMountProfile, FramePixels, PlatformKind};
use crate::FlightVisualError;

impl From<CameraIntrinsics> for Intrinsics {
    /// Copies the pinhole fields. The calibration mode has no counterpart.
    fn from(intrinsics: CameraIntrinsics) -> Self {
        Self {
            width: intrinsics.width,
            height: intrinsics.height,
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
        }
    }
}

impl CameraIntrinsics {
    /// Live frame intrinsics from the pinhole model and a calibration mode.
    #[must_use]
    pub const fn from_intrinsics(
        intrinsics: Intrinsics,
        calibration_confidence: CalibrationMode,
    ) -> Self {
        Self {
            width: intrinsics.width,
            height: intrinsics.height,
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
            calibration_confidence,
        }
    }
}

impl CameraMountProfile {
    /// A mount profile from a [`CameraMount`].
    ///
    /// `cam_from_body` gets the rotation of the mount without a change: the
    /// body axes are forward, right, and down (FRD), and the camera axes are
    /// `+X` right, `+Y` down, and `+Z` forward. The profile has no lever arm,
    /// so the lever arm of the mount is not kept.
    #[must_use]
    pub const fn from_camera_mount(
        mount: &CameraMount,
        has_airframe_occlusion: bool,
        platform: PlatformKind,
    ) -> Self {
        Self {
            cam_from_body: mount.cam_from_body,
            has_airframe_occlusion,
            platform,
        }
    }
}

impl<'a> TryFrom<ImageView<'a>> for FramePixels<'a> {
    type Error = FlightVisualError;

    /// Borrows 8-bit gray and 8-bit RGB pixels without a copy. A stride of
    /// 8-bit samples is also a stride in bytes, so the stride copies.
    ///
    /// # Errors
    /// Refuses 16-bit gray. Apply `ImageView::to_gray8_percentile` first.
    fn try_from(view: ImageView<'a>) -> Result<Self, Self::Error> {
        let (width, height) = (view.width(), view.height());
        match view.pixels() {
            Pixels::Gray8 { data, stride } => Ok(Self::Gray8 {
                data,
                width,
                height,
                stride,
            }),
            Pixels::Rgb8 { data, stride } => Ok(Self::Rgb8 {
                data,
                width,
                height,
                stride,
            }),
            Pixels::Gray16 { .. } => Err(FlightVisualError::Invalid(
                "the live engine accepts 8-bit pixels only; convert 16-bit gray with \
                 ImageView::to_gray8_percentile first"
                    .to_owned(),
            )),
            _ => Err(FlightVisualError::Invalid(
                "the live engine does not accept this pixel format".to_owned(),
            )),
        }
    }
}
