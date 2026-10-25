use crate::ResourceLimits;
use image::{GrayImage, Luma};
use nalgebra::{UnitQuaternion, Vector3};
use navigate_visual::{
    CameraModel, CameraPose, Frame, FrameStamp, LocalFrame, MapRevision, PosePrior, ReferenceView,
};
pub(crate) fn limits() -> ResourceLimits {
    ResourceLimits {
        image_pixels: 10000,
        features_per_image: 512,
        candidates: 3,
        reference_bytes: 200000,
        query_feature_bytes: 10000,
    }
}
fn pose() -> CameraPose {
    CameraPose {
        position: Vector3::new(0.0, 0.0, 100.0),
        orientation: UnitQuaternion::identity(),
    }
}
pub(crate) fn reference() -> ReferenceView {
    ReferenceView {
        map: MapRevision {
            release_id: "test".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(0.0, 0.0).unwrap(),
        pose: pose(),
        image: GrayImage::from_pixel(100, 80, Luma([100])),
        depth_m: vec![100.0; 8000],
    }
}
pub(crate) fn query() -> (Frame, PosePrior) {
    (
        Frame {
            stamp: FrameStamp {
                sequence: 1,
                capture_time_ns: 1,
            },
            camera: CameraModel {
                width: 100,
                height: 80,
                fx: 90.0,
                fy: 90.0,
                cx: 49.5,
                cy: 39.5,
            },
            image: reference().image,
        },
        PosePrior {
            pose: pose(),
            position_radius_m: 10.0,
            attitude_radius_rad: 0.1,
        },
    )
}
