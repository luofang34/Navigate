//! Route keyframes of the terrain map: stored reference views that the
//! navigator uses inside their validity envelope (ADR-0022, representation
//! K).

use std::sync::Arc;

use navigate_data::product::{Calibration, PlannedPose, ValidityEnvelope};
use navigate_data::{DataError, Sha256Digest};
use navigate_visual::{CameraModel, LocalFrame, ReferenceView};
use navigate_visual_render::RenderError;
use navigate_visual_render::keyframes::{camera_orientation, reference_view};

use super::belief::Belief;
use super::map::TerrainMap;
use navigate_camera::{CameraMount, Intrinsics};

/// One stored keyframe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Keyframe {
    pub pose: PlannedPose,
    pub blob: Sha256Digest,
    pub bytes: u64,
}

/// The keyframes of one product chunk that the navigator uses.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KeyframeSet {
    /// The camera that the views were rendered for.
    pub calibration: Calibration,
    /// The poses about each planned pose in which a view is valid.
    pub envelope: ValidityEnvelope,
    /// Angle of the optical axis of each view below the horizontal, radians.
    pub depression_rad: f64,
    pub entries: Vec<Keyframe>,
}

/// The camera of a product chunk when it is the declared camera resampled
/// to another size: the same field of view, so a frame of the declared
/// camera can be resampled to the stored views.
pub(crate) fn resampled_to(declared: &Intrinsics, stored: &Calibration) -> Option<Intrinsics> {
    let resampled = declared.resized([stored.width, stored.height]).ok()?;
    let k = CameraModel {
        width: resampled.width,
        height: resampled.height,
        fx: resampled.fx,
        fy: resampled.fy,
        cx: resampled.cx,
        cy: resampled.cy,
    };
    (calibration_of(&k) == *stored).then_some(resampled)
}

/// The calibration of a camera in the integer units of a product chunk.
pub(crate) fn calibration_of(camera: &CameraModel) -> Calibration {
    let micro = |v: f64| (v * 1e6).round() as i64;
    Calibration {
        width: camera.width,
        height: camera.height,
        fx_upx: micro(camera.fx),
        fy_upx: micro(camera.fy),
        cx_upx: micro(camera.cx),
        cy_upx: micro(camera.cy),
    }
}

impl KeyframeSet {
    /// The keyframe nearest to the believed optical centre, when the
    /// believed camera pose is inside its envelope: the horizontal
    /// distance, the height difference, and the angle of the rotation from
    /// the stored view to the believed view. The rotation angle bounds the
    /// heading difference and also holds a banked or pitched view to the
    /// envelope.
    pub(crate) fn inside(&self, belief: &Belief, mount: &CameraMount) -> Option<&Keyframe> {
        let (eye, _) = belief.eye(mount).ok()?;
        let lateral_m = |entry: &&Keyframe| {
            let ([lat, lon], _, _) = entry.pose.to_si();
            let north = (eye[0] - lat).to_radians() * 6_371_000.0;
            let east = (eye[1] - lon).to_radians() * 6_371_000.0 * lat.to_radians().cos();
            north.hypot(east)
        };
        let nearest = self
            .entries
            .iter()
            .min_by(|a, b| lateral_m(a).total_cmp(&lateral_m(b)))?;
        let ([lat, lon], height, heading) = nearest.pose.to_si();
        let frame = LocalFrame::wgs84_enu(lat, lon).ok()?;
        let believed = belief.camera_in(frame, mount).ok()?;
        let stored = camera_orientation(heading, self.depression_rad);
        let e = &self.envelope;
        let inside = believed.position.xy().norm() * 1e3 <= f64::from(e.lateral_mm)
            && (believed.position.z - height).abs() * 1e3 <= f64::from(e.vertical_mm)
            && believed.orientation.angle_to(&stored) * 1e6 <= f64::from(e.heading_urad);
        inside.then_some(nearest)
    }
}

/// A decoded keyframe, kept while consecutive frames use the same one.
#[derive(Clone)]
pub(crate) struct HeldKeyframe {
    pub blob: Sha256Digest,
    pub view: Arc<ReferenceView>,
}

impl std::fmt::Debug for HeldKeyframe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeldKeyframe")
            .field("blob", &self.blob)
            .finish_non_exhaustive()
    }
}

/// Why a stored keyframe gave no reference view.
#[derive(Debug)]
pub(crate) enum KeyframeError {
    /// The store failed or gave changed data.
    Read(DataError),
    /// The blob does not decode.
    Decode(RenderError),
}

/// The reference view of `keyframe` of `set`, from `held` when it holds
/// that keyframe, else read from the store, checked, and decoded.
pub(crate) fn view_of(
    map: &TerrainMap,
    set: &KeyframeSet,
    keyframe: &Keyframe,
    held: &mut Option<HeldKeyframe>,
) -> Result<Arc<ReferenceView>, KeyframeError> {
    if let Some(h) = held.as_ref().filter(|h| h.blob == keyframe.blob) {
        return Ok(Arc::clone(&h.view));
    }
    let bytes = map
        .reader
        .blob(keyframe.blob, keyframe.bytes)
        .map_err(KeyframeError::Read)?;
    let view = reference_view(map.revision(), &keyframe.pose, set.depression_rad, &bytes)
        .map_err(KeyframeError::Decode)?;
    let view = Arc::new(view);
    *held = Some(HeldKeyframe {
        blob: keyframe.blob,
        view: Arc::clone(&view),
    });
    Ok(view)
}
