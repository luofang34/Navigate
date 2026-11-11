//! The reference view of one frame: a stored keyframe when the believed
//! camera pose is inside its envelope, else a view that the navigator
//! renders from the package.

use std::fmt;
use std::sync::Arc;

use nalgebra::Vector3;
use navigate_camera::CameraMount;
use navigate_data::product::Calibration;
use navigate_visual::{CameraModel, LocalFrame, ReferenceView};
use navigate_visual_render::{RenderError, SceneRequest, SceneWindow, SparseSceneSampler};

use super::belief::Belief;
use super::keyframes::{HeldKeyframe, KeyframeError, view_of};
use super::map::TerrainMap;
use super::names::{BELIEF_OUT_OF_RANGE, KEYFRAME, OUTSIDE_COVERAGE, RENDER, RENDER_REFUSED};

/// A new scene window is this much larger than the frame needs, so that
/// the next frames of a moving vehicle stay inside it.
const WINDOW_MARGIN: f64 = 1.5;
/// Largest angle from the local down at which the reach of a view is
/// computed; past it the reach is the far distance.
const MAX_REACH_ANGLE_RAD: f64 = 1.4;

/// Why a frame has no reference view.
#[derive(Debug)]
pub(crate) enum NoView {
    /// The frame is not matched, for the stated reason.
    Unused(&'static str),
    /// The store failed or gave changed data.
    Read(String),
}

/// The reference view of one frame and the path that gave it.
pub(crate) struct Reference {
    pub view: Arc<ReferenceView>,
    pub path: &'static str,
}

/// The scene window of the last rendered reference.
struct Resident {
    half_side_m: f64,
    imagery_zoom: u32,
    elevation_zoom: u32,
    window: SceneWindow,
}

impl Resident {
    /// Whether a square of half side `need_m` about `eye` is inside the
    /// window, at the same zoom levels.
    fn covers(&self, eye: [f64; 3], need_m: f64, zooms: (u32, u32)) -> bool {
        let Ok(p) = self.window.scene.frame.local([eye[0], eye[1], 0.0]) else {
            return false;
        };
        zooms == (self.imagery_zoom, self.elevation_zoom)
            && p.x.abs().max(p.y.abs()) + need_m <= self.half_side_m
    }
}

/// What one camera keeps between frames to make reference views. Each part
/// is immutable behind its `Arc`, so a clone shares it.
#[derive(Clone, Default)]
pub(crate) struct Views {
    resident: Option<Arc<Resident>>,
    held: Option<HeldKeyframe>,
}

impl fmt::Debug for Views {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Views")
            .field("resident", &self.resident.as_ref().map(|r| r.half_side_m))
            .field("held", &self.held)
            .finish()
    }
}

/// How far a view reaches over the ground from the point below the camera,
/// and the size of one camera pixel on the ground, in metres. The height
/// is above the ellipsoid, which is not less than the height above the
/// ground of a vehicle over land, so the reach is not short.
fn reach_and_footprint_m(
    camera: &CameraModel,
    belief: &Belief,
    mount: &CameraMount,
    far_m: f64,
) -> Option<([f64; 3], f64, f64)> {
    let (eye, pose) = belief.eye(mount).ok()?;
    let axis = pose.orientation * -Vector3::z();
    let off_nadir = (-axis.z).clamp(-1.0, 1.0).acos();
    let half_diagonal =
        (f64::from(camera.width).hypot(f64::from(camera.height)) / 2.0 / camera.fx.min(camera.fy))
            .atan();
    let height = eye[2].max(1.0);
    let edge = off_nadir + half_diagonal;
    let reach = if edge < MAX_REACH_ANGLE_RAD {
        (height * edge.tan()).min(far_m)
    } else {
        far_m
    };
    let footprint = height / off_nadir.min(MAX_REACH_ANGLE_RAD).cos() / camera.fx.max(camera.fy);
    Some((eye, reach, footprint))
}

impl Views {
    /// The reference view for a frame of `camera` at the believed pose,
    /// with a position search of `radius_m`.
    pub(crate) fn reference(
        &mut self,
        map: &TerrainMap,
        camera: &CameraModel,
        calibration: Calibration,
        mount: &CameraMount,
        belief: &Belief,
        radius_m: f64,
    ) -> Result<Reference, NoView> {
        for set in map.keyframes_for(calibration) {
            if let Some(keyframe) = set.inside(belief, mount) {
                return match view_of(map, set, keyframe, &mut self.held) {
                    Ok(view) => Ok(Reference {
                        view,
                        path: KEYFRAME,
                    }),
                    Err(KeyframeError::Read(error)) => Err(NoView::Read(error.to_string())),
                    Err(KeyframeError::Decode(error)) => Err(NoView::Read(error.to_string())),
                };
            }
        }
        self.render(map, camera, mount, belief, radius_m)
    }

    /// Renders the reference at the believed camera pose from the package.
    pub(crate) fn render(
        &mut self,
        map: &TerrainMap,
        camera: &CameraModel,
        mount: &CameraMount,
        belief: &Belief,
        radius_m: f64,
    ) -> Result<Reference, NoView> {
        let config = &map.config;
        let (eye, reach_m, footprint_m) =
            reach_and_footprint_m(camera, belief, mount, config.far_m)
                .ok_or(NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        let resident = self.window(map, eye, reach_m + radius_m, footprint_m)?;
        let scene = &resident.window.scene;
        let refused = |error: RenderError| match error {
            RenderError::Data(e) => NoView::Read(e.to_string()),
            RenderError::Decode(e) => NoView::Read(e.to_string()),
            other => {
                tracing::debug!(error = %other, "the terrain reference was not rendered");
                NoView::Unused(RENDER_REFUSED)
            }
        };
        let pose = belief
            .camera_in(scene.frame, mount)
            .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        let (view, _) =
            SparseSceneSampler::new(scene, config.far_m, config.steps, config.refinements)
                .and_then(|sampler| {
                    sampler.render_adaptive(
                        camera,
                        &pose,
                        config.stride,
                        config.luma_edge,
                        config.depth_edge,
                    )
                })
                .map_err(refused)?;
        let seen = view.depth_m.iter().filter(|d| **d > 0.0).count();
        if (seen as f64) < config.min_view_coverage * view.depth_m.len() as f64 {
            return Err(NoView::Unused(OUTSIDE_COVERAGE));
        }
        Ok(Reference {
            view: Arc::new(view),
            path: RENDER,
        })
    }

    /// The resident scene window when it holds a square of half side
    /// `need_m` about `eye`, else a new window from the package.
    fn window(
        &mut self,
        map: &TerrainMap,
        eye: [f64; 3],
        need_m: f64,
        footprint_m: f64,
    ) -> Result<Arc<Resident>, NoView> {
        let no_tiles = NoView::Unused(OUTSIDE_COVERAGE);
        let (Some(imagery_zoom), Some(elevation_zoom)) = (
            map.imagery.zoom_for(footprint_m),
            map.elevation.zooms.last().copied(),
        ) else {
            return Err(no_tiles);
        };
        let zooms = (imagery_zoom, elevation_zoom);
        if let Some(resident) = self
            .resident
            .as_ref()
            .filter(|r| r.covers(eye, need_m, zooms))
        {
            return Ok(Arc::clone(resident));
        }
        let half_side_m = need_m * WINDOW_MARGIN;
        let request = SceneRequest {
            imagery_layer: map.imagery.layer.key,
            imagery_zoom,
            elevation_layer: map.elevation.layer.key,
            elevation_zoom,
            centre: [eye[0], eye[1]],
            half_side_m,
        };
        let window = map.reader.scene(&request).map_err(|error| match error {
            RenderError::Data(e) => NoView::Read(e.to_string()),
            RenderError::Decode(e) => NoView::Read(e.to_string()),
            other => {
                tracing::debug!(error = %other, "no scene window for the terrain reference");
                NoView::Unused(OUTSIDE_COVERAGE)
            }
        })?;
        let resident = Arc::new(Resident {
            half_side_m,
            imagery_zoom,
            elevation_zoom,
            window,
        });
        self.resident = Some(Arc::clone(&resident));
        Ok(resident)
    }
}

/// The geodetic point, `[lat, lon]` in degrees, of a position in a frame.
pub(crate) fn lat_lon(frame: LocalFrame, position: Vector3<f64>) -> [f64; 2] {
    let [lat, lon, _] = frame.geodetic(position);
    [lat, lon]
}
