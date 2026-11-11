//! The reference view of one frame: a stored keyframe when the believed
//! camera pose is inside its envelope, else a view that the navigator
//! renders from the package.

use std::fmt;
use std::sync::Arc;

use nalgebra::Vector3;
use navigate_camera::{CameraMount, Intrinsics};
use navigate_visual::{CameraModel, ReferenceView};
use navigate_visual_render::{RenderError, SceneRequest, SceneWindow, SparseSceneSampler};

use super::belief::Belief;
use super::config::{WorkingCamera, WorkingLimit};
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

/// The reference view of one frame, the path that gave it, and the camera
/// of the view: the frame is resampled to that camera for the match.
pub(crate) struct Reference {
    pub view: Arc<ReferenceView>,
    pub path: &'static str,
    pub working: WorkingCamera,
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

/// The camera and the belief of one frame, as a reference view needs them.
pub(crate) struct Sight<'a> {
    /// The camera as the host declared it.
    pub declared: &'a Intrinsics,
    pub mount: &'a CameraMount,
    pub belief: &'a Belief,
    /// The position search radius, metres.
    pub radius_m: f64,
    /// Height above the ellipsoid of the ground that the last match saw.
    pub ground_height_m: Option<f64>,
}

/// The pinhole model of intrinsics.
pub(crate) fn model(k: &Intrinsics) -> CameraModel {
    CameraModel {
        width: k.width,
        height: k.height,
        fx: k.fx,
        fy: k.fy,
        cx: k.cx,
        cy: k.cy,
    }
}

impl Sight<'_> {
    /// The believed optical centre (`[lat, lon]` in degrees and the height
    /// in metres), how far the view reaches over the ground from the point
    /// below the camera, and the range to the ground along the optical
    /// axis, in metres. Without the ground height of a match, the height
    /// is above the ellipsoid, which is not less than the height above
    /// the ground of a vehicle over land, so the reach is not short.
    fn geometry(&self, far_m: f64) -> Option<([f64; 3], f64, f64)> {
        let (eye, pose) = self.belief.eye(self.mount).ok()?;
        let axis = pose.orientation * -Vector3::z();
        let off_nadir = (-axis.z).clamp(-1.0, 1.0).acos();
        let k = self.declared;
        let half_diagonal =
            (f64::from(k.width).hypot(f64::from(k.height)) / 2.0 / k.fx.min(k.fy)).atan();
        let height = (eye[2] - self.ground_height_m.unwrap_or(0.0)).max(1.0);
        let edge = off_nadir + half_diagonal;
        let reach = if edge < MAX_REACH_ANGLE_RAD {
            (height * edge.tan()).min(far_m)
        } else {
            far_m
        };
        let range = height / off_nadir.min(MAX_REACH_ANGLE_RAD).cos();
        Some((eye, reach, range))
    }
}

impl Views {
    /// The reference view for the frame of `sight`.
    pub(crate) fn reference(
        &mut self,
        map: &TerrainMap,
        sight: &Sight<'_>,
    ) -> Result<Reference, NoView> {
        for (set, intrinsics) in map.keyframes_for(sight.declared) {
            if let Some(keyframe) = set.inside(sight.belief, sight.mount) {
                let (_, _, range_m) = sight
                    .geometry(map.config.far_m)
                    .ok_or(NoView::Unused(BELIEF_OUT_OF_RANGE))?;
                return match view_of(map, set, keyframe, &mut self.held) {
                    Ok(view) => Ok(Reference {
                        view,
                        path: KEYFRAME,
                        working: WorkingCamera {
                            intrinsics,
                            ground_pixel_m: range_m / intrinsics.fx,
                            limit: WorkingLimit::Keyframe,
                        },
                    }),
                    Err(KeyframeError::Read(error)) => Err(NoView::Read(error.to_string())),
                    Err(KeyframeError::Decode(error)) => Err(NoView::Read(error.to_string())),
                };
            }
        }
        self.render(map, sight)
    }

    /// Renders the reference at the believed camera pose from the package,
    /// at the working resolution of the frame.
    pub(crate) fn render(
        &mut self,
        map: &TerrainMap,
        sight: &Sight<'_>,
    ) -> Result<Reference, NoView> {
        let config = &map.config;
        let (eye, reach_m, range_m) = sight
            .geometry(config.far_m)
            .ok_or(NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        // One map pixel for one pixel of the declared camera selects the
        // zoom; the working camera then takes the pixels of that zoom.
        let native_pixel_m = range_m / sight.declared.fx.max(sight.declared.fy);
        let (Some(imagery_zoom), Some(elevation_zoom)) = (
            map.imagery.zoom_for(native_pixel_m),
            map.elevation.zooms.last().copied(),
        ) else {
            return Err(NoView::Unused(OUTSIDE_COVERAGE));
        };
        let texel_m = map.imagery.ground_sample_m(imagery_zoom);
        let working = config
            .working_camera(sight.declared, range_m, texel_m)
            .ok_or(NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        let zooms = (imagery_zoom, elevation_zoom);
        let resident = self.window(map, eye, reach_m + sight.radius_m, zooms)?;
        let scene = &resident.window.scene;
        let refused = |error: RenderError| match error {
            RenderError::Data(e) => NoView::Read(e.to_string()),
            RenderError::Decode(e) => NoView::Read(e.to_string()),
            other => {
                tracing::debug!(error = %other, "the terrain reference was not rendered");
                NoView::Unused(RENDER_REFUSED)
            }
        };
        let pose = sight
            .belief
            .camera_in(scene.frame, sight.mount)
            .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        let (view, _) =
            SparseSceneSampler::new(scene, config.far_m, config.steps, config.refinements)
                .and_then(|sampler| {
                    sampler.render_adaptive(
                        &model(&working.intrinsics),
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
            working,
        })
    }

    /// The resident scene window when it holds a square of half side
    /// `need_m` about `eye` at the zoom levels `zooms` (imagery,
    /// elevation), else a new window from the package.
    fn window(
        &mut self,
        map: &TerrainMap,
        eye: [f64; 3],
        need_m: f64,
        zooms: (u32, u32),
    ) -> Result<Arc<Resident>, NoView> {
        let (imagery_zoom, elevation_zoom) = zooms;
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
