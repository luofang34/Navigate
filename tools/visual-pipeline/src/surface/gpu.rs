//! Resident `wgpu-terrain-v1` GPU renderer and depth-buffer unprojector for [`Surface`].

mod atlas;
mod camera;

pub(crate) use camera::surface_pose_to_gpu;

use super::{Surface, mercator_tile_coords};
use crate::PipelineError;
use atlas::{
    GPU_TILE_PIXELS, MAX_ATLAS_DIM_PX, MAX_SUBATLAS_TILES, build_image_atlas,
    build_image_atlas_window, build_terrain_atlas, build_terrain_atlas_for_imagery_window,
    sample_depth, subatlas_contains_comfortably, subatlas_window_around, surface_imagery_bounds,
};
use camera::{camera_model, default_limits, manifest_hex64};
use image::GrayImage;
use nalgebra::{Matrix3, Vector2};
use navigate_visual::{CameraModel, CameraPose, LocalFrame, MapRevision, ReferenceRenderer};
use navigate_visual_render::{
    AtlasLayout, GpuRenderer, ImageAtlas, RenderLimits, RenderScene, TerrainAtlas, wgpu,
};
use std::{
    collections::HashSet,
    sync::{Mutex, OnceLock},
};

static FAILED_GPU_INIT: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CameraKey {
    width: u32,
    height: u32,
    fx_bits: u64,
    fy_bits: u64,
    cx_bits: u64,
    cy_bits: u64,
}

impl CameraKey {
    fn from_model(c: CameraModel) -> Self {
        Self {
            width: c.width,
            height: c.height,
            fx_bits: c.fx.to_bits(),
            fy_bits: c.fy.to_bits(),
            cx_bits: c.cx.to_bits(),
            cy_bits: c.cy.to_bits(),
        }
    }
}

#[derive(Clone)]
pub(super) struct SceneTemplate {
    map: MapRevision,
    frame: LocalFrame,
    pub(super) imagery_bounds: [u32; 4],
    pub(super) subatlas_window: Option<[u32; 4]>,
    pub(super) imagery_layout: AtlasLayout,
    imagery_pixels: Vec<u8>,
    imagery_coverage: Vec<u8>,
    pub(super) terrain_layout: AtlasLayout,
    terrain_heights: Vec<f32>,
    terrain_coverage: Vec<u8>,
}

impl SceneTemplate {
    pub(super) fn from_surface(surface: &Surface) -> Option<Self> {
        Self::from_surface_with_span(surface, MAX_SUBATLAS_TILES)
    }

    pub(super) fn from_surface_with_span(surface: &Surface, max_tiles: u32) -> Option<Self> {
        let frame = LocalFrame::wgs84_enu(surface.origin_lat, surface.origin_lon).ok()?;
        let imagery_bounds = surface_imagery_bounds(surface)?;
        let max_dim = max_tiles
            .saturating_mul(GPU_TILE_PIXELS)
            .min(MAX_ATLAS_DIM_PX);
        let (subatlas_window, imagery, terrain) = if max_tiles >= MAX_SUBATLAS_TILES
            && let (Some(img), Some(ter)) =
                (build_image_atlas(surface), build_terrain_atlas(surface))
        {
            (None, img, ter)
        } else {
            let (cx, cy, _, _) =
                mercator_tile_coords(surface.origin_lat, surface.origin_lon, surface.imagery_zoom);
            let win = subatlas_window_around(imagery_bounds, cx, cy, max_tiles);
            let img = build_image_atlas_window(surface, win, max_dim)?;
            let ter = build_terrain_atlas_for_imagery_window(surface, win, max_dim)?;
            (Some(win), img, ter)
        };
        Some(Self {
            map: MapRevision {
                release_id: "catalog-surface".into(),
                manifest_sha256: manifest_hex64(&surface.catalog_sha256),
            },
            frame,
            imagery_bounds,
            subatlas_window,
            imagery_layout: imagery.layout,
            imagery_pixels: imagery.pixels,
            imagery_coverage: imagery.coverage,
            terrain_layout: terrain.layout,
            terrain_heights: terrain.heights_m,
            terrain_coverage: terrain.coverage,
        })
    }

    pub(super) fn update_subatlas_at(
        &mut self,
        surface: &Surface,
        lat_deg: f64,
        lon_deg: f64,
        max_tex_dim: u32,
    ) -> bool {
        let Some(cur_win) = self.subatlas_window else {
            return false;
        };
        let (cx, cy, _, _) = mercator_tile_coords(lat_deg, lon_deg, surface.imagery_zoom);
        if subatlas_contains_comfortably(self.imagery_bounds, cur_win, cx, cy) {
            return false;
        }
        let max_tiles = (max_tex_dim / GPU_TILE_PIXELS).clamp(1, MAX_SUBATLAS_TILES);
        let next_win = subatlas_window_around(self.imagery_bounds, cx, cy, max_tiles);
        if next_win == cur_win {
            return false;
        }
        let Some(img) = build_image_atlas_window(surface, next_win, max_tex_dim) else {
            return false;
        };
        let Some(ter) = build_terrain_atlas_for_imagery_window(surface, next_win, max_tex_dim)
        else {
            return false;
        };
        self.subatlas_window = Some(next_win);
        self.imagery_layout = img.layout;
        self.imagery_pixels = img.pixels;
        self.imagery_coverage = img.coverage;
        self.terrain_layout = ter.layout;
        self.terrain_heights = ter.heights_m;
        self.terrain_coverage = ter.coverage;
        true
    }

    pub(super) fn to_render_scene(&self) -> RenderScene {
        RenderScene {
            map: self.map.clone(),
            frame: self.frame,
            imagery: ImageAtlas {
                layout: self.imagery_layout,
                pixels: self.imagery_pixels.clone(),
                coverage: self.imagery_coverage.clone(),
            },
            terrain: TerrainAtlas {
                layout: self.terrain_layout,
                heights_m: self.terrain_heights.clone(),
                coverage: self.terrain_coverage.clone(),
            },
        }
    }
}

/// Grayscale view and optical-axis depth buffer rendered on the GPU for one [`Surface`] pose.
pub struct GpuSurfaceView {
    /// Rendered grayscale orthoimagery view.
    pub image: GrayImage,
    /// Row-major positive optical-axis depth in metres (`0.0` where uncovered or sky).
    pub depth_m: Vec<f32>,
    /// Pinhole camera model of the view.
    pub camera: CameraModel,
    /// Camera pose in `frame`.
    pub pose: CameraPose,
    /// WGS84 `ENU` frame of `pose`.
    pub frame: LocalFrame,
}

impl GpuSurfaceView {
    /// Unproject subpixel image coordinates `(u, v)` from `depth_m` into [`Surface`] ENU coordinates.
    #[must_use]
    pub fn unproject_pixels(
        &self,
        surface: &Surface,
        pixels: &[[f64; 2]],
    ) -> Vec<Option<[f64; 3]>> {
        pixels
            .iter()
            .map(|&[u, v]| {
                let d = sample_depth(&self.depth_m, self.camera.width, self.camera.height, u, v)?;
                let world_gpu = self
                    .camera
                    .unproject(&self.pose, Vector2::new(u, v), f64::from(d));
                let [lat, lon, alt] = self.frame.geodetic(world_gpu);
                Some(surface.local(lat, lon, alt))
            })
            .collect()
    }
}

/// Execution queue role for hardware GPU device and queue isolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuQueueRole {
    /// Foreground tracking: real-time pose estimation and view polish.
    ForegroundTracking,
    /// Background tile builder: asynchronous DEM ray-marching and feature extraction.
    BackgroundTileBuilder,
    /// Background anchor worker: asynchronous map anchor acquisition and polish.
    BackgroundAnchor,
    /// Shared execution queue (single-queue fallback).
    Shared,
}

impl GpuQueueRole {
    /// Return the string identifier for device and queue labeling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ForegroundTracking => "foreground-tracking",
            Self::BackgroundTileBuilder => "background-tile-builder",
            Self::BackgroundAnchor => "background-anchor",
            Self::Shared => "shared",
        }
    }

    /// Return true when the role uses an isolated hardware queue.
    #[must_use]
    pub const fn is_isolated(self) -> bool {
        !matches!(self, Self::Shared)
    }
}

/// Resident hardware GPU terrain renderer bridged to [`Surface`] coordinates.
pub struct SurfaceGpuRenderer {
    template: SceneTemplate,
    limits: RenderLimits,
    active_key: CameraKey,
    renderer: GpuRenderer,
    role: GpuQueueRole,
}

impl SurfaceGpuRenderer {
    /// Upload `surface` atlases onto an isolated hardware GPU device and queue.
    #[must_use]
    pub fn try_new_with_role(
        surface: &Surface,
        k_render: &Matrix3<f64>,
        render_size: [u32; 2],
        role: GpuQueueRole,
    ) -> Option<Self> {
        let camera = camera_model(k_render, render_size)?;
        let fail_key = format!(
            "{}:{}:{}:{}",
            surface.catalog_sha256,
            surface.imagery_tiles.len(),
            surface.dem_tiles.len(),
            role.as_str()
        );
        let failed = FAILED_GPU_INIT.get_or_init(|| Mutex::new(HashSet::new()));
        if failed.lock().ok().is_some_and(|s| s.contains(&fail_key)) {
            return None;
        }
        let Some(mut template) = SceneTemplate::from_surface(surface) else {
            record_gpu_init_failure(failed, fail_key, &surface.catalog_sha256, role, "atlas");
            return None;
        };
        let limits = default_limits(&camera);
        let renderer = match pollster::block_on(GpuRenderer::new(
            template.to_render_scene(),
            camera,
            limits,
        )) {
            Ok(r) => r,
            Err(first_err) => {
                if let Some(smaller) = SceneTemplate::from_surface_with_span(surface, 32)
                    && let Ok(r) = pollster::block_on(GpuRenderer::new(
                        smaller.to_render_scene(),
                        camera,
                        limits,
                    ))
                {
                    template = smaller;
                    r
                } else {
                    record_gpu_init_failure(
                        failed,
                        fail_key,
                        &surface.catalog_sha256,
                        role,
                        &first_err.to_string(),
                    );
                    return None;
                }
            }
        };
        Some(Self {
            template,
            limits,
            active_key: CameraKey::from_model(camera),
            renderer,
            role,
        })
    }

    /// Upload `surface` atlases onto a hardware GPU when a compatible adapter is available.
    #[must_use]
    pub fn try_new(
        surface: &Surface,
        k_render: &Matrix3<f64>,
        render_size: [u32; 2],
    ) -> Option<Self> {
        Self::try_new_with_role(surface, k_render, render_size, GpuQueueRole::Shared)
    }

    /// Return the active GPU backend and shader revision label.
    #[must_use]
    pub fn backend_identity(&self) -> &str {
        self.renderer.backend_identity()
    }

    /// Return the assigned GPU queue role.
    #[must_use]
    pub const fn role(&self) -> GpuQueueRole {
        self.role
    }

    /// Return true when the assigned queue role is isolated.
    #[must_use]
    pub const fn is_queue_isolated(&self) -> bool {
        self.role.is_isolated()
    }

    /// Return the queue identity string combining role and backend identity.
    #[must_use]
    pub fn queue_identity(&self) -> String {
        format!(
            "{}:{}",
            self.role.as_str(),
            self.renderer.backend_identity()
        )
    }

    /// Borrow the underlying [`wgpu::Device`].
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        self.renderer.device()
    }

    /// Borrow the underlying [`wgpu::Queue`].
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        self.renderer.queue()
    }

    fn rebind_renderer(
        &mut self,
        camera: CameraModel,
        key: CameraKey,
    ) -> Result<(), PipelineError> {
        let limits = default_limits(&camera);
        let device = self.renderer.device().clone();
        let queue = self.renderer.queue().clone();
        let base_adapter = self
            .renderer
            .backend_identity()
            .strip_prefix("wgpu-terrain-v1/")
            .unwrap_or("gpu");
        let clean_adapter = base_adapter
            .strip_prefix(&format!("{}/", self.role.as_str()))
            .unwrap_or(base_adapter);
        let label = if self.role.is_isolated() {
            format!("{}/{}", self.role.as_str(), clean_adapter)
        } else {
            format!("shared/{}", clean_adapter)
        };
        self.renderer = pollster::block_on(GpuRenderer::with_device(
            self.template.to_render_scene(),
            camera,
            limits,
            device,
            queue,
            label,
        ))?;
        self.limits = limits;
        self.active_key = key;
        Ok(())
    }

    /// Render grayscale imagery and optical depth at `(eye_enu, world_to_cam)` on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when camera validation or GPU execution fails.
    pub fn render_view(
        &mut self,
        surface: &Surface,
        eye_enu: [f64; 3],
        world_to_cam: &Matrix3<f64>,
        k_render: &Matrix3<f64>,
        render_size: [u32; 2],
    ) -> Result<GpuSurfaceView, PipelineError> {
        let camera = camera_model(k_render, render_size)
            .ok_or_else(|| PipelineError::Invalid("invalid GPU camera intrinsics".into()))?;
        let key = CameraKey::from_model(camera);
        let max_tex_dim = self
            .renderer
            .device()
            .limits()
            .max_texture_dimension_2d
            .min(MAX_ATLAS_DIM_PX);
        let fwd_z = world_to_cam[(2, 2)];
        let agl_est = (eye_enu[2] - surface.min_elev_m).clamp(50.0, 4_000.0);
        let look_t = if fwd_z < -0.05 {
            (-agl_est / fwd_z).clamp(0.0, 1_500.0)
        } else {
            500.0
        };
        let focus_enu = [
            eye_enu[0] + world_to_cam[(2, 0)] * look_t,
            eye_enu[1] + world_to_cam[(2, 1)] * look_t,
            eye_enu[2],
        ];
        let [focus_lat, focus_lon, _] = surface.geodetic(focus_enu);
        let subatlas_changed =
            self.template
                .update_subatlas_at(surface, focus_lat, focus_lon, max_tex_dim);
        if key != self.active_key || subatlas_changed {
            self.rebind_renderer(camera, key)?;
        }
        let pose = surface_pose_to_gpu(surface, self.template.frame, eye_enu, world_to_cam)?;
        let view = match self.renderer.render_blocking(pose) {
            Ok(v) => v,
            Err(e) => {
                if let Ok(new_renderer) = pollster::block_on(GpuRenderer::new(
                    self.template.to_render_scene(),
                    camera,
                    self.limits,
                )) {
                    self.renderer = new_renderer;
                    if self.role.is_isolated() {
                        self.rebind_renderer(camera, key)?;
                    }
                    self.renderer.render_blocking(pose)?
                } else {
                    return Err(PipelineError::Render(e));
                }
            }
        };
        Ok(GpuSurfaceView {
            image: view.image,
            depth_m: view.depth_m,
            camera,
            pose,
            frame: self.template.frame,
        })
    }
}

fn record_gpu_init_failure(
    failed: &Mutex<HashSet<String>>,
    fail_key: String,
    catalog_sha256: &str,
    role: GpuQueueRole,
    reason: &str,
) {
    if let Ok(mut guard) = failed.lock()
        && guard.insert(fail_key)
    {
        tracing::warn!(
            catalog_sha256,
            role = role.as_str(),
            reason,
            "GPU terrain renderer unavailable; falling back to adaptive CPU raycasting"
        );
    }
}

#[cfg(test)]
mod tests;
