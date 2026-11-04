//! Resident `wgpu-terrain-v1` GPU scene sharing (`Arc<GpuScene>`) and per-worker viewports.

pub mod atlas;
pub mod camera;

pub use camera::surface_pose_to_gpu;

use super::{Surface, mercator_tile_coords};
use crate::error::FlightVisualError;
use atlas::{
    GPU_TILE_PIXELS, MAX_ATLAS_DIM_PX, MAX_SUBATLAS_TILES, build_image_atlas,
    build_image_atlas_window, build_terrain_atlas, build_terrain_atlas_for_imagery_window,
    sample_depth, subatlas_contains_comfortably, subatlas_window_around, surface_imagery_bounds,
};
use camera::{CameraKey, camera_model, default_limits, manifest_hex64};
use image::GrayImage;
use nalgebra::{Matrix3, Vector2};
use navigate_visual::{CameraModel, CameraPose, LocalFrame, MapRevision, ReferenceRenderer};
use navigate_visual_render::{GpuRenderer, GpuScene, RenderLimits, RenderScene, wgpu};
use std::{
    collections::HashSet,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

static FAILED_GPU_INIT: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

pub(super) fn build_render_scene_for_surface(surface: &Surface) -> Option<RenderScene> {
    build_render_scene_with_span(surface, MAX_SUBATLAS_TILES).map(|(scene, _, _)| scene)
}

fn build_render_scene_with_span(
    surface: &Surface,
    max_tiles: u32,
) -> Option<(RenderScene, [u32; 4], Option<[u32; 4]>)> {
    let frame = LocalFrame::wgs84_enu(surface.origin_lat, surface.origin_lon).ok()?;
    let imagery_bounds = surface_imagery_bounds(surface)?;
    let max_dim = max_tiles
        .saturating_mul(GPU_TILE_PIXELS)
        .min(MAX_ATLAS_DIM_PX);
    let (subatlas_window, imagery, terrain) = if max_tiles >= MAX_SUBATLAS_TILES
        && let (Some(img), Some(ter)) = (build_image_atlas(surface), build_terrain_atlas(surface))
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
    let scene = RenderScene {
        map: MapRevision {
            release_id: "catalog-surface".into(),
            manifest_sha256: manifest_hex64(&surface.catalog_sha256),
        },
        frame,
        imagery,
        terrain,
    };
    Some((scene, imagery_bounds, subatlas_window))
}

struct SharedSceneState {
    scene: Arc<GpuScene>,
    subatlas_window: Option<[u32; 4]>,
    epoch: u64,
}

/// Immutable GPU-resident scene textures shared across foreground and background viewports.
pub struct SharedSurfaceGpu {
    frame: LocalFrame,
    imagery_bounds: [u32; 4],
    state: Mutex<SharedSceneState>,
    epoch: AtomicU64,
}

impl SharedSurfaceGpu {
    /// Upload `surface` imagery and terrain atlases once onto hardware GPU memory.
    #[must_use]
    pub fn try_new(surface: &Surface) -> Option<Arc<Self>> {
        let fail_key = format!(
            "{}:{}:{}",
            surface.catalog_sha256,
            surface.imagery_tiles.len(),
            surface.dem_tiles.len(),
        );
        let failed = FAILED_GPU_INIT.get_or_init(|| Mutex::new(HashSet::new()));
        if failed.lock().ok().is_some_and(|s| s.contains(&fail_key)) {
            return None;
        }
        let k_boot = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
        let cam_boot = camera_model(&k_boot, [800, 600])?;
        let scene_bytes = default_limits(&cam_boot).scene_bytes;

        let Some((render_scene, imagery_bounds, mut subatlas_window)) =
            build_render_scene_with_span(surface, MAX_SUBATLAS_TILES)
        else {
            record_gpu_init_failure(failed, fail_key, &surface.catalog_sha256, "atlas");
            return None;
        };
        let frame = render_scene.frame;
        let gpu_scene = match pollster::block_on(GpuScene::new(render_scene, scene_bytes)) {
            Ok(s) => s,
            Err(first_err) => {
                if let Some((smaller_scene, _, smaller_win)) =
                    build_render_scene_with_span(surface, 32)
                    && let Ok(s) = pollster::block_on(GpuScene::new(smaller_scene, scene_bytes))
                {
                    subatlas_window = smaller_win;
                    s
                } else {
                    record_gpu_init_failure(
                        failed,
                        fail_key,
                        &surface.catalog_sha256,
                        &first_err.to_string(),
                    );
                    return None;
                }
            }
        };
        Some(Arc::new(Self {
            frame,
            imagery_bounds,
            state: Mutex::new(SharedSceneState {
                scene: Arc::new(gpu_scene),
                subatlas_window,
                epoch: 1,
            }),
            epoch: AtomicU64::new(1),
        }))
    }

    /// Return the total GPU texture allocation in bytes for the shared scene atlases.
    #[must_use]
    pub fn resident_texture_bytes(&self) -> usize {
        self.state
            .lock()
            .map_or(0, |g| g.scene.resident_texture_bytes())
    }

    /// Return the validated WGS84 [`LocalFrame`] of the shared scene.
    #[must_use]
    pub const fn frame(&self) -> LocalFrame {
        self.frame
    }

    fn current_scene(&self) -> Option<(Arc<GpuScene>, u64)> {
        let g = self.state.lock().ok()?;
        Some((Arc::clone(&g.scene), g.epoch))
    }

    fn maybe_shift_subatlas(
        &self,
        surface: &Surface,
        lat_deg: f64,
        lon_deg: f64,
        max_tex_dim: u32,
        scene_bytes: usize,
    ) -> Option<(Arc<GpuScene>, u64)> {
        let mut g = self.state.lock().ok()?;
        let cur_win = g.subatlas_window?;
        let (cx, cy, _, _) = mercator_tile_coords(lat_deg, lon_deg, surface.imagery_zoom);
        if subatlas_contains_comfortably(self.imagery_bounds, cur_win, cx, cy) {
            return None;
        }
        let max_tiles = (max_tex_dim / GPU_TILE_PIXELS).clamp(1, MAX_SUBATLAS_TILES);
        let next_win = subatlas_window_around(self.imagery_bounds, cx, cy, max_tiles);
        if next_win == cur_win {
            return None;
        }
        let img = build_image_atlas_window(surface, next_win, max_tex_dim)?;
        let ter = build_terrain_atlas_for_imagery_window(surface, next_win, max_tex_dim)?;
        let next_render_scene = RenderScene {
            map: MapRevision {
                release_id: "catalog-surface".into(),
                manifest_sha256: manifest_hex64(&surface.catalog_sha256),
            },
            frame: self.frame,
            imagery: img,
            terrain: ter,
        };
        let adapter_label = g
            .scene
            .backend_identity()
            .strip_prefix("wgpu-terrain-v1/")
            .unwrap_or_else(|| g.scene.backend_identity())
            .to_owned();
        let uploaded = pollster::block_on(GpuScene::with_device(
            next_render_scene,
            scene_bytes,
            g.scene.device().clone(),
            g.scene.queue().clone(),
            adapter_label,
        ))
        .ok()?;
        g.scene = Arc::new(uploaded);
        g.subatlas_window = Some(next_win);
        g.epoch = g.epoch.wrapping_add(1);
        self.epoch.store(g.epoch, Ordering::Release);
        Some((Arc::clone(&g.scene), g.epoch))
    }
}

/// Grayscale view and optical-axis depth buffer rendered on the GPU for one [`Surface`] pose.
pub struct GpuSurfaceView {
    /// Rendered grayscale orthoimagery view.
    pub image: GrayImage,
    /// Row-major positive optical-axis depth in meters (`0.0` where uncovered or sky).
    pub depth_m: Vec<f32>,
    /// Pinhole camera model of the view.
    pub camera: CameraModel,
    /// Camera pose in `frame`.
    pub pose: CameraPose,
    /// WGS84 `ENU` frame of `pose`.
    pub frame: LocalFrame,
}

impl GpuSurfaceView {
    /// Unproject subpixel image coordinates `(u, v)` from `depth_m` into [`Surface`] `ENU` coordinates.
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

/// Execution queue role for GPU viewport labeling and worker telemetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuQueueRole {
    /// Foreground tracking: real-time pose estimation and view polish.
    ForegroundTracking,
    /// Background tile builder: asynchronous DEM ray-marching and feature extraction.
    BackgroundTileBuilder,
    /// Background anchor worker: asynchronous map anchor acquisition and polish.
    BackgroundAnchor,
    /// Shared execution queue.
    Shared,
}

impl GpuQueueRole {
    /// Return the string identifier for viewport labeling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ForegroundTracking => "foreground-tracking",
            Self::BackgroundTileBuilder => "background-tile-builder",
            Self::BackgroundAnchor => "background-anchor",
            Self::Shared => "shared",
        }
    }

    /// Return true when the role uses a dedicated worker viewport label.
    #[must_use]
    pub const fn is_isolated(self) -> bool {
        !matches!(self, Self::Shared)
    }
}

/// Per-worker GPU viewport bound to a [`SharedSurfaceGpu`] scene.
pub struct SurfaceGpuRenderer {
    shared: Arc<SharedSurfaceGpu>,
    bound_epoch: u64,
    limits: RenderLimits,
    active_key: CameraKey,
    renderer: GpuRenderer,
    role: GpuQueueRole,
}

impl SurfaceGpuRenderer {
    /// Create a lightweight per-worker viewport sharing `shared`'s GPU scene textures.
    #[must_use]
    pub fn from_shared(
        shared: &Arc<SharedSurfaceGpu>,
        k_render: &Matrix3<f64>,
        render_size: [u32; 2],
        role: GpuQueueRole,
    ) -> Option<Self> {
        let camera = camera_model(k_render, render_size)?;
        let limits = default_limits(&camera);
        let (scene, epoch) = shared.current_scene()?;
        let renderer =
            GpuRenderer::from_scene_with_label(scene, camera, limits, role.as_str()).ok()?;
        Some(Self {
            shared: Arc::clone(shared),
            bound_epoch: epoch,
            limits,
            active_key: CameraKey::from_model(camera),
            renderer,
            role,
        })
    }

    /// Upload `surface` atlases and create a viewport with `role`.
    #[must_use]
    pub fn try_new_with_role(
        surface: &Surface,
        k_render: &Matrix3<f64>,
        render_size: [u32; 2],
        role: GpuQueueRole,
    ) -> Option<Self> {
        let shared = SharedSurfaceGpu::try_new(surface)?;
        Self::from_shared(&shared, k_render, render_size, role)
    }

    /// Upload `surface` atlases and create a default shared-role viewport.
    #[must_use]
    pub fn try_new(
        surface: &Surface,
        k_render: &Matrix3<f64>,
        render_size: [u32; 2],
    ) -> Option<Self> {
        Self::try_new_with_role(surface, k_render, render_size, GpuQueueRole::Shared)
    }

    /// Borrow the underlying [`SharedSurfaceGpu`].
    #[must_use]
    pub const fn shared_scene(&self) -> &Arc<SharedSurfaceGpu> {
        &self.shared
    }

    /// Return the GPU scene texture bytes shared by this viewport.
    #[must_use]
    pub fn resident_scene_texture_bytes(&self) -> usize {
        self.shared.resident_texture_bytes()
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

    fn bind_scene(
        &mut self,
        scene: Arc<GpuScene>,
        epoch: u64,
        camera: CameraModel,
        key: CameraKey,
    ) -> Result<(), FlightVisualError> {
        let limits = default_limits(&camera);
        self.renderer =
            GpuRenderer::from_scene_with_label(scene, camera, limits, self.role.as_str())?;
        self.bound_epoch = epoch;
        self.limits = limits;
        self.active_key = key;
        Ok(())
    }

    /// Render grayscale imagery and optical depth at `(eye_enu, world_to_cam)` on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when camera validation or GPU execution fails.
    pub fn render_view(
        &mut self,
        surface: &Surface,
        eye_enu: [f64; 3],
        world_to_cam: &Matrix3<f64>,
        k_render: &Matrix3<f64>,
        render_size: [u32; 2],
    ) -> Result<GpuSurfaceView, FlightVisualError> {
        let camera = camera_model(k_render, render_size)
            .ok_or_else(|| FlightVisualError::Invalid("invalid GPU camera intrinsics".into()))?;
        let key = CameraKey::from_model(camera);
        let limits = default_limits(&camera);
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
        if let Some((new_scene, new_epoch)) = self.shared.maybe_shift_subatlas(
            surface,
            focus_lat,
            focus_lon,
            max_tex_dim,
            limits.scene_bytes,
        ) {
            self.bind_scene(new_scene, new_epoch, camera, key)?;
        } else if self.shared.epoch.load(Ordering::Acquire) != self.bound_epoch
            && let Some((cur_scene, cur_epoch)) = self.shared.current_scene()
        {
            self.bind_scene(cur_scene, cur_epoch, camera, key)?;
        } else if key != self.active_key {
            self.renderer.set_camera(camera, limits)?;
            self.limits = limits;
            self.active_key = key;
        }
        let pose = surface_pose_to_gpu(surface, self.shared.frame, eye_enu, world_to_cam)?;
        let view = self.renderer.render_blocking(pose)?;
        Ok(GpuSurfaceView {
            image: view.image,
            depth_m: view.depth_m,
            camera,
            pose,
            frame: self.shared.frame,
        })
    }
}

fn record_gpu_init_failure(
    failed: &Mutex<HashSet<String>>,
    fail_key: String,
    catalog_sha256: &str,
    reason: &str,
) {
    if let Ok(mut guard) = failed.lock()
        && guard.insert(fail_key)
    {
        tracing::warn!(
            catalog_sha256 = %catalog_sha256,
            reason = %reason,
            "GPU terrain scene upload unavailable; falling back to CPU raycasting"
        );
    }
}
