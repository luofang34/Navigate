//! Resident `wgpu-terrain-v1` GPU renderer and depth-buffer unprojector for [`Surface`].

mod atlas;

use super::Surface;
use crate::PipelineError;
use atlas::{MAX_SCENE_BYTES, build_image_atlas, build_terrain_atlas, sample_depth};
use image::GrayImage;
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector2, Vector3};
use navigate_visual::{CameraModel, CameraPose, LocalFrame, MapRevision, ReferenceRenderer};
use navigate_visual_render::{
    AtlasLayout, GpuRenderer, ImageAtlas, RenderLimits, RenderScene, TerrainAtlas, wgpu,
};
use sha2::{Digest, Sha256};

const MAX_VIEW_PIXELS: usize = 4_194_304;

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
struct SceneTemplate {
    map: MapRevision,
    frame: LocalFrame,
    imagery_layout: AtlasLayout,
    imagery_pixels: Vec<u8>,
    imagery_coverage: Vec<u8>,
    terrain_layout: AtlasLayout,
    terrain_heights: Vec<f32>,
    terrain_coverage: Vec<u8>,
}

impl SceneTemplate {
    fn from_surface(surface: &Surface) -> Option<Self> {
        let frame = LocalFrame::spherical_enu(surface.origin_lat, surface.origin_lon).ok()?;
        let imagery = build_image_atlas(surface)?;
        let terrain = build_terrain_atlas(surface)?;
        Some(Self {
            map: MapRevision {
                release_id: "catalog-surface".into(),
                manifest_sha256: manifest_hex64(&surface.catalog_sha256),
            },
            frame,
            imagery_layout: imagery.layout,
            imagery_pixels: imagery.pixels,
            imagery_coverage: imagery.coverage,
            terrain_layout: terrain.layout,
            terrain_heights: terrain.heights_m,
            terrain_coverage: terrain.coverage,
        })
    }

    fn to_render_scene(&self) -> RenderScene {
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
    /// Spherical ENU frame of `pose`.
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
        let template = SceneTemplate::from_surface(surface)?;
        let camera = camera_model(k_render, render_size)?;
        let limits = default_limits(&camera);
        let renderer =
            pollster::block_on(GpuRenderer::new(template.to_render_scene(), camera, limits))
                .ok()?;
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
        if key != self.active_key {
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
        }
        let pose = surface_pose_to_gpu(surface, self.template.frame, eye_enu, world_to_cam)?;
        let view = match self.renderer.render_blocking(pose) {
            Ok(v) => v,
            Err(e) => {
                if let Ok(mut new_renderer) = pollster::block_on(GpuRenderer::new(
                    self.template.to_render_scene(),
                    camera,
                    self.limits,
                )) {
                    if self.role.is_isolated() {
                        let dev = new_renderer.device().clone();
                        let q = new_renderer.queue().clone();
                        let base_adapter = new_renderer
                            .backend_identity()
                            .strip_prefix("wgpu-terrain-v1/")
                            .unwrap_or("gpu");
                        let label = format!("{}/{}", self.role.as_str(), base_adapter);
                        if let Ok(reconfigured) = pollster::block_on(GpuRenderer::with_device(
                            self.template.to_render_scene(),
                            camera,
                            self.limits,
                            dev,
                            q,
                            label,
                        )) {
                            new_renderer = reconfigured;
                        }
                    }
                    self.renderer = new_renderer;
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

pub(crate) fn surface_pose_to_gpu(
    surface: &Surface,
    frame: LocalFrame,
    eye_enu: [f64; 3],
    world_to_cam: &Matrix3<f64>,
) -> Result<CameraPose, PipelineError> {
    let [lat, lon, alt] = surface.geodetic(eye_enu);
    let position = frame.local([lat, lon, alt])?;
    let gl_to_cv = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
    let raw_eye_to_world = world_to_cam.transpose() * gl_to_cv;
    let ortho = orthonormalize_so3(&raw_eye_to_world);
    let orientation =
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(ortho));
    let pose = CameraPose {
        position,
        orientation,
    };
    pose.validate()?;
    Ok(pose)
}

fn orthonormalize_so3(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
        let mut r = u * v_t;
        if r.determinant() < 0.0 {
            let mut u_fix = u;
            u_fix.set_column(2, &(-u.column(2)));
            r = u_fix * v_t;
        }
        r
    } else {
        Matrix3::identity()
    }
}

fn camera_model(k: &Matrix3<f64>, size: [u32; 2]) -> Option<CameraModel> {
    let model = CameraModel {
        width: size[0],
        height: size[1],
        fx: k[(0, 0)],
        fy: k[(1, 1)],
        cx: k[(0, 2)],
        cy: k[(1, 2)],
    };
    model.validate().ok()?;
    Some(model)
}

fn default_limits(camera: &CameraModel) -> RenderLimits {
    let corner_norm = [
        (0.0 - camera.cx) / camera.fx,
        (f64::from(camera.width - 1) - camera.cx) / camera.fx,
    ]
    .into_iter()
    .fold(0.0_f64, |acc, dx| {
        let dy0 = (camera.cy - 0.0) / camera.fy;
        let dy1 = (camera.cy - f64::from(camera.height - 1)) / camera.fy;
        acc.max((dx * dx + dy0 * dy0 + 1.0).sqrt())
            .max((dx * dx + dy1 * dy1 + 1.0).sqrt())
    })
    .max(1.0);
    let max_far = (195_000.0 / corner_norm).clamp(1_000.0, 65_000.0) as f32;
    RenderLimits {
        scene_bytes: MAX_SCENE_BYTES,
        pixels: MAX_VIEW_PIXELS,
        far_m: max_far,
        steps: 144,
        refinements: 12,
    }
}

fn manifest_hex64(raw: &str) -> String {
    if raw.len() == 64 && raw.bytes().all(|b| b.is_ascii_hexdigit()) {
        raw.to_ascii_lowercase()
    } else {
        let digest = Sha256::digest(raw.as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests;
