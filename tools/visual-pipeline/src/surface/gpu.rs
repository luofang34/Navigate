//! Resident `wgpu-terrain-v1` GPU renderer and depth-buffer unprojector for [`Surface`].

use super::Surface;
use crate::PipelineError;
use image::{
    GrayImage,
    imageops::{FilterType, colorops::grayscale, resize},
};
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector2, Vector3};
use navigate_visual::{CameraModel, CameraPose, LocalFrame, MapRevision, ReferenceRenderer};
use navigate_visual_render::{
    AtlasLayout, GpuRenderer, ImageAtlas, RenderLimits, RenderScene, TerrainAtlas,
};
use sha2::{Digest, Sha256};

const GPU_TILE_PIXELS: u32 = 256;
const MAX_SCENE_BYTES: usize = 512 * 1024 * 1024;
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

/// Resident hardware GPU terrain renderer bridged to [`Surface`] coordinates.
pub struct SurfaceGpuRenderer {
    template: SceneTemplate,
    limits: RenderLimits,
    active_key: CameraKey,
    renderer: GpuRenderer,
}

impl SurfaceGpuRenderer {
    /// Upload `surface` atlases onto a hardware GPU when a compatible adapter is available.
    #[must_use]
    pub fn try_new(
        surface: &Surface,
        k_render: &Matrix3<f64>,
        render_size: [u32; 2],
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
        })
    }

    /// Return the active GPU backend and shader revision label.
    #[must_use]
    pub fn backend_identity(&self) -> &str {
        self.renderer.backend_identity()
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
            let label = self
                .renderer
                .backend_identity()
                .strip_prefix("wgpu-terrain-v1/")
                .unwrap_or("shared")
                .to_owned();
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
        let view = self.renderer.render_blocking(pose)?;
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

fn build_image_atlas(surface: &Surface) -> Option<ImageAtlas> {
    if surface.imagery_tiles.is_empty() {
        return None;
    }
    let (mut min_x, mut max_x) = (u32::MAX, 0_u32);
    let (mut min_y, mut max_y) = (u32::MAX, 0_u32);
    for &(tx, ty) in surface.imagery_tiles.keys() {
        min_x = min_x.min(tx);
        max_x = max_x.max(tx);
        min_y = min_y.min(ty);
        max_y = max_y.max(ty);
    }
    let cols = max_x.checked_sub(min_x)?.checked_add(1)?;
    let rows = max_y.checked_sub(min_y)?.checked_add(1)?;
    let width = cols.checked_mul(GPU_TILE_PIXELS)?;
    let height = rows.checked_mul(GPU_TILE_PIXELS)?;
    let total = (width as usize).checked_mul(height as usize)?;
    if total.saturating_mul(2) > MAX_SCENE_BYTES {
        return None;
    }
    let mut pixels = vec![0_u8; total];
    let mut coverage = vec![0_u8; total];
    let tp = GPU_TILE_PIXELS as usize;
    let stride = width as usize;

    for (&(tx, ty), rgb) in &surface.imagery_tiles {
        let gray = grayscale(rgb);
        let tile = if gray.dimensions() == (GPU_TILE_PIXELS, GPU_TILE_PIXELS) {
            gray
        } else {
            resize(
                &gray,
                GPU_TILE_PIXELS,
                GPU_TILE_PIXELS,
                FilterType::Triangle,
            )
        };
        let ox = ((tx - min_x) as usize) * tp;
        let oy = ((ty - min_y) as usize) * tp;
        let raw = tile.as_raw();
        for r in 0..tp {
            let dst = (oy + r) * stride + ox;
            pixels[dst..dst + tp].copy_from_slice(&raw[r * tp..(r + 1) * tp]);
            coverage[dst..dst + tp].fill(255);
        }
    }
    Some(ImageAtlas {
        layout: AtlasLayout {
            zoom: surface.imagery_zoom,
            origin_tile: [min_x, min_y],
            tile_pixels: GPU_TILE_PIXELS,
            size: [width, height],
        },
        pixels,
        coverage,
    })
}

fn build_terrain_atlas(surface: &Surface) -> Option<TerrainAtlas> {
    if surface.dem_tiles.is_empty() {
        return None;
    }
    let (mut min_x, mut max_x) = (u32::MAX, 0_u32);
    let (mut min_y, mut max_y) = (u32::MAX, 0_u32);
    for &(tx, ty) in surface.dem_tiles.keys() {
        min_x = min_x.min(tx);
        max_x = max_x.max(tx);
        min_y = min_y.min(ty);
        max_y = max_y.max(ty);
    }
    let cols = max_x.checked_sub(min_x)?.checked_add(1)?;
    let rows = max_y.checked_sub(min_y)?.checked_add(1)?;
    let width = cols.checked_mul(GPU_TILE_PIXELS)?;
    let height = rows.checked_mul(GPU_TILE_PIXELS)?;
    let total = (width as usize).checked_mul(height as usize)?;
    if total.saturating_mul(4) > MAX_SCENE_BYTES {
        return None;
    }
    let mut heights_m = vec![0.0_f32; total];
    let mut coverage = vec![0_u8; total];
    let tp = GPU_TILE_PIXELS as usize;
    let stride = width as usize;
    let src_s = surface.dem_tile_size.max(1) as usize;

    for (&(tx, ty), elev) in &surface.dem_tiles {
        if elev.len() != src_s * src_s {
            continue;
        }
        let resampled = resample_dem_tile(elev, src_s, tp);
        let ox = ((tx - min_x) as usize) * tp;
        let oy = ((ty - min_y) as usize) * tp;
        for r in 0..tp {
            let dst = (oy + r) * stride + ox;
            heights_m[dst..dst + tp].copy_from_slice(&resampled[r * tp..(r + 1) * tp]);
            coverage[dst..dst + tp].fill(255);
        }
    }
    Some(TerrainAtlas {
        layout: AtlasLayout {
            zoom: surface.dem_zoom,
            origin_tile: [min_x, min_y],
            tile_pixels: GPU_TILE_PIXELS,
            size: [width, height],
        },
        heights_m,
        coverage,
    })
}

fn resample_dem_tile(src: &[f32], src_s: usize, dst_s: usize) -> Vec<f32> {
    let clean = |v: f32| {
        if v.is_finite() {
            v.clamp(-99_000.0, 99_000.0)
        } else {
            0.0
        }
    };
    if src_s == dst_s {
        return src.iter().copied().map(clean).collect();
    }
    let mut out = vec![0.0_f32; dst_s * dst_s];
    for y in 0..dst_s {
        let sy = ((y as f64 + 0.5) * (src_s as f64) / (dst_s as f64) - 0.5)
            .clamp(0.0, (src_s - 1) as f64);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(src_s - 1);
        let fy = (sy - (y0 as f64)) as f32;
        for x in 0..dst_s {
            let sx = ((x as f64 + 0.5) * (src_s as f64) / (dst_s as f64) - 0.5)
                .clamp(0.0, (src_s - 1) as f64);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(src_s - 1);
            let fx = (sx - (x0 as f64)) as f32;
            let h00 = clean(src[y0 * src_s + x0]);
            let h10 = clean(src[y0 * src_s + x1]);
            let h01 = clean(src[y1 * src_s + x0]);
            let h11 = clean(src[y1 * src_s + x1]);
            out[y * dst_s + x] = h00 * (1.0 - fx) * (1.0 - fy)
                + h10 * fx * (1.0 - fy)
                + h01 * (1.0 - fx) * fy
                + h11 * fx * fy;
        }
    }
    out
}

fn sample_depth(depth_m: &[f32], width: u32, height: u32, u: f64, v: f64) -> Option<f32> {
    if !u.is_finite() || !v.is_finite() || width == 0 || height == 0 {
        return None;
    }
    let (w, h) = (width as usize, height as usize);
    if u < 0.0 || v < 0.0 || u > f64::from(width - 1) || v > f64::from(height - 1) {
        return None;
    }
    let x0 = (u.floor() as usize).min(w - 1);
    let y0 = (v.floor() as usize).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let d00 = depth_m[y0 * w + x0];
    let d10 = depth_m[y0 * w + x1];
    let d01 = depth_m[y1 * w + x0];
    let d11 = depth_m[y1 * w + x1];
    if d00 > 0.5 && d10 > 0.5 && d01 > 0.5 && d11 > 0.5 {
        let d_min = d00.min(d10).min(d01).min(d11);
        let d_max = d00.max(d10).max(d01).max(d11);
        if (d_max - d_min) <= 0.025 * d_max {
            let fx = (u - (x0 as f64)) as f32;
            let fy = (v - (y0 as f64)) as f32;
            return Some(
                d00 * (1.0 - fx) * (1.0 - fy)
                    + d10 * fx * (1.0 - fy)
                    + d01 * (1.0 - fx) * fy
                    + d11 * fx * fy,
            );
        }
    }
    let xn = (u.round() as usize).min(w - 1);
    let yn = (v.round() as usize).min(h - 1);
    let dn = depth_m[yn * w + xn];
    (dn > 0.5).then_some(dn)
}

#[cfg(test)]
mod tests;
