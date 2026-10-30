//! CPU sparse keypoint ray-casting and gradient-adaptive terrain rendering.

use std::fmt;

use image::GrayImage;
use nalgebra::{Rotation3, Vector3};
use navigate_visual::{CameraModel, CameraPose, LocalFrame, ReferenceView};

use crate::{AtlasLayout, ImageAtlas, RenderError, RenderScene, TerrainAtlas};

/// Minimum optical-axis ray-marching depth in metres.
const NEAR_DEPTH_M: f64 = 10.0;

/// Maximum pixel count accepted by [`SparseSceneSampler::render_adaptive`].
const MAX_ADAPTIVE_PIXELS: usize = 67_108_864;

/// Terrain intersection and sampled luma for one camera ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseRayHit {
    /// Hit coordinates `[east_m, north_m, up_m]` in the scene local frame.
    pub world_local: [f64; 3],
    /// Positive optical-axis depth in metres.
    pub depth_m: f32,
    /// Bilinearly sampled grayscale value at the hit point.
    pub luma: u8,
}

/// Ray-march work counts from [`SparseSceneSampler::render_adaptive`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveRenderStats {
    /// Total pixels in the rendered view (`width * height`).
    pub total_pixels: usize,
    /// Exact terrain rays marched to produce the view.
    pub rays_cast: usize,
}

/// CPU sparse keypoint ray-caster and adaptive terrain renderer over a [`RenderScene`].
#[derive(Clone, Copy)]
pub struct SparseSceneSampler<'a> {
    scene: &'a RenderScene,
    far_m: f64,
    steps: u32,
    refinements: u32,
}

impl fmt::Debug for SparseSceneSampler<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseSceneSampler")
            .field("frame", &self.scene.frame)
            .field("imagery", &self.scene.imagery.layout)
            .field("terrain", &self.scene.terrain.layout)
            .field("far_m", &self.far_m)
            .field("steps", &self.steps)
            .field("refinements", &self.refinements)
            .finish()
    }
}

impl<'a> SparseSceneSampler<'a> {
    /// Create a CPU sampler for `scene` with bounded ray-marching parameters.
    ///
    /// # Errors
    /// Returns [`RenderError`] when the frame, map revision, atlas data, `far_m`, or `steps` are invalid.
    pub fn new(
        scene: &'a RenderScene,
        far_m: f64,
        steps: u32,
        refinements: u32,
    ) -> Result<Self, RenderError> {
        let [lat, lon] = scene.frame.anchor();
        scene.frame.local([lat, lon, 0.0])?;
        scene.imagery.validate()?;
        scene.terrain.validate()?;
        if scene.map.release_id.is_empty()
            || scene.map.manifest_sha256.len() != 64
            || !scene
                .map
                .manifest_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Err(RenderError::Invalid("scene map identity"));
        }
        if !far_m.is_finite() || far_m <= 0.0 || steps == 0 {
            return Err(RenderError::Invalid("sparse ray-march parameters"));
        }
        Ok(Self {
            scene,
            far_m,
            steps,
            refinements,
        })
    }

    /// Cast terrain rays only for the requested subpixel coordinates `(u, v)`.
    ///
    /// # Errors
    /// Returns [`RenderError`] when `camera`, `pose`, or any pixel coordinate is invalid.
    pub fn raycast_pixels(
        &self,
        camera: &CameraModel,
        pose: &CameraPose,
        pixels: &[[f64; 2]],
    ) -> Result<Vec<Option<SparseRayHit>>, RenderError> {
        camera.validate()?;
        pose.validate()?;
        if pixels.iter().any(|[u, v]| !u.is_finite() || !v.is_finite()) {
            return Err(RenderError::Invalid("pixel coordinates must be finite"));
        }
        let ctx = RayCtx {
            camera,
            eye: pose.position,
            rot: pose.orientation.to_rotation_matrix(),
        };
        let mut hits = Vec::with_capacity(pixels.len());
        for &[u, v] in pixels {
            hits.push(self.raycast_one(&ctx, u, v));
        }
        Ok(hits)
    }

    /// Render a terrain reference view with coarse-to-fine block interpolation and edge refinement.
    ///
    /// # Errors
    /// Returns [`RenderError`] when `camera`, `pose`, or `depth_rel_threshold` is invalid.
    pub fn render_adaptive(
        &self,
        camera: &CameraModel,
        pose: &CameraPose,
        stride: u32,
        luma_grad_threshold: u8,
        depth_rel_threshold: f32,
    ) -> Result<(ReferenceView, AdaptiveRenderStats), RenderError> {
        camera.validate()?;
        pose.validate()?;
        if !depth_rel_threshold.is_finite() || depth_rel_threshold < 0.0 {
            return Err(RenderError::Invalid(
                "depth relative threshold must be finite and non-negative",
            ));
        }
        let total_pixels = (camera.width as usize)
            .checked_mul(camera.height as usize)
            .filter(|&n| n <= MAX_ADAPTIVE_PIXELS)
            .ok_or(RenderError::Invalid("adaptive view exceeds pixel limit"))?;

        let ctx = RayCtx {
            camera,
            eye: pose.position,
            rot: pose.orientation.to_rotation_matrix(),
        };
        let width = camera.width as usize;
        let mut luma_buf = vec![0_u8; total_pixels];
        let mut depth_buf = vec![0.0_f32; total_pixels];
        let mut rays_cast = 0_usize;

        if stride <= 1 {
            rays_cast = total_pixels;
            for y in 0..camera.height {
                for x in 0..camera.width {
                    if let Some(hit) = self.raycast_one(&ctx, f64::from(x), f64::from(y)) {
                        let idx = (y as usize) * width + (x as usize);
                        luma_buf[idx] = hit.luma;
                        depth_buf[idx] = hit.depth_m;
                    }
                }
            }
        } else {
            let xs = grid_axis(camera.width, stride);
            let ys = grid_axis(camera.height, stride);
            let nx = xs.len();
            let mut state = vec![0_u8; total_pixels];
            let mut grid_hits = Vec::with_capacity(nx * ys.len());

            for &gy in &ys {
                for &gx in &xs {
                    let hit = self.raycast_one(&ctx, f64::from(gx), f64::from(gy));
                    let idx = (gy as usize) * width + (gx as usize);
                    state[idx] = 2;
                    rays_cast += 1;
                    if let Some(h) = hit {
                        luma_buf[idx] = h.luma;
                        depth_buf[idx] = h.depth_m;
                    }
                    grid_hits.push(hit);
                }
            }

            for (by, y_pair) in ys.windows(2).enumerate() {
                let [y0, y1] = [y_pair[0], y_pair[1]];
                for (bx, x_pair) in xs.windows(2).enumerate() {
                    let [x0, x1] = [x_pair[0], x_pair[1]];
                    let corners = [
                        grid_hits[by * nx + bx],
                        grid_hits[by * nx + bx + 1],
                        grid_hits[(by + 1) * nx + bx],
                        grid_hits[(by + 1) * nx + bx + 1],
                    ];
                    if let Some(c) =
                        smooth_corners(corners, luma_grad_threshold, depth_rel_threshold)
                    {
                        interpolate_block(
                            &c,
                            [x0, x1],
                            [y0, y1],
                            width,
                            &mut state,
                            &mut luma_buf,
                            &mut depth_buf,
                        );
                    } else {
                        for y in y0..=y1 {
                            for x in x0..=x1 {
                                let idx = (y as usize) * width + (x as usize);
                                if state[idx] == 2 {
                                    continue;
                                }
                                state[idx] = 2;
                                rays_cast += 1;
                                if let Some(hit) =
                                    self.raycast_one(&ctx, f64::from(x), f64::from(y))
                                {
                                    luma_buf[idx] = hit.luma;
                                    depth_buf[idx] = hit.depth_m;
                                } else {
                                    luma_buf[idx] = 0;
                                    depth_buf[idx] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        let image = GrayImage::from_raw(camera.width, camera.height, luma_buf)
            .ok_or(RenderError::Invalid("adaptive output dimensions"))?;
        Ok((
            ReferenceView {
                map: self.scene.map.clone(),
                frame: self.scene.frame,
                pose: *pose,
                image,
                depth_m: depth_buf,
            },
            AdaptiveRenderStats {
                total_pixels,
                rays_cast,
            },
        ))
    }

    fn raycast_one(&self, ctx: &RayCtx<'_>, u: f64, v: f64) -> Option<SparseRayHit> {
        if self.far_m <= NEAR_DEPTH_M {
            return None;
        }
        let ray_eye = Vector3::new(
            (u - ctx.camera.cx) / ctx.camera.fx,
            (ctx.camera.cy - v) / ctx.camera.fy,
            -1.0,
        );
        let ray = ctx.rot * ray_eye;
        let eye = ctx.eye;
        let ratio = self.far_m / NEAR_DEPTH_M;
        let inv_steps = 1.0 / f64::from(self.steps);

        let mut above = false;
        let mut previous = NEAR_DEPTH_M;
        let mut bracket = None;
        for step in 0..=self.steps {
            let distance = NEAR_DEPTH_M * ratio.powf(f64::from(step) * inv_steps);
            let point = eye + ray * distance;
            let (mercator_xy, alt_m) = point_mercator_and_alt(self.scene.frame, point);
            if let Some(terrain_h) = sample_terrain(&self.scene.terrain, mercator_xy) {
                let clearance = alt_m - terrain_h;
                if clearance <= 0.0 && above {
                    bracket = Some((previous, distance));
                    break;
                }
                above = clearance > 0.0;
            } else {
                above = false;
            }
            previous = distance;
        }

        let (mut low, mut high) = bracket?;
        for _ in 0..self.refinements {
            let middle = 0.5 * (low + high);
            let point = eye + ray * middle;
            let (mercator_xy, alt_m) = point_mercator_and_alt(self.scene.frame, point);
            if let Some(terrain_h) = sample_terrain(&self.scene.terrain, mercator_xy)
                && alt_m - terrain_h <= 0.0
            {
                high = middle;
            } else {
                low = middle;
            }
        }

        let distance = 0.5 * (low + high);
        let hit = eye + ray * distance;
        let (mercator_xy, _) = point_mercator_and_alt(self.scene.frame, hit);
        let luma = sample_imagery(&self.scene.imagery, mercator_xy)?;
        Some(SparseRayHit {
            world_local: [hit.x, hit.y, hit.z],
            depth_m: distance as f32,
            luma,
        })
    }
}

struct RayCtx<'a> {
    camera: &'a CameraModel,
    eye: Vector3<f64>,
    rot: Rotation3<f64>,
}

fn grid_axis(len: u32, stride: u32) -> Vec<u32> {
    let mut coords: Vec<u32> = (0..len).step_by(stride as usize).collect();
    if coords.last().copied() != Some(len - 1) {
        coords.push(len - 1);
    }
    coords
}

fn smooth_corners(
    corners: [Option<SparseRayHit>; 4],
    luma_grad_threshold: u8,
    depth_rel_threshold: f32,
) -> Option<[SparseRayHit; 4]> {
    let hit = [corners[0]?, corners[1]?, corners[2]?, corners[3]?];
    let mut l_min = u8::MAX;
    let mut l_max = u8::MIN;
    let mut d_min = f32::INFINITY;
    let mut d_max = 0.0_f32;
    for h in hit {
        if h.depth_m <= 0.0 {
            return None;
        }
        l_min = l_min.min(h.luma);
        l_max = l_max.max(h.luma);
        d_min = d_min.min(h.depth_m);
        d_max = d_max.max(h.depth_m);
    }
    if l_max - l_min > luma_grad_threshold || (d_max - d_min) > depth_rel_threshold * d_min {
        return None;
    }
    Some(hit)
}

fn interpolate_block(
    c: &[SparseRayHit; 4],
    [x0, x1]: [u32; 2],
    [y0, y1]: [u32; 2],
    width: usize,
    state: &mut [u8],
    luma_buf: &mut [u8],
    depth_buf: &mut [f32],
) {
    let dx = f64::from(x1 - x0).max(1.0);
    let dy = f64::from(y1 - y0).max(1.0);
    let inv_d = c.map(|h| 1.0 / f64::from(h.depth_m));
    let lumas = c.map(|h| f64::from(h.luma));
    for y in y0..=y1 {
        let ty = f64::from(y - y0) / dy;
        for x in x0..=x1 {
            let idx = (y as usize) * width + (x as usize);
            if state[idx] != 0 {
                continue;
            }
            let tx = f64::from(x - x0) / dx;
            let l_top = lumas[0] * (1.0 - tx) + lumas[1] * tx;
            let l_bot = lumas[2] * (1.0 - tx) + lumas[3] * tx;
            let id_top = inv_d[0] * (1.0 - tx) + inv_d[1] * tx;
            let id_bot = inv_d[2] * (1.0 - tx) + inv_d[3] * tx;
            state[idx] = 1;
            luma_buf[idx] = (l_top * (1.0 - ty) + l_bot * ty).round().clamp(0.0, 255.0) as u8;
            depth_buf[idx] = (1.0 / (id_top * (1.0 - ty) + id_bot * ty)) as f32;
        }
    }
}

fn point_mercator_and_alt(frame: LocalFrame, point: Vector3<f64>) -> ([f64; 2], f64) {
    if matches!(frame, LocalFrame::AnchorMercator { .. }) {
        (frame.mercator_xy(point), point.z)
    } else {
        let [lat_deg, lon_deg, alt_m] = frame.geodetic(point);
        let mx = (lon_deg + 180.0) / 360.0;
        let my = (1.0 - lat_deg.to_radians().tan().asinh() / std::f64::consts::PI) * 0.5;
        ([mx, my], alt_m)
    }
}

fn sample_bilinear_covered(
    layout: AtlasLayout,
    coverage: &[u8],
    mercator_xy: [f64; 2],
    value_at: impl Fn(usize) -> f64,
) -> Option<f64> {
    let tiles = f64::from(1_u32 << layout.zoom);
    let tp = f64::from(layout.tile_pixels);
    let gx = (mercator_xy[0] * tiles - f64::from(layout.origin_tile[0])) * tp - 0.5;
    let gy = (mercator_xy[1] * tiles - f64::from(layout.origin_tile[1])) * tp - 0.5;
    let [width, height] = layout.size;
    if !gx.is_finite()
        || !gy.is_finite()
        || gx < 0.0
        || gy < 0.0
        || gx > f64::from(width - 1)
        || gy > f64::from(height - 1)
    {
        return None;
    }
    let w = width as usize;
    let (x0, y0) = (gx.floor() as usize, gy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(height as usize - 1));
    let (fx, fy) = (gx - gx.floor(), gy - gy.floor());
    let load = |x: usize, y: usize| -> (f64, f64) {
        let idx = y * w + x;
        if coverage.get(idx).copied() == Some(255) {
            (value_at(idx), 1.0)
        } else {
            (0.0, 0.0)
        }
    };
    let ((v00, c00), (v10, c10)) = (load(x0, y0), load(x1, y0));
    let ((v01, c01), (v11, c11)) = (load(x0, y1), load(x1, y1));
    let cov = (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + (c01 * (1.0 - fx) + c11 * fx) * fy;
    (cov > 0.9999)
        .then_some((v00 * (1.0 - fx) + v10 * fx) * (1.0 - fy) + (v01 * (1.0 - fx) + v11 * fx) * fy)
}

fn sample_terrain(atlas: &TerrainAtlas, mercator_xy: [f64; 2]) -> Option<f64> {
    sample_bilinear_covered(atlas.layout, &atlas.coverage, mercator_xy, |idx| {
        f64::from(atlas.heights_m.get(idx).copied().unwrap_or(0.0))
    })
}

fn sample_imagery(atlas: &ImageAtlas, mercator_xy: [f64; 2]) -> Option<u8> {
    let val = sample_bilinear_covered(atlas.layout, &atlas.coverage, mercator_xy, |idx| {
        f64::from(atlas.pixels.get(idx).copied().unwrap_or(0))
    })?;
    Some(val.round().clamp(0.0, 255.0) as u8)
}

#[cfg(test)]
mod tests;
