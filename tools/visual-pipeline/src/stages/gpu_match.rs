//! GPU terrain rendering and correspondence helpers for offline photo stages.

use super::mask_allows;
use crate::{
    PipelineError,
    matcher_client::PipelineMatcher,
    surface::{ContextSurface, GpuSurfaceView, Surface, SurfaceGpuRenderer},
};
use image::{GrayImage, RgbImage};
use nalgebra::Matrix3;

fn boot_gpu(surface: &Surface) -> Option<SurfaceGpuRenderer> {
    let k_boot = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    SurfaceGpuRenderer::try_new(surface, &k_boot, [800, 600])
}

/// Per-view terrain correspondence query for `match` and `refine` stages.
pub(super) struct TerrainMatchQuery<'a> {
    pub eye: [f64; 3],
    pub rot: &'a Matrix3<f64>,
    pub k_render: &'a Matrix3<f64>,
    pub render_size: [u32; 2],
    pub qry_small_rgb: &'a RgbImage,
    pub qry_small_gray: &'a GrayImage,
    pub valid_mask: &'a GrayImage,
    pub full_size: [u32; 2],
    pub q_scale: [f64; 2],
}

/// Cached depth buffer for neighbor-photo 3D unprojection in `run_track_stage`.
pub(super) enum AnchorDepthBuffer {
    Gpu(GpuSurfaceView),
    Cpu {
        depth: Vec<f32>,
        pts: Vec<[f64; 3]>,
        size: [u32; 2],
    },
}

impl AnchorDepthBuffer {
    /// Unproject reference pixel `(u, v)` into `surface` ENU coordinates.
    pub(super) fn unproject(&self, surface: &Surface, u: f64, v: f64) -> Option<[f64; 3]> {
        match self {
            Self::Gpu(gv) => {
                let max_x = f64::from(gv.camera.width.saturating_sub(1));
                let max_y = f64::from(gv.camera.height.saturating_sub(1));
                gv.unproject_pixels(surface, &[[u.clamp(0.0, max_x), v.clamp(0.0, max_y)]])
                    .into_iter()
                    .next()
                    .flatten()
            }
            Self::Cpu { depth, pts, size } => {
                let rx = (u.round() as i64).clamp(0, i64::from(size[0]) - 1) as u32;
                let ry = (v.round() as i64).clamp(0, i64::from(size[1]) - 1) as u32;
                let idx = (ry * size[0] + rx) as usize;
                (depth[idx] > 0.5).then_some(pts[idx])
            }
        }
    }
}

/// Resident GPU terrain renderer with automatic CPU raycasting fallback.
pub(super) struct StageTerrainRenderer {
    primary_gpu: Option<SurfaceGpuRenderer>,
    context_gpu: Option<SurfaceGpuRenderer>,
}

impl StageTerrainRenderer {
    /// Build a stage terrain renderer for `surface` (`GPU` when available, `CPU` fallback otherwise).
    pub(super) fn new(surface: &ContextSurface) -> Self {
        let mut primary_gpu = boot_gpu(&surface.primary);
        let context_gpu = if let Some(ctx) = &surface.context {
            let cg = primary_gpu.is_some().then(|| boot_gpu(ctx)).flatten();
            if cg.is_none() {
                primary_gpu = None;
            }
            cg
        } else {
            None
        };
        Self {
            primary_gpu,
            context_gpu,
        }
    }

    /// Render a terrain depth buffer for neighbor-photo tracking at `(eye, rot)`.
    pub(super) fn render_anchor_depth(
        &mut self,
        surface: &ContextSurface,
        eye: [f64; 3],
        rot: &Matrix3<f64>,
        k_render: &Matrix3<f64>,
        render_size: [u32; 2],
    ) -> AnchorDepthBuffer {
        if let Some(pri_gpu) = &mut self.primary_gpu
            && let Ok(gv) = pri_gpu.render_view(&surface.primary, eye, rot, k_render, render_size)
        {
            return AnchorDepthBuffer::Gpu(gv);
        }
        let (_, depth, pts) = surface.render(eye, rot, k_render, render_size);
        AnchorDepthBuffer::Cpu {
            depth,
            pts,
            size: render_size,
        }
    }
}

pub(super) type MatchedView3d = (usize, Vec<[f64; 3]>, Vec<[f64; 2]>);

impl StageTerrainRenderer {
    /// Render synthetic terrain at `(q.eye, q.rot)`, match against query, and unproject 3D points.
    pub(super) fn match_view(
        &mut self,
        surface: &ContextSurface,
        q: &TerrainMatchQuery<'_>,
        matcher: &mut PipelineMatcher,
    ) -> Result<MatchedView3d, PipelineError> {
        if let Some(pri_gpu) = &mut self.primary_gpu
            && let Ok(pri_view) =
                pri_gpu.render_view(&surface.primary, q.eye, q.rot, q.k_render, q.render_size)
        {
            let ctx_view = if let (Some(ctx), Some(c_gpu)) =
                (&surface.context, &mut self.context_gpu)
                && pri_view.depth_m.iter().any(|&d| d <= 0.5)
            {
                let eye_geo = surface.primary.geodetic(q.eye);
                let ctx_eye = ctx.local(eye_geo[0], eye_geo[1], eye_geo[2]);
                let r_ctx_to_pri = surface.primary.enu_from_ecef * ctx.enu_from_ecef.transpose();
                let w2c_ctx = q.rot * r_ctx_to_pri;
                c_gpu
                    .render_view(ctx, ctx_eye, &w2c_ctx, q.k_render, q.render_size)
                    .ok()
            } else {
                None
            };

            let raw_matches = if let Some(ref cv) = ctx_view {
                let mut comp = pri_view.image.clone();
                let comp_raw = comp.as_mut();
                let cv_raw = cv.image.as_raw();
                for (idx, (&pd, &cd)) in pri_view.depth_m.iter().zip(&cv.depth_m).enumerate() {
                    if pd <= 0.5 && cd > 0.5 {
                        comp_raw[idx] = cv_raw[idx];
                    }
                }
                matcher.match_gray(&comp, q.qry_small_gray)?
            } else {
                matcher.match_gray(&pri_view.image, q.qry_small_gray)?
            };

            let raw_count = raw_matches.len();
            let max_x = f64::from(q.render_size[0].saturating_sub(1));
            let max_y = f64::from(q.render_size[1].saturating_sub(1));
            let mut ref_px = Vec::with_capacity(raw_count);
            let mut qry_px = Vec::with_capacity(raw_count);
            for m in &raw_matches {
                let qx_full = m.query.x / q.q_scale[0];
                let qy_full = m.query.y / q.q_scale[1];
                if !mask_allows(q.valid_mask, q.full_size, qx_full, qy_full) {
                    continue;
                }
                ref_px.push([
                    m.reference.x.clamp(0.0, max_x),
                    m.reference.y.clamp(0.0, max_y),
                ]);
                qry_px.push([m.query.x, m.query.y]);
            }
            let pri_pts = pri_view.unproject_pixels(&surface.primary, &ref_px);
            let mut world_pts = Vec::with_capacity(ref_px.len());
            let mut img_px = Vec::with_capacity(ref_px.len());
            for ((i, opt_w), q_pt) in pri_pts.into_iter().enumerate().zip(qry_px) {
                let resolved = opt_w.or_else(|| {
                    ctx_view.as_ref().and_then(|cv| {
                        cv.unproject_pixels(&surface.primary, &[ref_px[i]])
                            .into_iter()
                            .next()
                            .flatten()
                    })
                });
                if let Some(w) = resolved {
                    world_pts.push(w);
                    img_px.push(q_pt);
                }
            }
            return Ok((raw_count, world_pts, img_px));
        }

        let (ref_rgb, ref_depth, ref_pts) = surface.render(q.eye, q.rot, q.k_render, q.render_size);
        let raw_matches = matcher.match_rgb(&ref_rgb, q.qry_small_rgb)?;
        let raw_count = raw_matches.len();
        let mut world_pts = Vec::with_capacity(raw_count);
        let mut img_px = Vec::with_capacity(raw_count);
        for m in &raw_matches {
            let rx =
                (m.reference.x.round() as i64).clamp(0, i64::from(q.render_size[0]) - 1) as u32;
            let ry =
                (m.reference.y.round() as i64).clamp(0, i64::from(q.render_size[1]) - 1) as u32;
            let r_idx = (ry * q.render_size[0] + rx) as usize;
            if ref_depth[r_idx] <= 0.5 {
                continue;
            }
            let qx_full = m.query.x / q.q_scale[0];
            let qy_full = m.query.y / q.q_scale[1];
            if mask_allows(q.valid_mask, q.full_size, qx_full, qy_full) {
                world_pts.push(ref_pts[r_idx]);
                img_px.push([m.query.x, m.query.y]);
            }
        }
        Ok((raw_count, world_pts, img_px))
    }
}
