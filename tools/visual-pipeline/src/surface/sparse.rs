//! Sparse subpixel keypoint ray-casting and parallel adaptive rendering on [`Surface`].

mod adaptive;

use adaptive::{
    BlockCell, SmoothQuad, build_axis_locs, grid_axis, smooth_surface_corners, worker_count,
};
use image::{Rgb, RgbImage};
use nalgebra::{Matrix3, Vector3};
use std::thread;

use super::{Surface, SurfaceRender};

/// Terrain intersection and orthoimagery colour for one camera ray on [`Surface`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceRayHit {
    /// Hit coordinates `[east_m, north_m, up_m]` in the surface ENU frame.
    pub world_enu: [f64; 3],
    /// Positive optical-axis depth in metres.
    pub depth_m: f32,
    /// Sampled RGB colour at the hit point.
    pub rgb: Rgb<u8>,
}

/// Ray-march work counts from [`Surface::render_adaptive`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceAdaptiveStats {
    /// Total pixels in the rendered view (`width * height`).
    pub total_pixels: usize,
    /// Exact terrain rays marched to produce the view.
    pub rays_cast: usize,
}

pub(super) struct RayProjector {
    eye: Vector3<f64>,
    world_to_cam: Matrix3<f64>,
    cam_to_world: Matrix3<f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

impl RayProjector {
    fn new(eye: [f64; 3], world_to_cam: &Matrix3<f64>, k: &Matrix3<f64>) -> Self {
        Self {
            eye: Vector3::new(eye[0], eye[1], eye[2]),
            world_to_cam: *world_to_cam,
            cam_to_world: world_to_cam.transpose(),
            fx: k[(0, 0)].max(1e-6),
            fy: k[(1, 1)].max(1e-6),
            cx: k[(0, 2)],
            cy: k[(1, 2)],
        }
    }
}

impl Surface {
    /// Lift subpixel image coordinates `(u, v)` to 3D ENU world hits without full-frame rendering.
    pub fn raycast_pixels(
        &self,
        eye: [f64; 3],
        world_to_cam: &Matrix3<f64>,
        k: &Matrix3<f64>,
        pixels: &[[f64; 2]],
    ) -> Vec<Option<SurfaceRayHit>> {
        let proj = RayProjector::new(eye, world_to_cam, k);
        let workers = worker_count(pixels.len() / 64);
        if workers <= 1 {
            return pixels
                .iter()
                .map(|&[u, v]| self.raycast_one(&proj, u, v))
                .collect();
        }
        let mut hits = vec![None; pixels.len()];
        let chunk_len = pixels.len().div_ceil(workers);
        thread::scope(|scope| {
            for (px_chunk, out_chunk) in pixels.chunks(chunk_len).zip(hits.chunks_mut(chunk_len)) {
                let proj_ref = &proj;
                scope.spawn(move || {
                    for (&[u, v], dst) in px_chunk.iter().zip(out_chunk.iter_mut()) {
                        *dst = self.raycast_one(proj_ref, u, v);
                    }
                });
            }
        });
        hits
    }

    /// Render a pinhole view with coarse-grid ray-casting and block edge/horizon refinement.
    #[expect(
        clippy::too_many_arguments,
        reason = "adaptive pinhole surface renderer takes pose, intrinsics, dimensions, and refinement thresholds"
    )]
    pub fn render_adaptive(
        &self,
        eye: [f64; 3],
        world_to_cam: &Matrix3<f64>,
        k: &Matrix3<f64>,
        size: [u32; 2],
        stride: u32,
        rgb_grad_threshold: u8,
        depth_rel_threshold: f32,
    ) -> (SurfaceRender, SurfaceAdaptiveStats) {
        let [w, h] = size;
        let total_pixels = (w as usize).saturating_mul(h as usize);
        if w == 0 || h == 0 {
            return (
                (RgbImage::new(w, h), Vec::new(), Vec::new()),
                SurfaceAdaptiveStats {
                    total_pixels: 0,
                    rays_cast: 0,
                },
            );
        }

        let proj = RayProjector::new(eye, world_to_cam, k);
        let width = w as usize;
        let height = h as usize;
        let mut rgb_raw = vec![0_u8; total_pixels * 3];
        let mut depth = vec![0.0_f32; total_pixels];
        let mut points = vec![[0.0_f64; 3]; total_pixels];

        if stride <= 1 || w <= 1 || h <= 1 {
            self.render_dense_rows(&proj, w, height, &mut rgb_raw, &mut depth, &mut points);
            let rgb = RgbImage::from_raw(w, h, rgb_raw).unwrap_or_else(|| RgbImage::new(w, h));
            return (
                (rgb, depth, points),
                SurfaceAdaptiveStats {
                    total_pixels,
                    rays_cast: total_pixels,
                },
            );
        }

        let xs = grid_axis(w, stride);
        let ys = grid_axis(h, stride);
        let (nx, ny) = (xs.len(), ys.len());
        let mut grid_hits = vec![None; nx * ny];
        let grid_workers = worker_count(ny / 2);
        if grid_workers <= 1 {
            for (gy_idx, &gy) in ys.iter().enumerate() {
                for (gx_idx, &gx) in xs.iter().enumerate() {
                    grid_hits[gy_idx * nx + gx_idx] =
                        self.raycast_one(&proj, f64::from(gx), f64::from(gy));
                }
            }
        } else {
            let rows_per_worker = ny.div_ceil(grid_workers);
            thread::scope(|scope| {
                for (y_slice, out_slice) in ys
                    .chunks(rows_per_worker)
                    .zip(grid_hits.chunks_mut(rows_per_worker * nx))
                {
                    let proj_ref = &proj;
                    let xs_ref = &xs;
                    scope.spawn(move || {
                        for (row_idx, &gy) in y_slice.iter().enumerate() {
                            let row_out = &mut out_slice[row_idx * nx..(row_idx + 1) * nx];
                            for (&gx, dst) in xs_ref.iter().zip(row_out.iter_mut()) {
                                *dst = self.raycast_one(proj_ref, f64::from(gx), f64::from(gy));
                            }
                        }
                    });
                }
            });
        }

        let bx_count = nx - 1;
        let mut blocks = Vec::with_capacity(bx_count * (ny - 1));
        let mut smooth_quads = Vec::new();
        for (by, y_pair) in ys.windows(2).enumerate() {
            for (bx, x_pair) in xs.windows(2).enumerate() {
                let corners = [
                    grid_hits[by * nx + bx],
                    grid_hits[by * nx + bx + 1],
                    grid_hits[(by + 1) * nx + bx],
                    grid_hits[(by + 1) * nx + bx + 1],
                ];
                let cell = if corners.iter().all(Option::is_none) {
                    BlockCell::Empty
                } else if let Some(c) =
                    smooth_surface_corners(corners, rgb_grad_threshold, depth_rel_threshold)
                {
                    let q_idx = smooth_quads.len();
                    smooth_quads.push(SmoothQuad::from_corners(
                        c,
                        [x_pair[0], x_pair[1]],
                        [y_pair[0], y_pair[1]],
                    ));
                    BlockCell::Smooth(q_idx)
                } else {
                    BlockCell::Refine
                };
                blocks.push(cell);
            }
        }

        let x_loc = build_axis_locs(w, &xs);
        let y_loc = build_axis_locs(h, &ys);
        let refine_rays = self.fill_adaptive_rows(
            &proj,
            (&x_loc, &y_loc),
            (&grid_hits, nx),
            (&blocks, &smooth_quads, bx_count),
            width,
            &mut rgb_raw,
            &mut depth,
            &mut points,
        );
        let rgb = RgbImage::from_raw(w, h, rgb_raw).unwrap_or_else(|| RgbImage::new(w, h));
        (
            (rgb, depth, points),
            SurfaceAdaptiveStats {
                total_pixels,
                rays_cast: (nx * ny) + refine_rays,
            },
        )
    }

    pub(super) fn raycast_one(&self, proj: &RayProjector, u: f64, v: f64) -> Option<SurfaceRayHit> {
        if !u.is_finite() || !v.is_finite() {
            return None;
        }
        let ray_cam = Vector3::new((u - proj.cx) / proj.fx, (v - proj.cy) / proj.fy, 1.0);
        let dir_world = proj.cam_to_world * (ray_cam / ray_cam.norm());
        let (hit, rgb) = self.intersect_ray(proj.eye, dir_world)?;
        let rel_cam = proj.world_to_cam * (hit - proj.eye);
        (rel_cam.z > 0.5).then_some(SurfaceRayHit {
            world_enu: [hit.x, hit.y, hit.z],
            depth_m: rel_cam.z as f32,
            rgb,
        })
    }
}
