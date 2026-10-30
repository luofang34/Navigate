//! Parallel coarse-grid interpolation and row-strip helpers for [`super::Surface::render_adaptive`].

use super::{RayProjector, Surface, SurfaceRayHit};
use std::thread;

#[derive(Clone, Copy)]
pub(super) enum AxisLoc {
    GridEdge {
        grid_idx: usize,
        block_a: usize,
        block_b: usize,
    },
    Interior {
        block: usize,
    },
}

#[derive(Clone, Copy)]
pub(super) struct SmoothQuad {
    x0: u32,
    dx: f64,
    y0: u32,
    dy: f64,
    inv_d: [f64; 4],
    w_over_d: [[f64; 3]; 4],
    rgb: [[f64; 4]; 3],
}

impl SmoothQuad {
    pub(super) fn from_corners(
        c: [SurfaceRayHit; 4],
        [x0, x1]: [u32; 2],
        [y0, y1]: [u32; 2],
    ) -> Self {
        let inv_d = c.map(|h| 1.0 / f64::from(h.depth_m));
        let w_over_d = [0, 1, 2, 3].map(|i| c[i].world_enu.map(|v| v * inv_d[i]));
        let rgb = [0, 1, 2].map(|ch| c.map(|h| f64::from(h.rgb.0[ch])));
        Self {
            x0,
            dx: f64::from(x1 - x0),
            y0,
            dy: f64::from(y1 - y0),
            inv_d,
            w_over_d,
            rgb,
        }
    }

    fn eval(&self, x: u32, y: u32) -> ([u8; 3], f32, [f64; 3]) {
        let tx = if self.dx > 0.0 {
            f64::from(x - self.x0) / self.dx
        } else {
            0.0
        };
        let ty = if self.dy > 0.0 {
            f64::from(y - self.y0) / self.dy
        } else {
            0.0
        };
        let bilerp = |v: [f64; 4]| -> f64 {
            let top = v[0] * (1.0 - tx) + v[1] * tx;
            let bot = v[2] * (1.0 - tx) + v[3] * tx;
            top * (1.0 - ty) + bot * ty
        };
        let d = 1.0 / bilerp(self.inv_d);
        let rgb = self
            .rgb
            .map(|vals| bilerp(vals).round().clamp(0.0, 255.0) as u8);
        let world = [0, 1, 2].map(|axis| bilerp(self.w_over_d.map(|w| w[axis])) * d);
        (rgb, d as f32, world)
    }
}

#[derive(Clone, Copy)]
pub(super) enum BlockCell {
    Empty,
    Smooth(usize),
    Refine,
}

impl Surface {
    pub(super) fn render_dense_rows(
        &self,
        proj: &RayProjector,
        w: u32,
        height: usize,
        rgb_raw: &mut [u8],
        depth: &mut [f32],
        points: &mut [[f64; 3]],
    ) {
        let width = w as usize;
        let workers = worker_count(height / 8);
        let rows_per = height.div_ceil(workers);
        thread::scope(|scope| {
            for (chunk_idx, ((rgb_c, depth_c), pts_c)) in rgb_raw
                .chunks_mut(rows_per * width * 3)
                .zip(depth.chunks_mut(rows_per * width))
                .zip(points.chunks_mut(rows_per * width))
                .enumerate()
            {
                let y_base = chunk_idx * rows_per;
                scope.spawn(move || {
                    let local_rows = depth_c.len() / width;
                    for r in 0..local_rows {
                        let y = (y_base + r) as u32;
                        for x in 0..w {
                            if let Some(hit) = self.raycast_one(proj, f64::from(x), f64::from(y)) {
                                let idx = r * width + (x as usize);
                                rgb_c[idx * 3..idx * 3 + 3].copy_from_slice(&hit.rgb.0);
                                depth_c[idx] = hit.depth_m;
                                pts_c[idx] = hit.world_enu;
                            }
                        }
                    }
                });
            }
        });
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "parallel row filler writes disjoint RGB, depth, and ENU slices from shared grid state"
    )]
    pub(super) fn fill_adaptive_rows(
        &self,
        proj: &RayProjector,
        (x_loc, y_loc): (&[AxisLoc], &[AxisLoc]),
        (grid_hits, nx): (&[Option<SurfaceRayHit>], usize),
        (blocks, smooth_quads, bx_count): (&[BlockCell], &[SmoothQuad], usize),
        width: usize,
        rgb_raw: &mut [u8],
        depth: &mut [f32],
        points: &mut [[f64; 3]],
    ) -> usize {
        let height = y_loc.len();
        let workers = worker_count(height / 8);
        let rows_per = height.div_ceil(workers);
        let mut counts = vec![0_usize; rgb_raw.chunks(rows_per * width * 3).len()];

        thread::scope(|scope| {
            for (chunk_idx, (((rgb_c, depth_c), pts_c), ray_dst)) in rgb_raw
                .chunks_mut(rows_per * width * 3)
                .zip(depth.chunks_mut(rows_per * width))
                .zip(points.chunks_mut(rows_per * width))
                .zip(counts.iter_mut())
                .enumerate()
            {
                let y_base = chunk_idx * rows_per;
                scope.spawn(move || {
                    let local_rows = depth_c.len() / width;
                    let mut local_rays = 0_usize;
                    for r in 0..local_rows {
                        let y = y_base + r;
                        let yl = y_loc[y];
                        for (x, &xl) in x_loc.iter().enumerate() {
                            let idx = r * width + x;
                            let rgb_dst = &mut rgb_c[idx * 3..idx * 3 + 3];
                            let depth_dst = &mut depth_c[idx];
                            let pt_dst = &mut pts_c[idx];
                            match (xl, yl) {
                                (
                                    AxisLoc::GridEdge { grid_idx: gx, .. },
                                    AxisLoc::GridEdge { grid_idx: gy, .. },
                                ) => {
                                    if let Some(h) = grid_hits[gy * nx + gx] {
                                        rgb_dst.copy_from_slice(&h.rgb.0);
                                        *depth_dst = h.depth_m;
                                        *pt_dst = h.world_enu;
                                    }
                                }
                                (
                                    AxisLoc::Interior { block: bx },
                                    AxisLoc::Interior { block: by },
                                ) => {
                                    self.eval_pair(
                                        proj,
                                        smooth_quads,
                                        (x as u32, y as u32),
                                        (blocks[by * bx_count + bx], BlockCell::Empty),
                                        &mut local_rays,
                                        (rgb_dst, depth_dst, pt_dst),
                                    );
                                }
                                (
                                    AxisLoc::GridEdge {
                                        block_a: bx0,
                                        block_b: bx1,
                                        ..
                                    },
                                    AxisLoc::Interior { block: by },
                                ) => {
                                    self.eval_pair(
                                        proj,
                                        smooth_quads,
                                        (x as u32, y as u32),
                                        (blocks[by * bx_count + bx0], blocks[by * bx_count + bx1]),
                                        &mut local_rays,
                                        (rgb_dst, depth_dst, pt_dst),
                                    );
                                }
                                (
                                    AxisLoc::Interior { block: bx },
                                    AxisLoc::GridEdge {
                                        block_a: by0,
                                        block_b: by1,
                                        ..
                                    },
                                ) => {
                                    self.eval_pair(
                                        proj,
                                        smooth_quads,
                                        (x as u32, y as u32),
                                        (blocks[by0 * bx_count + bx], blocks[by1 * bx_count + bx]),
                                        &mut local_rays,
                                        (rgb_dst, depth_dst, pt_dst),
                                    );
                                }
                            }
                        }
                    }
                    *ray_dst = local_rays;
                });
            }
        });
        counts.into_iter().sum()
    }

    fn eval_pair(
        &self,
        proj: &RayProjector,
        smooth_quads: &[SmoothQuad],
        (x, y): (u32, u32),
        (a, b): (BlockCell, BlockCell),
        rays: &mut usize,
        (rgb_dst, depth_dst, pt_dst): (&mut [u8], &mut f32, &mut [f64; 3]),
    ) {
        if matches!(a, BlockCell::Refine) || matches!(b, BlockCell::Refine) {
            *rays += 1;
            if let Some(h) = self.raycast_one(proj, f64::from(x), f64::from(y)) {
                rgb_dst.copy_from_slice(&h.rgb.0);
                *depth_dst = h.depth_m;
                *pt_dst = h.world_enu;
            }
        } else if let BlockCell::Smooth(q_idx) = a {
            let (c, d, w) = smooth_quads[q_idx].eval(x, y);
            rgb_dst.copy_from_slice(&c);
            *depth_dst = d;
            *pt_dst = w;
        } else if let BlockCell::Smooth(q_idx) = b {
            let (c, d, w) = smooth_quads[q_idx].eval(x, y);
            rgb_dst.copy_from_slice(&c);
            *depth_dst = d;
            *pt_dst = w;
        }
    }
}

pub(super) fn worker_count(work_units: usize) -> usize {
    if work_units <= 1 {
        return 1;
    }
    let avail = thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    avail.min(16).min(work_units).max(1)
}

pub(super) fn grid_axis(len: u32, stride: u32) -> Vec<u32> {
    let mut coords: Vec<u32> = (0..len).step_by(stride as usize).collect();
    if coords.last().copied() != Some(len - 1) {
        coords.push(len - 1);
    }
    coords
}

pub(super) fn build_axis_locs(len: u32, coords: &[u32]) -> Vec<AxisLoc> {
    let mut locs = Vec::with_capacity(len as usize);
    let last_block = coords.len().saturating_sub(2);
    let mut g = 0_usize;
    for pos in 0..len {
        while g + 1 < coords.len() && coords[g + 1] < pos {
            g += 1;
        }
        if coords[g] == pos {
            let block_a = g.saturating_sub(1).min(last_block);
            let block_b = g.min(last_block);
            locs.push(AxisLoc::GridEdge {
                grid_idx: g,
                block_a,
                block_b,
            });
        } else if g + 1 < coords.len() && coords[g + 1] == pos {
            g += 1;
            let block_a = g.saturating_sub(1).min(last_block);
            let block_b = g.min(last_block);
            locs.push(AxisLoc::GridEdge {
                grid_idx: g,
                block_a,
                block_b,
            });
        } else {
            locs.push(AxisLoc::Interior {
                block: g.min(last_block),
            });
        }
    }
    locs
}

pub(super) fn smooth_surface_corners(
    corners: [Option<SurfaceRayHit>; 4],
    rgb_grad_threshold: u8,
    depth_rel_threshold: f32,
) -> Option<[SurfaceRayHit; 4]> {
    if !depth_rel_threshold.is_finite() || depth_rel_threshold < 0.0 {
        return None;
    }
    let hit = [corners[0]?, corners[1]?, corners[2]?, corners[3]?];
    let mut c_min = [u8::MAX; 3];
    let mut c_max = [u8::MIN; 3];
    let mut d_min = f32::INFINITY;
    let mut d_max = 0.0_f32;
    for h in hit {
        if h.depth_m <= 0.0 {
            return None;
        }
        for ch in 0..3 {
            c_min[ch] = c_min[ch].min(h.rgb.0[ch]);
            c_max[ch] = c_max[ch].max(h.rgb.0[ch]);
        }
        d_min = d_min.min(h.depth_m);
        d_max = d_max.max(h.depth_m);
    }
    for ch in 0..3 {
        if c_max[ch] - c_min[ch] > rgb_grad_threshold {
            return None;
        }
    }
    if (d_max - d_min) > depth_rel_threshold * d_min {
        return None;
    }
    Some(hit)
}
