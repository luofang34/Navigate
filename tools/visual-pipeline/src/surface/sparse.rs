//! Sparse subpixel keypoint ray-casting and adaptive rendering on [`Surface`].

use image::{Rgb, RgbImage};
use nalgebra::{Matrix3, Vector3};

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

struct RayProjector {
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
        let mut hits = Vec::with_capacity(pixels.len());
        for &[u, v] in pixels {
            hits.push(self.raycast_one(&proj, u, v));
        }
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

        let mut rgb = RgbImage::from_pixel(w, h, Rgb([0, 0, 0]));
        let mut depth = vec![0.0_f32; total_pixels];
        let mut points = vec![[0.0_f64; 3]; total_pixels];
        let proj = RayProjector::new(eye, world_to_cam, k);
        let width = w as usize;

        if stride <= 1 {
            for y in 0..h {
                for x in 0..w {
                    if let Some(hit) = self.raycast_one(&proj, f64::from(x), f64::from(y)) {
                        let idx = (y as usize) * width + (x as usize);
                        rgb.put_pixel(x, y, hit.rgb);
                        depth[idx] = hit.depth_m;
                        points[idx] = hit.world_enu;
                    }
                }
            }
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
        let nx = xs.len();
        let mut state = vec![0_u8; total_pixels];
        let mut grid_hits = Vec::with_capacity(nx * ys.len());
        let mut rays_cast = 0_usize;

        for &gy in &ys {
            for &gx in &xs {
                let hit = self.raycast_one(&proj, f64::from(gx), f64::from(gy));
                let idx = (gy as usize) * width + (gx as usize);
                state[idx] = 2;
                rays_cast += 1;
                if let Some(c) = hit {
                    rgb.put_pixel(gx, gy, c.rgb);
                    depth[idx] = c.depth_m;
                    points[idx] = c.world_enu;
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
                if corners.iter().all(Option::is_none) {
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let idx = (y as usize) * width + (x as usize);
                            if state[idx] == 0 {
                                state[idx] = 1;
                            }
                        }
                    }
                } else if let Some(c) =
                    smooth_surface_corners(corners, rgb_grad_threshold, depth_rel_threshold)
                {
                    let mut out = BlockBuffers {
                        state: &mut state,
                        rgb: &mut rgb,
                        depth: &mut depth,
                        points: &mut points,
                    };
                    interpolate_surface_block(&c, [x0, x1], [y0, y1], width, &mut out);
                } else {
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let idx = (y as usize) * width + (x as usize);
                            if state[idx] == 2 {
                                continue;
                            }
                            state[idx] = 2;
                            rays_cast += 1;
                            if let Some(hit) = self.raycast_one(&proj, f64::from(x), f64::from(y)) {
                                rgb.put_pixel(x, y, hit.rgb);
                                depth[idx] = hit.depth_m;
                                points[idx] = hit.world_enu;
                            } else {
                                rgb.put_pixel(x, y, Rgb([0, 0, 0]));
                                depth[idx] = 0.0;
                                points[idx] = [0.0; 3];
                            }
                        }
                    }
                }
            }
        }

        (
            (rgb, depth, points),
            SurfaceAdaptiveStats {
                total_pixels,
                rays_cast,
            },
        )
    }

    fn raycast_one(&self, proj: &RayProjector, u: f64, v: f64) -> Option<SurfaceRayHit> {
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

struct BlockBuffers<'a> {
    state: &'a mut [u8],
    rgb: &'a mut RgbImage,
    depth: &'a mut [f32],
    points: &'a mut [[f64; 3]],
}

fn grid_axis(len: u32, stride: u32) -> Vec<u32> {
    let mut coords: Vec<u32> = (0..len).step_by(stride as usize).collect();
    if coords.last().copied() != Some(len - 1) {
        coords.push(len - 1);
    }
    coords
}

fn smooth_surface_corners(
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

fn interpolate_surface_block(
    c: &[SurfaceRayHit; 4],
    [x0, x1]: [u32; 2],
    [y0, y1]: [u32; 2],
    width: usize,
    out: &mut BlockBuffers<'_>,
) {
    let dx = f64::from(x1 - x0);
    let dy = f64::from(y1 - y0);
    let inv_d = c.map(|h| 1.0 / f64::from(h.depth_m));
    let w_over_d = [0, 1, 2, 3].map(|i| c[i].world_enu.map(|v| v * inv_d[i]));
    for y in y0..=y1 {
        let ty = if dy > 0.0 {
            f64::from(y - y0) / dy
        } else {
            0.0
        };
        for x in x0..=x1 {
            let idx = (y as usize) * width + (x as usize);
            if out.state[idx] != 0 {
                continue;
            }
            let tx = if dx > 0.0 {
                f64::from(x - x0) / dx
            } else {
                0.0
            };
            let bilerp = |v: [f64; 4]| -> f64 {
                let top = v[0] * (1.0 - tx) + v[1] * tx;
                let bot = v[2] * (1.0 - tx) + v[3] * tx;
                top * (1.0 - ty) + bot * ty
            };
            let id = bilerp(inv_d);
            let d = 1.0 / id;
            let mut ch_out = [0_u8; 3];
            for (ch, slot) in ch_out.iter_mut().enumerate() {
                let vals = c.map(|h| f64::from(h.rgb.0[ch]));
                *slot = bilerp(vals).round().clamp(0.0, 255.0) as u8;
            }
            let world = [0, 1, 2].map(|axis| bilerp(w_over_d.map(|w| w[axis])) * d);
            out.state[idx] = 1;
            out.rgb.put_pixel(x, y, Rgb(ch_out));
            out.depth[idx] = d as f32;
            out.points[idx] = world;
        }
    }
}
