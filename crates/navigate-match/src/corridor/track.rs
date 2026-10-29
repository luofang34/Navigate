//! Spatial-radius-gated 3D-2D Kashin track matcher for `25-30 Hz` frame tracking.

use super::{codec::ensure, mat3_mul};
use crate::{
    MatchError, PreparedKashinQuery,
    kashin::{pack_kashin_i8, query::approx_dot_i8},
};

/// Gate parameters for [`ActiveTrackSet::match_radius`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackMatchConfig {
    /// Maximum pixel radius around each projected 3D landmark (`> 0.0`).
    pub search_radius_px: f32,
    /// Maximum 64-bit sign Hamming distance (`0..=64`).
    pub hamming_gate: u32,
    /// Minimum Kashin `i8` cosine similarity (`[-1.0, 1.0]`).
    pub min_dot: f32,
    /// Lowe second-best angular ratio threshold (`(0.0, 1.0]`).
    pub ratio_limit: f32,
}

impl Default for TrackMatchConfig {
    fn default() -> Self {
        Self {
            search_radius_px: 32.0,
            hamming_gate: 22,
            min_dot: 0.65,
            ratio_limit: 0.90,
        }
    }
}

/// Verified 3D-2D track correspondence produced by [`ActiveTrackSet::match_radius`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tracked3dMatch {
    /// Index of the matched landmark in [`ActiveTrackSet`].
    pub track_index: usize,
    /// Index of the matched keypoint in the query [`PreparedKashinQuery::pixels`].
    pub query_index: usize,
    /// Matched 3D ENU world coordinates `[east, north, up]` in meters.
    pub world_point_enu: [f64; 3],
    /// Predicted projection `[u, v]` of `world_point_enu` under the prior pose.
    pub predicted_pixel: [f64; 2],
    /// Observed keypoint pixel `[u, v]` in the current query frame.
    pub query_pixel: [f64; 2],
    /// Kashin `i8` cosine similarity score (`[-1.0, 1.0]`).
    pub score: f32,
}

/// Active 3D terrain landmarks tracked across sequential video frames.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ActiveTrackSet {
    /// 3D ENU coordinates `[east, north, up]` in meters.
    pub world_points_enu: Vec<[f64; 3]>,
    /// Contiguous 64-bit Kashin sign sketches (`world_points_enu.len()` elements).
    pub sign_bits: Vec<u64>,
    /// Bounded 64-byte rotated Kashin `i8` descriptors (`world_points_enu.len()` elements).
    pub kashin_i8: Vec<[u8; 64]>,
}

impl ActiveTrackSet {
    /// Create an empty [`ActiveTrackSet`].
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            world_points_enu: Vec::new(),
            sign_bits: Vec::new(),
            kashin_i8: Vec::new(),
        }
    }

    /// Build an [`ActiveTrackSet`] from a [`PreparedKashinQuery`] and ray-cast 3D ENU hits.
    #[must_use]
    pub fn from_prepared_query(
        query: &PreparedKashinQuery,
        world_hits_by_pixel: &[Option<[f64; 3]>],
        max_tracks: usize,
    ) -> Self {
        let cap = max_tracks.min(query.valid_indices().len());
        let mut out = Self {
            world_points_enu: Vec::with_capacity(cap),
            sign_bits: Vec::with_capacity(cap),
            kashin_i8: Vec::with_capacity(cap),
        };
        for (slot, &orig_idx) in query.valid_indices().iter().enumerate() {
            if out.world_points_enu.len() >= max_tracks {
                break;
            }
            let Some(Some(w)) = world_hits_by_pixel.get(orig_idx).copied() else {
                continue;
            };
            if !w.iter().all(|v| v.is_finite()) {
                continue;
            }
            out.world_points_enu.push(w);
            out.sign_bits.push(query.sign_bits()[slot]);
            out.kashin_i8.push(pack_kashin_i8(&query.rotated()[slot]));
        }
        out
    }

    /// Return the number of active 3D tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.world_points_enu.len()
    }

    /// Return `true` when zero active tracks are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.world_points_enu.is_empty()
    }

    /// Append additional 3D Kashin landmarks up to `max_tracks`.
    pub fn extend_capped(
        &mut self,
        world_points: &[[f64; 3]],
        sign_bits: &[u64],
        kashin_i8: &[[u8; 64]],
        max_tracks: usize,
    ) {
        let count = world_points.len().min(sign_bits.len()).min(kashin_i8.len());
        for i in 0..count {
            if self.world_points_enu.len() >= max_tracks {
                break;
            }
            if world_points[i].iter().all(|v| v.is_finite()) {
                self.world_points_enu.push(world_points[i]);
                self.sign_bits.push(sign_bits[i]);
                self.kashin_i8.push(kashin_i8[i]);
            }
        }
    }

    /// Match projected 3D tracks against `query` within a 2D pixel search radius.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `cfg` or track buffer lengths are invalid.
    pub fn match_radius(
        &self,
        query: &PreparedKashinQuery,
        eye_enu: [f64; 3],
        world_to_cam: &[f64; 9],
        intrinsics: [f64; 4],
        image_size: [u32; 2],
        cfg: &TrackMatchConfig,
    ) -> Result<Vec<Tracked3dMatch>, MatchError> {
        let n_tracks = self.world_points_enu.len();
        ensure(
            self.sign_bits.len() == n_tracks && self.kashin_i8.len() == n_tracks,
            "track buffer length mismatch",
        )?;
        let cfg_ok = cfg.search_radius_px.is_finite()
            && cfg.search_radius_px > 0.0
            && cfg.min_dot.is_finite()
            && cfg.ratio_limit.is_finite()
            && cfg.ratio_limit > 0.0;
        ensure(cfg_ok, "invalid TrackMatchConfig")?;
        let [w, h] = image_size;
        let [fx, fy, cx, cy] = intrinsics;
        let q_valid = query.valid_indices();
        if n_tracks == 0 || q_valid.is_empty() || w == 0 || h == 0 || fx <= 1e-6 || fy <= 1e-6 {
            return Ok(Vec::new());
        }

        let cell_px = f64::from(cfg.search_radius_px.max(4.0));
        let wf = f64::from(w);
        let hf = f64::from(h);
        let gx = ((wf / cell_px).ceil() as usize).max(1);
        let gy = ((hf / cell_px).ceil() as usize).max(1);
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); gx * gy];
        let q_pixels = query.pixels();
        for (slot, &orig_idx) in q_valid.iter().enumerate() {
            let [qx, qy] = q_pixels[orig_idx];
            if (0.0..wf).contains(&qx) && (0.0..hf).contains(&qy) {
                let bx = ((qx / cell_px).floor() as usize).min(gx - 1);
                let by = ((qy / cell_px).floor() as usize).min(gy - 1);
                buckets[by * gx + bx].push(slot);
            }
        }

        let r2_px = f64::from(cfg.search_radius_px) * f64::from(cfg.search_radius_px);
        let margin = f64::from(cfg.search_radius_px);
        let q_signs = query.sign_bits();
        let q_rots = query.rotated();
        let mut trk_best = vec![LocalNearest::EMPTY; n_tracks];
        let mut qry_best = vec![LocalNearest::EMPTY; q_valid.len()];
        let mut pred_uv = vec![[0.0_f64; 2]; n_tracks];

        for t_idx in 0..n_tracks {
            let world = self.world_points_enu[t_idx];
            let pc = mat3_mul(
                world_to_cam,
                [
                    world[0] - eye_enu[0],
                    world[1] - eye_enu[1],
                    world[2] - eye_enu[2],
                ],
            );
            if pc[2] <= 1.0 {
                continue;
            }
            let u = fx * (pc[0] / pc[2]) + cx;
            let v = fy * (pc[1] / pc[2]) + cy;
            if u < -margin || u >= wf + margin || v < -margin || v >= hf + margin {
                continue;
            }
            pred_uv[t_idx] = [u, v];
            let bx0 = (((u - margin) / cell_px).floor().max(0.0) as usize).min(gx - 1);
            let bx1 = (((u + margin) / cell_px).floor().max(0.0) as usize).min(gx - 1);
            let by0 = (((v - margin) / cell_px).floor().max(0.0) as usize).min(gy - 1);
            let by1 = (((v + margin) / cell_px).floor().max(0.0) as usize).min(gy - 1);
            let t_sign = self.sign_bits[t_idx];

            for by in by0..=by1 {
                for bx in bx0..=bx1 {
                    for &q_slot in &buckets[by * gx + bx] {
                        let [qx, qy] = q_pixels[q_valid[q_slot]];
                        let (dx, dy) = (qx - u, qy - v);
                        if dx * dx + dy * dy > r2_px {
                            continue;
                        }
                        let hd = (t_sign ^ q_signs[q_slot]).count_ones();
                        if hd > cfg.hamming_gate {
                            trk_best[t_idx].3 = trk_best[t_idx].3.min(hd);
                            qry_best[q_slot].3 = qry_best[q_slot].3.min(hd);
                        } else {
                            let score = approx_dot_i8(&self.kashin_i8[t_idx], &q_rots[q_slot]);
                            trk_best[t_idx].update(q_slot, score);
                            qry_best[q_slot].update(t_idx, score);
                        }
                    }
                }
            }
        }

        let ratio_sq = cfg.ratio_limit * cfg.ratio_limit;
        let mut matches = Vec::new();
        for (t_idx, fwd) in trk_best.iter().enumerate() {
            let ok = qry_best.get(fwd.0).is_some_and(|rev| {
                rev.0 == t_idx
                    && fwd.accepts(cfg.min_dot, ratio_sq)
                    && rev.accepts(cfg.min_dot, ratio_sq)
            });
            if ok {
                let orig_q = q_valid[fwd.0];
                matches.push(Tracked3dMatch {
                    track_index: t_idx,
                    query_index: orig_q,
                    world_point_enu: self.world_points_enu[t_idx],
                    predicted_pixel: pred_uv[t_idx],
                    query_pixel: q_pixels[orig_q],
                    score: fwd.1,
                });
            }
        }
        Ok(matches)
    }
}

#[derive(Clone, Copy)]
struct LocalNearest(usize, f32, f32, u32);

impl LocalNearest {
    const EMPTY: Self = Self(usize::MAX, -2.0, -2.0, u32::MAX);

    fn update(&mut self, index: usize, score: f32) {
        if score > self.1 {
            (self.2, self.1, self.0) = (self.1, score, index);
        } else if score > self.2 {
            self.2 = score;
        }
    }

    fn accepts(self, min_dot: f32, ratio_sq: f32) -> bool {
        if self.0 == usize::MAX || self.1 < min_dot {
            return false;
        }
        let runner = if self.3 <= 64 {
            self.2
                .max((std::f32::consts::PI * (self.3 as f32) / 64.0).cos())
        } else {
            self.2
        };
        runner <= -1.5 || (1.0 - self.1).max(0.0) <= ratio_sq * (1.0 - runner).max(0.0)
    }
}
