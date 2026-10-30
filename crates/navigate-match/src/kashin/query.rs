//! Reusable rotated Kashin query buffer and fast L1-dense bank matching.

use super::{
    HadamardProjector, PrecomputedMatch, PrecomputedReferenceBank, ViewSketch, codec::ensure,
    hadamard_moment_signature, octahedron_vertex, sign_bits64, try_normalize64,
};
use crate::{MatchError, SparseFeatures};

/// Pre-rotated query descriptors, sign sketches, and [`ViewSketch`] built once per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedKashinQuery {
    sketch: ViewSketch,
    pixels: Vec<[f64; 2]>,
    valid_indices: Vec<usize>,
    sign_bits: Vec<u64>,
    rotated: Vec<[f32; 64]>,
}

impl PreparedKashinQuery {
    /// Normalize and Hadamard-project `query` features once with `projector`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `query` descriptor length or pixel coordinates are invalid.
    pub fn from_features(
        projector: &HadamardProjector,
        query: &SparseFeatures,
    ) -> Result<Self, MatchError> {
        let count = query.pixels.len();
        let ok = query.descriptors.len() == count * 64
            && (query.scores.is_empty() || query.scores.len() == count)
            && query.pixels.iter().flatten().all(|v| v.is_finite());
        ensure(ok, "query feature buffer is invalid")?;

        let chunks = &query.descriptors.as_chunks::<64>().0[..count];
        let mut valid_indices = Vec::with_capacity(count);
        let mut sign_bits = Vec::with_capacity(count);
        let mut rotated = Vec::with_capacity(count);
        let mut raw_bins = [0.0_f32; 128];
        let mut moments = [[0.0_f32; 64]; 4];
        let mut sketch_valid = 0usize;

        for (idx, desc) in chunks.iter().enumerate() {
            let Some(unit) = try_normalize64(desc) else {
                continue;
            };
            let rot = projector.project(&unit);
            valid_indices.push(idx);
            sign_bits.push(sign_bits64(&rot));
            rotated.push(rot);

            let score = query.scores.get(idx).copied().unwrap_or(1.0);
            if score.is_finite() && score >= 0.0 {
                let weight = if score > 1e-6 { score } else { 1e-3 };
                let (primary_v, primary_peak) = octahedron_vertex(&rot);
                let mass = weight * (1.0 + primary_peak);
                raw_bins[primary_v] += mass;
                raw_bins[primary_v ^ 1] -= 0.5 * mass;
                for block in 0..4 {
                    let start = block << 4;
                    let (local_v, peak) = octahedron_vertex(&rot[start..start + 16]);
                    let (sub_v, sub_mass) = ((block << 5) + local_v, 0.35 * weight * (1.0 + peak));
                    raw_bins[sub_v] += sub_mass;
                    raw_bins[sub_v ^ 1] -= 0.5 * sub_mass;
                }
                for (dst, &val) in moments[(primary_v >> 5) & 3].iter_mut().zip(&rot) {
                    *dst += weight * val;
                }
                sketch_valid += 1;
            }
        }

        let sketch = finish_sketch(projector.seed(), sketch_valid, &mut raw_bins, &mut moments);
        Ok(Self {
            sketch,
            pixels: query.pixels.clone(),
            valid_indices,
            sign_bits,
            rotated,
        })
    }

    /// Return the 160-byte [`ViewSketch`] of the query feature set.
    #[must_use]
    pub const fn sketch(&self) -> &ViewSketch {
        &self.sketch
    }

    /// Return the number of input query keypoints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Return `true` when the query contains zero keypoints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Return the original query pixel coordinates.
    #[must_use]
    pub fn pixels(&self) -> &[[f64; 2]] {
        &self.pixels
    }

    /// Return indices into [`Self::pixels`] for all valid non-zero descriptors.
    #[must_use]
    pub fn valid_indices(&self) -> &[usize] {
        &self.valid_indices
    }

    /// Return contiguous 64-bit sign sketches aligned with [`Self::valid_indices`].
    #[must_use]
    pub fn sign_bits(&self) -> &[u64] {
        &self.sign_bits
    }

    /// Return contiguous Hadamard-rotated unit descriptors aligned with [`Self::valid_indices`].
    #[must_use]
    pub fn rotated(&self) -> &[[f32; 64]] {
        &self.rotated
    }
}

impl PrecomputedReferenceBank {
    /// Rank at most `top_k` `(entry_index, score)` pairs against a [`PreparedKashinQuery`].
    #[must_use]
    pub fn shortlist_prepared(
        &self,
        query: &PreparedKashinQuery,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        self.shortlist_sketch(&query.sketch, top_k)
    }

    pub(super) fn shortlist_sketch(
        &self,
        q_sketch: &ViewSketch,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        if top_k == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let mut ranked: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| (idx, entry.sketch.similarity(q_sketch)))
            .collect();
        let cmp =
            |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
        if top_k < ranked.len() {
            ranked.select_nth_unstable_by(top_k - 1, cmp);
            ranked.truncate(top_k);
        }
        ranked.sort_by(cmp);
        ranked
    }

    /// Match all keypoints in `query` against `entry_index`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `entry_index` or gate thresholds are invalid.
    pub fn match_prepared(
        &self,
        entry_index: usize,
        query: &PreparedKashinQuery,
        hamming_gate: u32,
        min_dot: f32,
        ratio_limit: f32,
    ) -> Result<Vec<PrecomputedMatch>, MatchError> {
        self.match_prepared_limit(
            entry_index,
            query,
            usize::MAX,
            hamming_gate,
            min_dot,
            ratio_limit,
        )
    }

    /// Match the first `max_query_keypoints` of `query` against `entry_index` without reallocation.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `entry_index` or gate thresholds are invalid.
    pub fn match_prepared_limit(
        &self,
        entry_index: usize,
        query: &PreparedKashinQuery,
        max_query_keypoints: usize,
        hamming_gate: u32,
        min_dot: f32,
        ratio_limit: f32,
    ) -> Result<Vec<PrecomputedMatch>, MatchError> {
        let entry = self
            .entries
            .get(entry_index)
            .ok_or(MatchError::Output("bad index"))?;
        entry.validate()?;
        ensure(
            min_dot.is_finite() && ratio_limit.is_finite() && ratio_limit > 0.0,
            "gate",
        )?;
        let active_q = query
            .valid_indices
            .partition_point(|&orig_idx| orig_idx < max_query_keypoints);
        if entry.pixels.is_empty() || active_q == 0 {
            return Ok(Vec::new());
        }

        let q_signs = &query.sign_bits[..active_q];
        let q_rots = &query.rotated[..active_q];
        let has_i8 = entry.kashin_i8.len() == entry.pixels.len();
        let mut ref_best = vec![Nearest::EMPTY; entry.pixels.len()];
        let mut qry_best = vec![Nearest::EMPTY; active_q];

        for (i, &ref_sign) in entry.sign_bits.iter().enumerate() {
            for (j, &q_sign) in q_signs.iter().enumerate() {
                let hd = (ref_sign ^ q_sign).count_ones();
                if hd > hamming_gate {
                    ref_best[i].3 = ref_best[i].3.min(hd);
                    qry_best[j].3 = qry_best[j].3.min(hd);
                } else {
                    let score = if has_i8 {
                        approx_dot_i8(&entry.kashin_i8[i], &q_rots[j])
                    } else {
                        entry.descriptors[i].approx_dot_query(&q_rots[j])
                    };
                    ref_best[i].update(j, score);
                    qry_best[j].update(i, score);
                }
            }
        }

        let r2 = ratio_limit * ratio_limit;
        let mut out = Vec::new();
        for (i, fwd) in ref_best.iter().enumerate() {
            let ok = qry_best.get(fwd.0).is_some_and(|rev| {
                rev.0 == i && fwd.accepts(min_dot, r2) && rev.accepts(min_dot, r2)
            });
            if ok {
                let orig_q = query.valid_indices[fwd.0];
                let world_point_enu = entry
                    .world_points_enu
                    .get(i)
                    .copied()
                    .filter(|pt| pt.iter().all(|v| v.is_finite()));
                out.push(PrecomputedMatch {
                    reference_pixel: entry.pixels[i].map(f64::from),
                    query_pixel: query.pixels[orig_q],
                    world_point_enu,
                    score: fwd.1,
                });
            }
        }
        Ok(out)
    }
}

pub(crate) fn approx_dot_i8(code: &[u8; 64], rotated_query: &[f32; 64]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_sq = 0i32;
    for (&raw, &q) in code.iter().zip(rotated_query) {
        let v = raw as i8;
        let vi = i32::from(v);
        norm_sq += vi * vi;
        dot += f32::from(v) * q;
    }
    if norm_sq <= 0 {
        return 0.0;
    }
    (dot / (norm_sq as f32).sqrt()).clamp(-1.0, 1.0)
}

fn finish_sketch(
    seed: u64,
    valid: usize,
    raw_bins: &mut [f32; 128],
    moments: &mut [[f32; 64]; 4],
) -> ViewSketch {
    if valid == 0 {
        return ViewSketch::empty();
    }
    let mean = raw_bins.iter().sum::<f32>() / 128.0;
    for bin in raw_bins.iter_mut() {
        *bin -= mean;
    }
    let norm = raw_bins.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        return ViewSketch::empty();
    }
    let vertex_bins = raw_bins.map(|v| (v / norm * 127.0).round().clamp(-127.0, 127.0) as i8);
    ViewSketch {
        vertex_bins,
        sign_signature: hadamard_moment_signature(seed, moments),
    }
}

#[derive(Clone, Copy)]
struct Nearest(usize, f32, f32, u32);

impl Nearest {
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
