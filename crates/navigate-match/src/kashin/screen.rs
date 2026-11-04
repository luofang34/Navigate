//! Allocation-free integer `i8` Kashin screening over [`VirtualCorridorView`] candidates.

use super::PreparedKashinQuery;
use crate::corridor::VirtualCorridorView;

const SENTINEL_Q16: i32 = -131_072;
const ONE_Q16: i32 = 65_536;
const STRICT_MIN_Q16: i32 = 40_632;
const SMOOTH_MIN_Q16: i32 = 27_525;

/// Worker-owned scratch buffers for allocation-free [`kashin_screen_stats`] calls.
#[derive(Clone, Debug, Default)]
pub struct KashinScratch {
    smooth_fwd: Vec<(usize, i32)>,
    smooth_rev: Vec<(usize, i32)>,
    strict_fwd: Vec<(usize, i32, i32)>,
    strict_rev: Vec<(usize, i32)>,
    smooth_dots: Vec<i32>,
}

impl KashinScratch {
    /// Create preallocated [`KashinScratch`] buffers sized for `cap` landmarks.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            smooth_fwd: Vec::with_capacity(cap),
            smooth_rev: Vec::with_capacity(cap),
            strict_fwd: Vec::with_capacity(cap),
            strict_rev: Vec::with_capacity(cap),
            smooth_dots: Vec::with_capacity(cap),
        }
    }

    fn reset(&mut self, n_r: usize, n_q: usize) {
        self.smooth_fwd.clear();
        self.smooth_fwd.resize(n_r, (usize::MAX, SENTINEL_Q16));
        self.smooth_rev.clear();
        self.smooth_rev.resize(n_q, (usize::MAX, SENTINEL_Q16));
        self.strict_fwd.clear();
        self.strict_fwd
            .resize(n_r, (usize::MAX, SENTINEL_Q16, SENTINEL_Q16));
        self.strict_rev.clear();
        self.strict_rev.resize(n_q, (usize::MAX, SENTINEL_Q16));
        self.smooth_dots.clear();
    }
}

/// Score `virt` against `query` with safe auto-vectorized `[i8; 64]` dot products.
///
/// Returns `(strict_mutual_count, top_k_smooth_cosine_sum)`. After the first call at a
/// given `cap`, [`KashinScratch`] performs zero heap allocations.
#[must_use]
pub fn kashin_screen_stats(
    scratch: &mut KashinScratch,
    virt: &VirtualCorridorView,
    query: &PreparedKashinQuery,
    cap: usize,
    top_k: usize,
) -> (usize, f64) {
    let n_r = virt.sign_bits.len().min(virt.kashin_i8.len()).min(cap);
    let q_signs = query.sign_bits();
    let q_i8s = query.rotated_i8();
    let q_invs = query.q_inv_q28();
    let n_q = q_signs.len().min(q_i8s.len()).min(q_invs.len()).min(cap);
    if n_r == 0 || n_q == 0 {
        return (0, 0.0);
    }

    scratch.reset(n_r, n_q);
    for r in 0..n_r {
        let rs = virt.sign_bits[r];
        let Some((r_i8, r_inv_q28)) = decode_ref_i8_and_inv(&virt.kashin_i8[r]) else {
            continue;
        };
        let s_slot = &mut scratch.smooth_fwd[r];
        let st_slot = &mut scratch.strict_fwd[r];
        for q in 0..n_q {
            let ham = (rs ^ q_signs[q]).count_ones();
            if ham > 25 {
                continue;
            }
            let acc = dot_i8_64(&r_i8, &q_i8s[q]);
            let scaled = ((i64::from(acc) * r_inv_q28) >> 12) * q_invs[q];
            let cos_q16 = (scaled >> 12) as i32;
            if cos_q16 > s_slot.1 {
                *s_slot = (q, cos_q16);
            }
            if cos_q16 > scratch.smooth_rev[q].1 {
                scratch.smooth_rev[q] = (r, cos_q16);
            }
            if ham <= 24 {
                if cos_q16 > st_slot.1 {
                    *st_slot = (q, cos_q16, st_slot.1);
                } else if cos_q16 > st_slot.2 {
                    st_slot.2 = cos_q16;
                }
                if cos_q16 > scratch.strict_rev[q].1 {
                    scratch.strict_rev[q] = (r, cos_q16);
                }
            }
        }
    }

    let strict_cnt = scratch
        .strict_fwd
        .iter()
        .enumerate()
        .filter(|&(r, &(q, b1, b2))| {
            q < n_q
                && scratch.strict_rev[q].0 == r
                && b1 >= STRICT_MIN_Q16
                && (b2 <= SENTINEL_Q16 / 2
                    || i64::from((ONE_Q16 - b1).max(0)) * 100
                        <= i64::from((ONE_Q16 - b2).max(0)) * 85)
        })
        .count();

    for (r, &(q, d)) in scratch.smooth_fwd.iter().enumerate() {
        if q < n_q && scratch.smooth_rev[q].0 == r && d >= SMOOTH_MIN_Q16 {
            scratch.smooth_dots.push(d);
        }
    }
    scratch.smooth_dots.sort_unstable_by(|a, b| b.cmp(a));
    let sum_q16: i64 = scratch
        .smooth_dots
        .iter()
        .take(top_k)
        .map(|&d| i64::from(d))
        .sum();
    (strict_cnt, (sum_q16 as f64) / f64::from(ONE_Q16))
}

#[inline]
fn decode_ref_i8_and_inv(code: &[u8; 64]) -> Option<([i8; 64], i64)> {
    let mut out = [0_i8; 64];
    let mut norm_sq = 0_u64;
    for (dst, &raw) in out.iter_mut().zip(code) {
        let v = raw as i8;
        *dst = v;
        let vi = i64::from(v);
        norm_sq += (vi * vi) as u64;
    }
    if norm_sq == 0 {
        return None;
    }
    let norm_q8 = (norm_sq << 16).isqrt().max(1);
    let inv_q28 = ((1_i64 << 28) / (norm_q8 as i64)).max(1);
    Some((out, inv_q28))
}

#[inline]
fn dot_i8_64(a: &[i8; 64], b: &[i8; 64]) -> i32 {
    let mut acc = 0_i32;
    for d in 0..64 {
        acc += i32::from(a[d]) * i32::from(b[d]);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DEFAULT_KASHIN_SEED, HadamardProjector, MatchError, SparseFeatures,
        kashin::{pack_kashin_i8, sign_bits64, try_normalize64},
    };

    #[test]
    fn i8_kashin_screen_matches_float_cosine_and_reuses_scratch() -> Result<(), MatchError> {
        let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
        let mut ref_sign = Vec::new();
        let mut ref_kashin = Vec::new();
        let mut query = SparseFeatures::empty();
        for i in 0..96_usize {
            let mut raw = [0.0_f32; 64];
            for (k, slot) in raw.iter_mut().enumerate() {
                let h = ((i * 131 + k * 977 + 17) % 1024) as f32 / 512.0 - 1.0;
                *slot = h;
            }
            let Some(unit) = try_normalize64(&raw) else {
                continue;
            };
            let rot = projector.project(&unit);
            ref_sign.push(sign_bits64(&rot));
            ref_kashin.push(pack_kashin_i8(&rot));
            // First 40 query descriptors match the reference closely; rest are distinct.
            let mut q_raw = raw;
            if i >= 40 {
                q_raw.reverse();
            }
            let Some(q_unit) = try_normalize64(&q_raw) else {
                continue;
            };
            query.pixels.push([i as f64 * 4.0, 10.0]);
            query.model_pixels.push([i as f32 * 4.0, 10.0]);
            query.scores.push(0.9);
            query.descriptors.extend_from_slice(&q_unit);
        }
        let virt = VirtualCorridorView {
            features: SparseFeatures::empty(),
            world_points_enu: Vec::new(),
            sign_bits: ref_sign,
            kashin_i8: ref_kashin,
            visible_tiles: 1,
        };
        let prepared = PreparedKashinQuery::from_features(&projector, &query)?;
        let mut scratch = KashinScratch::with_capacity(128);
        let (strict_cnt, smooth) = kashin_screen_stats(&mut scratch, &virt, &prepared, 128, 32);
        let cap_fwd = scratch.smooth_fwd.capacity();
        let cap_dots = scratch.smooth_dots.capacity();
        let (strict2, smooth2) = kashin_screen_stats(&mut scratch, &virt, &prepared, 128, 32);

        assert_eq!(strict_cnt, 40);
        assert_eq!(strict2, strict_cnt);
        assert!((smooth - 32.0).abs() < 0.25, "expected ~32.0, got {smooth}");
        assert!((smooth2 - smooth).abs() < 1e-12);
        assert_eq!(scratch.smooth_fwd.capacity(), cap_fwd);
        assert_eq!(scratch.smooth_dots.capacity(), cap_dots);
        Ok(())
    }
}
