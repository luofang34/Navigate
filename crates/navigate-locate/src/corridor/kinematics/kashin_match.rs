//! Kashin `i8` sketch similarity and frustum-wide descriptor matching.

use navigate_match::{ActiveTrackSet, PreparedKashinQuery, VirtualCorridorView};

/// Compute the approximate cosine similarity between a Kashin `i8` code and a rotated query.
#[must_use]
pub fn approx_dot(code: &[u8; 64], rot: &[f32; 64]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_sq = 0_i32;
    for (&raw, &q) in code.iter().zip(rot) {
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

/// Match active 3D landmarks against a query frame without a pixel radius limit.
#[must_use]
pub fn match_tracks_kashin(
    tracks: &ActiveTrackSet,
    query: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let (world, px, _) = match_tracks_kashin_scored(tracks, query, cap);
    (world, px)
}

/// Match active 3D landmarks against a query frame and return a continuous Kashin alignment score.
#[must_use]
pub fn match_tracks_kashin_scored(
    tracks: &ActiveTrackSet,
    query: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>, f32) {
    let n_r = tracks.len().min(cap);
    let q_valid = query.valid_indices();
    let n_q = q_valid.len().min(cap);
    if n_r == 0 || n_q == 0 {
        return (Vec::new(), Vec::new(), 0.0);
    }
    let q_signs = query.sign_bits();
    let q_rots = query.rotated();
    let q_px = query.pixels();
    let mut fwd = vec![(usize::MAX, -2.0_f32, -2.0_f32); n_r];
    let mut rev = vec![(usize::MAX, -2.0_f32); n_q];

    for (r, fwd_slot) in fwd.iter_mut().enumerate() {
        let rs = tracks.sign_bits[r];
        let rc = &tracks.kashin_i8[r];
        for q in 0..n_q {
            if (rs ^ q_signs[q]).count_ones() > 24 {
                continue;
            }
            let dot = approx_dot(rc, &q_rots[q]);
            if dot > fwd_slot.1 {
                *fwd_slot = (q, dot, fwd_slot.1);
            } else if dot > fwd_slot.2 {
                fwd_slot.2 = dot;
            }
            if dot > rev[q].1 {
                rev[q] = (r, dot);
            }
        }
    }
    let mut world = Vec::new();
    let mut px = Vec::new();
    let mut mutual_dot_sum = 0.0_f32;
    let mut top_dots = [0.0_f32; 16];
    for (r, &(q, b1, b2)) in fwd.iter().enumerate() {
        if b1 > 0.45 && b1 > top_dots[15] {
            top_dots[15] = b1;
            let mut k = 15;
            while k > 0 && top_dots[k] > top_dots[k - 1] {
                top_dots.swap(k, k - 1);
                k -= 1;
            }
        }
        if q < n_q
            && rev[q].0 == r
            && b1 >= 0.62
            && (b2 <= -1.5 || (1.0 - b1).max(0.0) <= 0.85 * (1.0 - b2).max(0.0))
        {
            world.push(tracks.world_points_enu[r]);
            px.push(q_px[q_valid[q]]);
            mutual_dot_sum += b1.max(0.0);
        }
    }
    let top_sum: f32 = top_dots.iter().sum();
    (world, px, mutual_dot_sum * 4.0 + top_sum)
}

/// Match virtual corridor landmarks against a query frame.
#[must_use]
pub fn match_virtual_kashin(
    virt: &VirtualCorridorView,
    query: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let (world, px, _) = match_virtual_kashin_scored(virt, query, cap);
    (world, px)
}

/// Match virtual corridor landmarks against a query frame and return a Kashin alignment score.
#[must_use]
pub fn match_virtual_kashin_scored(
    virt: &VirtualCorridorView,
    query: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>, f32) {
    let probe = ActiveTrackSet {
        world_points_enu: virt.world_points_enu.iter().take(cap).copied().collect(),
        sign_bits: virt.sign_bits.iter().take(cap).copied().collect(),
        kashin_i8: virt.kashin_i8.iter().take(cap).copied().collect(),
    };
    match_tracks_kashin_scored(&probe, query, cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use navigate_match::{DEFAULT_KASHIN_SEED, HadamardProjector, SparseFeatures};

    fn make_unit_desc(seed: u64) -> [f32; 64] {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut raw = [0.0_f32; 64];
        let mut norm2 = 0.0_f32;
        for slot in &mut raw {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let v = ((s >> 33) as i32 as f32) * (1.0 / (u32::MAX as f32)) - 0.5;
            *slot = v;
            norm2 += v * v;
        }
        let inv = 1.0 / norm2.sqrt().max(1e-6);
        raw.map(|v| v * inv)
    }

    fn encode_i8_sketch(proj: &HadamardProjector, desc: &[f32; 64]) -> (u64, [u8; 64]) {
        let rot = proj.project(desc);
        let mut sign = 0_u64;
        let mut code = [0_u8; 64];
        let scale = 127.0 / (3.5 * 0.125_f32);
        for (k, &v) in rot.iter().enumerate() {
            if v >= 0.0 {
                sign |= 1_u64 << k;
            }
            code[k] = (v * scale).round().clamp(-127.0, 127.0) as i8 as u8;
        }
        (sign, code)
    }

    #[test]
    fn kashin_scored_ranks_aligned_view_above_wrong_heading() {
        let proj = HadamardProjector::new(DEFAULT_KASHIN_SEED);
        let mut q_feat = SparseFeatures {
            pixels: Vec::new(),
            model_pixels: Vec::new(),
            scores: Vec::new(),
            descriptors: Vec::new(),
        };
        let mut aligned = ActiveTrackSet::default();
        let mut wrong = ActiveTrackSet::default();

        for i in 0..24_u64 {
            let d_true = make_unit_desc(100 + i);
            let d_wrong = make_unit_desc(9_000 + i);
            q_feat.pixels.push([(i as f64) * 12.0, 100.0]);
            q_feat.model_pixels.push([(i as f32) * 12.0, 100.0]);
            q_feat.scores.push(0.9);
            q_feat.descriptors.extend_from_slice(&d_true);

            let (s_true, k_true) = encode_i8_sketch(&proj, &d_true);
            aligned
                .world_points_enu
                .push([(i as f64) * 2.0, 0.0, 100.0]);
            aligned.sign_bits.push(s_true);
            aligned.kashin_i8.push(k_true);

            let (s_wrong, k_wrong) = encode_i8_sketch(&proj, &d_wrong);
            wrong.world_points_enu.push([(i as f64) * 2.0, 0.0, 100.0]);
            wrong.sign_bits.push(s_wrong);
            wrong.kashin_i8.push(k_wrong);
        }

        let query = PreparedKashinQuery::from_features(&proj, &q_feat).expect("valid query");
        let (w_ok, _, sim_ok) = match_tracks_kashin_scored(&aligned, &query, 64);
        let (w_bad, _, sim_bad) = match_tracks_kashin_scored(&wrong, &query, 64);
        assert!(
            w_ok.len() >= 20,
            "expected >= 20 mutual matches, got {}",
            w_ok.len()
        );
        assert!(
            w_bad.len() <= 2,
            "expected <= 2 random matches, got {}",
            w_bad.len()
        );
        assert!(
            sim_ok > sim_bad + 40.0,
            "aligned Kashin similarity ({sim_ok}) must exceed wrong heading ({sim_bad})"
        );
    }
}
