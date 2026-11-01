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
    let n_r = tracks.len().min(cap);
    let q_valid = query.valid_indices();
    let n_q = q_valid.len().min(cap);
    if n_r == 0 || n_q == 0 {
        return (Vec::new(), Vec::new());
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
    for (r, &(q, b1, b2)) in fwd.iter().enumerate() {
        if q < n_q
            && rev[q].0 == r
            && b1 >= 0.62
            && (b2 <= -1.5 || (1.0 - b1).max(0.0) <= 0.85 * (1.0 - b2).max(0.0))
        {
            world.push(tracks.world_points_enu[r]);
            px.push(q_px[q_valid[q]]);
        }
    }
    (world, px)
}

/// Match virtual corridor landmarks against a query frame.
#[must_use]
pub fn match_virtual_kashin(
    virt: &VirtualCorridorView,
    query: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let probe = ActiveTrackSet {
        world_points_enu: virt.world_points_enu.iter().take(cap).copied().collect(),
        sign_bits: virt.sign_bits.iter().take(cap).copied().collect(),
        kashin_i8: virt.kashin_i8.iter().take(cap).copied().collect(),
    };
    match_tracks_kashin(&probe, query, cap)
}
