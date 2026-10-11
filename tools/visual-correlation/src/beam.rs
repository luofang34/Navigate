//! Two-level pooled-feature search in image lag space, without geographic claims.
use crate::{
    Failure,
    score::{self, Grid, Score},
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, time::Instant};
fn pool(g: &Grid) -> Result<Grid, Failure> {
    if !g.w.is_multiple_of(2) || !g.h.is_multiple_of(2) {
        return Err("pool needs even dimensions".into());
    }
    let w = g.w / 2;
    let h = g.h / 2;
    let mut values = vec![0.0; w * h * g.c];
    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let indices = [
                2 * y * g.w + 2 * x,
                2 * y * g.w + 2 * x + 1,
                (2 * y + 1) * g.w + 2 * x,
                (2 * y + 1) * g.w + 2 * x + 1,
            ];
            let i = y * w + x;
            mask[i] = indices.iter().all(|j| g.mask[*j]);
            if mask[i] {
                for c in 0..g.c {
                    values[i * g.c + c] =
                        indices.iter().map(|j| g.values[j * g.c + c] * 0.25).sum();
                }
            }
        }
    }
    Ok(Grid::new(w, h, g.c, values, mask)?)
}
fn candidates(
    lags: &[[i32; 2]],
    scores: &[Score],
    beam: usize,
    window: i32,
    center: [i32; 2],
    radius: i32,
) -> (Vec<[i32; 2]>, Vec<[i32; 2]>) {
    let mut modes: Vec<[i32; 2]> = Vec::new();
    for i in crate::order(scores) {
        let p = lags[i];
        if modes
            .iter()
            .any(|q| (p[0] - q[0]).abs().max((p[1] - q[1]).abs()) <= 2)
        {
            continue;
        }
        modes.push(p);
        if modes.len() == beam {
            break;
        }
    }
    let mut fine = BTreeSet::new();
    for p in &modes {
        for dy in -window..=window {
            for dx in -window..=window {
                let v = [2 * p[0] + dx, 2 * p[1] + dy];
                if (v[0] - center[0]).abs() <= radius && (v[1] - center[1]).abs() <= radius {
                    fine.insert(v);
                }
            }
        }
    }
    (modes, fine.into_iter().collect())
}
pub fn compare(
    q: &Grid,
    m: &Grid,
    center: [i32; 2],
    radius: i32,
    min: usize,
) -> Result<Value, Failure> {
    let full: Vec<_> = (-radius..=radius)
        .flat_map(|dy| (-radius..=radius).map(move |dx| [center[0] + dx, center[1] + dy]))
        .collect();
    let oracle = score::direct(q, m, &full, min, true);
    let ranks = crate::order(&oracle);
    let best = *ranks.first().ok_or("no full-fine candidate")?;
    let mut rows = Vec::new();
    for repeat in 0..7 {
        for beam in [1, 4] {
            for window in [2, 4] {
                let start = Instant::now();
                let cq = pool(q)?;
                let cm = pool(m)?;
                let pool_ms = start.elapsed().as_secs_f64() * 1000.0;
                let coarse: Vec<_> = full
                    .iter()
                    .filter(|p| p[0] % 2 == 0 && p[1] % 2 == 0)
                    .map(|p| [p[0] / 2, p[1] / 2])
                    .collect();
                let minimum = (min as f64 / 4.0).ceil() as usize;
                let t = Instant::now();
                let scores = score::direct(&cq, &cm, &coarse, minimum, false);
                let coarse_ms = t.elapsed().as_secs_f64() * 1000.0;
                let t = Instant::now();
                let (modes, fine) = candidates(&coarse, &scores, beam, window, center, radius);
                let selection_ms = t.elapsed().as_secs_f64() * 1000.0;
                let t = Instant::now();
                let result = score::direct(q, m, &fine, min, false);
                let fine_ms = t.elapsed().as_secs_f64() * 1000.0;
                let winner = crate::order(&result).first().map(|i| fine[*i]);
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                rows.push(json!({"repeat":repeat,"beam":beam,"fine_window_cells":window,"coarse_modes":modes,"coarse_evaluations":coarse.len(),"unique_fine_evaluations":fine.len(),"full_fine_argmax_retained":fine.contains(&full[best]),"winner_lag":winner,"oracle_lag":full[best],"oracle_score":oracle[best].value,"oracle_top10_retained":ranks.iter().take(10).filter(|i|fine.contains(&full[**i])).count(),"pool_ms":pool_ms,"coarse_score_ms":coarse_ms,"selection_ms":selection_ms,"fine_score_ms":fine_ms,"total_ms":elapsed_ms,"pooled_payload_bytes":(cq.values.len()+cm.values.len())*4+cq.mask.len()+cm.mask.len()}));
            }
        }
    }
    Ok(
        json!({"scope":"image-lag algorithm proxy; pooled features change the coarse score; no geographic beam or independent truth","pool":"2x2 arithmetic mean, all four masks valid, no renormalization","nms_chebyshev_radius_coarse_cells":2,"full_fine_candidates":full.len(),"rows":rows}),
    )
}

#[cfg(test)]
#[path = "beam/tests.rs"]
mod tests;
