//! Window proposals use image coordinates and fixed search bounds.
use crate::Error;
use navigate_visual::PixelMatch;
use std::collections::BTreeSet;

pub(super) const SIDE: u32 = 2048;
const STEP: u32 = SIDE / 2;
pub(super) fn seed(pairs: &[PixelMatch], size: (u32, u32)) -> Result<Option<[u32; 4]>, Error> {
    if size.0 < SIDE || size.1 < SIDE {
        return Err(Error::Input(
            "acquisition needs a map at least 2048 by 2048".into(),
        ));
    }
    if pairs.is_empty() {
        return Ok(None);
    }
    let mut xs = Vec::with_capacity(pairs.len());
    let mut ys = Vec::with_capacity(pairs.len());
    for p in pairs {
        if !p.reference.iter().all(|v| v.is_finite()) {
            return Err(Error::Input("nonfinite coarse reference coordinate".into()));
        }
        xs.push(p.reference.x);
        ys.push(p.reference.y);
    }
    let start = |v: &mut [f64], limit: u32| {
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let median = if n.is_multiple_of(2) {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        } else {
            v[n / 2]
        };
        (((median - f64::from(STEP)) / f64::from(STEP)).round_ties_even() * f64::from(STEP))
            .clamp(0.0, f64::from(limit - SIDE)) as u32
    };
    Ok(Some([
        start(&mut xs, size.0),
        start(&mut ys, size.1),
        SIDE,
        SIDE,
    ]))
}
pub(super) fn neighbors(seed: [u32; 4], size: (u32, u32)) -> Vec<[u32; 4]> {
    let mut positions = BTreeSet::new();
    for dy in [-1_i64, 0, 1] {
        for dx in [-1_i64, 0, 1] {
            let x = (i64::from(seed[0]) + dx * i64::from(STEP)).clamp(0, i64::from(size.0 - SIDE))
                as u32;
            let y = (i64::from(seed[1]) + dy * i64::from(STEP)).clamp(0, i64::from(size.1 - SIDE))
                as u32;
            positions.insert([x, y, SIDE, SIDE]);
        }
    }
    let mut result: Vec<_> = positions.into_iter().collect();
    result.sort_by_key(|p| {
        let dx = i64::from(p[0]) - i64::from(seed[0]);
        let dy = i64::from(p[1]) - i64::from(seed[1]);
        (dx * dx + dy * dy, p[1], p[0])
    });
    result
}
pub(super) fn children(parent: [u32; 4]) -> Vec<[u32; 4]> {
    let mut result = Vec::new();
    for y in 0..3_u32 {
        for x in 0..3_u32 {
            result.push([parent[0] + x * 512, parent[1] + y * 512, 1024, 1024]);
        }
    }
    result.sort_by_key(|p| {
        let dx = i64::from(p[0]) - i64::from(parent[0] + 512);
        let dy = i64::from(p[1]) - i64::from(parent[1] + 512);
        (dx * dx + dy * dy, p[1], p[0])
    });
    result
}
#[cfg(test)]
mod tests;
