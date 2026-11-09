//! Identification without an attitude prior: the pyramid search.
//!
//! The search takes three bright centroids (a triangle) and finds the
//! catalogue triangles with the same three side angles in the
//! [`PairCatalogue`]. A catalogue triangle must also have the same
//! handedness: a mirror image of the sky has the same side angles but the
//! opposite sign of the triple product. A fourth centroid then confirms
//! one catalogue triangle with a unique fourth star (the pyramid). A
//! triangle with more than one confirmed catalogue triangle is ambiguous,
//! and the search goes to the next triangle.
//!
//! The triangles come in the order of Mortari: the index steps between the
//! three centroids grow slowly, so one false centroid does not block many
//! triangles in sequence. The search is bounded by the number of
//! centroids and by the candidate limit of each triangle.

use nalgebra::Vector3;

use crate::PairCatalogue;

/// A centroid index with its table star index.
pub(crate) type Pairing = (usize, u16);

/// The limits of the search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Search {
    /// Largest difference of a measured angle and a table angle.
    pub(crate) tolerance_rad: f64,
    /// Most catalogue triangles of one centroid triangle.
    pub(crate) max_candidates: usize,
}

/// The first pyramid that `accept` takes, from the measured unit
/// directions `rays` with their centroid indices.
pub(crate) fn search<T>(
    rays: &[(usize, Vector3<f64>)],
    table: &PairCatalogue,
    limits: Search,
    mut accept: impl FnMut([Pairing; 4]) -> Option<T>,
) -> Option<T> {
    let n = rays.len();
    for dj in 1..n {
        for dk in 1..n.saturating_sub(dj) {
            for i in 0..n.saturating_sub(dj + dk) {
                let triangle = [i, i + dj, i + dj + dk];
                if let Some(pyramid) = identify_triangle(rays, triangle, table, limits)
                    && let Some(result) = accept(pyramid)
                {
                    return Some(result);
                }
            }
        }
    }
    None
}

/// The unique confirmed pyramid of the centroid triangle `triangle`.
fn identify_triangle(
    rays: &[(usize, Vector3<f64>)],
    triangle: [usize; 3],
    table: &PairCatalogue,
    limits: Search,
) -> Option<[Pairing; 4]> {
    let [Some((ci, ri)), Some((cj, rj)), Some((ck, rk))] = triangle.map(|t| rays.get(t).copied())
    else {
        return None;
    };
    let sides = [ri.angle(&rj), rj.angle(&rk), ri.angle(&rk)];
    let longest = sides.iter().copied().fold(0.0, f64::max);
    let det = ri.dot(&rj.cross(&rk));
    // A thin triangle has no reliable handedness.
    if longest > table.max_angle_rad() || det.abs() < 3.0 * limits.tolerance_rad * longest {
        return None;
    }
    let candidates = triangle_candidates(table, sides, limits)?;
    let mut confirmed = None;
    for [a, b, c] in candidates {
        let units = [a, b, c].map(|s| table.star(s).map(|t| t.unit));
        let [Some(ea), Some(eb), Some(ec)] = units else {
            continue;
        };
        if ea.dot(&eb.cross(&ec)).signum() != det.signum() {
            continue;
        }
        let fourth = rays.iter().enumerate().find_map(|(r, (cr, rr))| {
            if triangle.contains(&r) {
                return None;
            }
            let angles = [ri.angle(rr), rj.angle(rr), rk.angle(rr)];
            confirm(table, [a, b, c], angles, limits.tolerance_rad).map(|d| (*cr, d))
        });
        if let Some(fourth) = fourth {
            if confirmed.is_some() {
                return None;
            }
            confirmed = Some([(ci, a), (cj, b), (ck, c), fourth]);
        }
    }
    confirmed
}

/// The table triangles `[a, b, c]` with the side angles `sides` (`ab`,
/// `bc`, `ac`). More candidates than the limit give `None`.
fn triangle_candidates(
    table: &PairCatalogue,
    sides: [f64; 3],
    limits: Search,
) -> Option<Vec<[u16; 3]>> {
    let [ab, bc, ac] = sides;
    let tol = limits.tolerance_rad;
    let mut by_first: Vec<(u16, u16)> = table
        .within(bc, tol)
        .iter()
        .flat_map(|p| [(p.first, p.second), (p.second, p.first)])
        .collect();
    by_first.sort_unstable();
    let mut out = Vec::new();
    for p in table.within(ab, tol) {
        for (a, b) in [(p.first, p.second), (p.second, p.first)] {
            let start = by_first.partition_point(|e| e.0 < b);
            let tail = by_first.get(start..).unwrap_or(&[]);
            for &(_, c) in tail.iter().take_while(|e| e.0 == b) {
                let fits = table
                    .angle(a, c)
                    .is_some_and(|angle| (angle - ac).abs() <= tol);
                if c != a && fits {
                    if out.len() == limits.max_candidates {
                        return None;
                    }
                    out.push([a, b, c]);
                }
            }
        }
    }
    Some(out)
}

/// The unique table star with the angles `angles` to the stars of
/// `triangle`.
fn confirm(
    table: &PairCatalogue,
    triangle: [u16; 3],
    angles: [f64; 3],
    tolerance_rad: f64,
) -> Option<u16> {
    let [a, b, c] = triangle;
    let [to_a, to_b, to_c] = angles;
    let near = |x: u16, d: u16, angle: f64| {
        table
            .angle(x, d)
            .is_some_and(|v| (v - angle).abs() <= tolerance_rad)
    };
    let mut found = None;
    for p in table.within(to_a, tolerance_rad) {
        let d = match (p.first == a, p.second == a) {
            (true, _) => p.second,
            (_, true) => p.first,
            _ => continue,
        };
        if triangle.contains(&d) || !near(b, d, to_b) || !near(c, d, to_c) {
            continue;
        }
        if found.is_some_and(|f| f != d) {
            return None;
        }
        found = Some(d);
    }
    found
}

#[cfg(test)]
mod tests;
