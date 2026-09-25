//! Retrieve a bounded list of initialization pairs from image associations.
use super::*;
fn score(camera: &CameraModel, graph: &ImageTracks, pair: [usize; 2]) -> Option<f64> {
    let mut motion = Vec::new();
    let mut cells = BTreeSet::new();
    let mut spanning = 0_usize;
    for track in &graph.tracks {
        let (Some(a), Some(b)) = (pixel(track, pair[0]), pixel(track, pair[1])) else {
            continue;
        };
        motion.push((b - a).norm());
        cells.insert((
            (a.x * 6.0 / f64::from(camera.width)) as i32,
            (a.y * 4.0 / f64::from(camera.height)) as i32,
        ));
        if track.observations.len() >= graph.observation_sha256.len() / 3 {
            spanning = spanning.wrapping_add(1);
        }
    }
    if motion.len() < 20 || cells.len() < 4 {
        return None;
    }
    motion.sort_by(f64::total_cmp);
    let median = motion[motion.len() / 2];
    (median >= 3.0).then_some(
        (motion.len() as f64 + spanning as f64)
            * median.min(f64::from(camera.width) * 0.15)
            * cells.len() as f64,
    )
}
/// Retrieve camera pairs with shared image coverage and visible motion.
///
/// The first proposals cover separate temporal parts of the group. This avoids
/// spending the complete initial solve budget on one visible surface.
/// This only schedules two-view geometry. Scores stay internal and do not
/// express pose confidence. Other algorithms may supply their own seed pairs.
/// Unexamined pairs and unsupported geometry remain unresolved.
///
/// # Errors
/// Rejects invalid calibrated image links and resource bounds.
pub fn candidate_seed_pairs(
    camera: &CameraModel,
    graph: &ImageTracks,
) -> Result<Vec<[usize; 2]>, ReconstructionError> {
    validation::graph(camera, graph)?;
    let count = graph.observation_sha256.len();
    let stride = (count / 8).max(1);
    let mut pairs = BTreeSet::new();
    for first in (0..count - 1).step_by(stride) {
        for gap in [2, 4, 8, 16, 24] {
            let second = (first + gap).min(count - 1);
            if first != second {
                pairs.insert([first, second]);
            }
        }
    }
    let mut ranked: Vec<_> = pairs
        .into_iter()
        .filter_map(|pair| score(camera, graph, pair).map(|score| (pair, score)))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut covered = BTreeSet::new();
    let mut first = Vec::new();
    let mut remaining = Vec::new();
    for (pair, _) in ranked {
        let part = (pair[0] + pair[1]) * 3 / (2 * count);
        if covered.insert(part) {
            first.push(pair);
        } else {
            remaining.push(pair);
        }
    }
    first.extend(remaining);
    Ok(first)
}

#[cfg(test)]
mod tests;
