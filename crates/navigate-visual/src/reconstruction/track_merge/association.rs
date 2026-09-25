//! Unique pixel agreement across shared source images.
use super::{BTreeMap, BTreeSet, Track, Vector2};
const RADIUS: f64 = 1.0;
type Grid = BTreeMap<(usize, i32, i32), Vec<(usize, Vector2<f64>)>>;
fn cell(pixel: Vector2<f64>) -> (i32, i32) {
    (pixel.x.floor() as i32, pixel.y.floor() as i32)
}
pub(super) fn grid(tracks: &[Track], previous: &[usize], shared: &BTreeSet<usize>) -> Grid {
    let mut grid = Grid::new();
    for &index in previous {
        for (&camera, &pixel) in &tracks[index].pixels {
            if shared.contains(&camera) {
                let (x, y) = cell(pixel);
                grid.entry((camera, x, y)).or_default().push((index, pixel));
            }
        }
    }
    grid
}
fn nearest(grid: &Grid, camera: usize, pixel: Vector2<f64>) -> Option<usize> {
    let (x, y) = cell(pixel);
    let mut found = Vec::new();
    for dx in -1..=1 {
        for dy in -1..=1 {
            if let Some(entries) = grid.get(&(camera, x.saturating_add(dx), y.saturating_add(dy))) {
                for &(id, point) in entries {
                    let distance = (point - pixel).norm();
                    if distance <= RADIUS {
                        found.push((id, distance));
                    }
                }
            }
        }
    }
    found.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let first = *found.first()?;
    if found.get(1).is_some_and(|next| next.1 - first.1 <= 1e-6) {
        return None;
    }
    Some(first.0)
}
pub(super) fn candidate(
    tracks: &[Track],
    grid: &Grid,
    pixels: &BTreeMap<usize, Vector2<f64>>,
) -> Option<usize> {
    let mut votes = BTreeMap::<usize, usize>::new();
    for (&camera, &pixel) in pixels {
        if let Some(index) = nearest(grid, camera, pixel) {
            let count = votes.entry(index).or_default();
            *count = count.saturating_add(1);
        }
    }
    let mut eligible = votes.into_iter().filter_map(|(index, count)| {
        (count >= 2
            && pixels.iter().all(|(camera, pixel)| {
                tracks[index]
                    .pixels
                    .get(camera)
                    .is_none_or(|previous| (previous - pixel).norm() <= RADIUS)
            }))
        .then_some(index)
    });
    let first = eligible.next()?;
    eligible.next().is_none().then_some(first)
}
