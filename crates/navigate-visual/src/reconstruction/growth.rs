//! Register views by shared scene support. Unsupported views remain gaps.
use super::reprojection::Point;
use super::*;
fn next_pose(
    camera: &CameraModel,
    graph: &ImageTracks,
    state: &State,
    frame: usize,
) -> Option<Pose> {
    let points: Vec<_> = state
        .points
        .iter()
        .filter_map(|(id, &world)| {
            Some(Point {
                world,
                pixel: pixel(&graph.tracks[*id], frame)?,
            })
        })
        .collect();
    let initial = *state.poses.iter().min_by_key(|(i, _)| i.abs_diff(frame))?.1;
    let proposal = p3p::estimate(camera, &points);
    let (pose, ids) = proposal
        .and_then(|p| reprojection::fit(camera, &points, p))
        .or_else(|| reprojection::fit(camera, &points, initial))?;
    let cells: BTreeSet<_> = ids
        .iter()
        .map(|&i| {
            (
                (points[i].pixel.x * 6.0 / f64::from(camera.width)) as i32,
                (points[i].pixel.y * 4.0 / f64::from(camera.height)) as i32,
            )
        })
        .collect();
    (cells.len() >= 4).then_some(pose)
}
fn next(
    camera: &CameraModel,
    graph: &ImageTracks,
    state: &State,
    pending: &BTreeSet<usize>,
) -> Option<(usize, Pose)> {
    let mut counts = BTreeMap::<usize, usize>::new();
    for id in state.points.keys() {
        for o in &graph.tracks[*id].observations {
            if pending.contains(&o.camera_index) {
                let n = counts.entry(o.camera_index).or_default();
                *n = n.wrapping_add(1);
            }
        }
    }
    let mut order: Vec<_> = counts.into_iter().collect();
    order.sort_by_key(|&(frame, count)| (std::cmp::Reverse(count), frame));
    order
        .into_iter()
        .filter(|&(_, n)| n >= 20)
        .find_map(|(frame, _)| next_pose(camera, graph, state, frame).map(|p| (frame, p)))
}
pub(super) fn run(
    camera: &CameraModel,
    graph: &ImageTracks,
    state: &mut State,
    seed: [usize; 2],
) -> Result<(), ReconstructionError> {
    let mut pending: BTreeSet<_> = (0..graph.observation_sha256.len())
        .filter(|i| !state.poses.contains_key(i))
        .collect();
    let mut retries = 0_usize;
    while !pending.is_empty() {
        if let Some((frame, pose)) = next(camera, graph, state, &pending) {
            state.poses.insert(frame, pose);
            pending.remove(&frame);
            structure::add_points(camera, graph, state, frame);
            if state.poses.len().is_multiple_of(8) {
                refinement::run(camera, graph, state, seed, 20)?;
            }
        } else {
            if retries == 3 {
                break;
            }
            refinement::run(camera, graph, state, seed, 20)?;
            retries = retries.wrapping_add(1);
            state.points = graph
                .tracks
                .iter()
                .enumerate()
                .filter_map(|(id, t)| {
                    structure::triangulate(camera, t, &state.poses).map(|p| (id, p))
                })
                .collect();
        }
    }
    Ok(())
}
