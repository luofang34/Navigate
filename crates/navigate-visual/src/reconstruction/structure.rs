//! Triangulate persistent image links against the supported local cameras.
use super::*;
use super::{reprojection::Point, triangulation::ray};
pub(super) fn observations(
    camera: &CameraModel,
    track: &ImageTrack,
    poses: &BTreeMap<usize, Pose>,
    world: Vector3<f64>,
) -> Vec<ScenePointObservation> {
    track
        .observations
        .iter()
        .filter(|o| {
            poses.get(&o.camera_index).is_some_and(|&pose| {
                reprojection::residual(
                    camera,
                    pose,
                    &Point {
                        world,
                        pixel: o.pixel,
                    },
                ) <= 3.0
            })
        })
        .cloned()
        .collect()
}
pub(super) fn triangulate(
    camera: &CameraModel,
    track: &ImageTrack,
    poses: &BTreeMap<usize, Pose>,
) -> Option<Vector3<f64>> {
    let seen: Vec<_> = track
        .observations
        .iter()
        .filter_map(|o| poses.get(&o.camera_index).map(|&p| (o, p)))
        .collect();
    let mut best = None;
    let mut angle = 0.7;
    for (i, (a, pa)) in seen.iter().enumerate() {
        for (b, pb) in &seen[i + 1..] {
            let Some((world, parallax)) =
                reprojection::triangulate(*pa, *pb, ray(camera, a.pixel), ray(camera, b.pixel))
            else {
                continue;
            };
            if parallax > angle
                && reprojection::residual(
                    camera,
                    *pa,
                    &Point {
                        world,
                        pixel: a.pixel,
                    },
                ) < 2.5
                && reprojection::residual(
                    camera,
                    *pb,
                    &Point {
                        world,
                        pixel: b.pixel,
                    },
                ) < 2.5
            {
                angle = parallax;
                best = Some(world);
            }
        }
    }
    let world = best?;
    let support = observations(camera, track, poses, world);
    (support.len() >= 2 && support.len() * 4 >= seen.len() * 3).then_some(world)
}
pub(super) fn add_points(
    camera: &CameraModel,
    graph: &ImageTracks,
    state: &mut State,
    frame: usize,
) {
    for (id, track) in graph.tracks.iter().enumerate() {
        if !state.points.contains_key(&id)
            && pixel(track, frame).is_some()
            && let Some(world) = triangulate(camera, track, &state.poses)
        {
            state.points.insert(id, world);
        }
    }
}
