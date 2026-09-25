//! Keep each physically supported two-view alternative separate.
use super::*;
use super::{
    reprojection::Point,
    triangulation::{Rays, ray},
};
pub(super) fn identity() -> Pose {
    Pose {
        r: nalgebra::Matrix3::identity(),
        t: Vector3::zeros(),
    }
}
pub(super) fn points(
    camera: &CameraModel,
    graph: &ImageTracks,
    seed: ReconstructionSeed,
) -> BTreeMap<usize, Vector3<f64>> {
    let second = Pose::from_scene(seed.second_pose);
    graph
        .tracks
        .iter()
        .enumerate()
        .filter_map(|(id, track)| {
            let a = pixel(track, seed.camera_indices[0])?;
            let b = pixel(track, seed.camera_indices[1])?;
            let (world, angle) =
                reprojection::triangulate(identity(), second, ray(camera, a), ray(camera, b))?;
            (angle >= 1.0
                && reprojection::residual(camera, identity(), &Point { world, pixel: a }) < 2.5
                && reprojection::residual(camera, second, &Point { world, pixel: b }) < 2.5)
                .then_some((id, world))
        })
        .collect()
}
/// Propose relative poses with a five-point solver. This does not select a unique
/// seed or a geographic location. Other algorithms can supply [`ReconstructionSeed`].
///
/// # Errors
/// Rejects invalid inputs or a camera pair without a fitted essential matrix.
pub fn propose_seeds(
    camera: &CameraModel,
    graph: &ImageTracks,
    camera_indices: [usize; 2],
) -> Result<Vec<SeedProposal>, ReconstructionError> {
    validation::graph(camera, graph)?;
    validation::pair(graph, camera_indices)?;
    let rays: Vec<_> = graph
        .tracks
        .iter()
        .filter_map(|track| {
            Some(Rays {
                a: ray(camera, pixel(track, camera_indices[0])?),
                b: ray(camera, pixel(track, camera_indices[1])?),
            })
        })
        .collect();
    let (matrix, _) = five_point::estimate(camera, &rays, camera_indices)?;
    let alternatives = triangulation::essential_poses(matrix).ok_or(ReconstructionError::Seed {
        camera_indices,
        reason: "invalid essential matrix decomposition",
    })?;
    Ok(alternatives
        .into_iter()
        .filter_map(|(r, t)| {
            let seed = ReconstructionSeed {
                camera_indices,
                second_pose: Pose { r, t }.to_scene(),
            };
            let triangulated_points = points(camera, graph, seed).len();
            (triangulated_points >= 20).then_some(SeedProposal {
                seed,
                triangulated_points,
            })
        })
        .collect())
}
