//! Reuse traced point estimates to initialize a conditional joint solve.
use super::{ImageTracks, ReconstructionError, SceneTriangulation, fixed_structure, reprojection};
use crate::{CameraModel, LocalSceneCamera, LocalScenePoint};
use nalgebra::Vector3;
use std::collections::BTreeMap;

/// A point estimate in the same arbitrary coordinates as the supplied cameras.
/// The host must retain the source scene and feature association identities.
#[derive(Clone, Debug)]
pub struct ScenePointSeed {
    /// Feature identity in the supplied image graph.
    pub feature_id: u64,
    /// Estimated position, with unknown geometric and registration error.
    pub position: Vector3<f64>,
}

/// Initialize points from triangulation and traced estimates in shared coordinates.
///
/// Strict triangulation takes priority. For each remaining feature, a supplied
/// estimate can initialize a point if at least two source pixels are within
/// 12 pixels of its projection with positive depth. This wider initialization
/// gate does not establish geometric support or accept a camera pose. Run a
/// joint refinement and separate camera checks before using these estimates.
/// No camera, source pixel, or seed position is changed or averaged.
///
/// # Errors
/// Rejects invalid graphs and cameras, and unknown, repeated, or nonfinite seeds.
pub fn initialize_scene_tracks(
    camera: &CameraModel,
    graph: &ImageTracks,
    cameras: &[LocalSceneCamera],
    seeds: &[ScenePointSeed],
) -> Result<SceneTriangulation, ReconstructionError> {
    let mut result = fixed_structure::triangulate_scene_tracks(camera, graph, cameras)?;
    let tracks: BTreeMap<_, _> = graph.tracks.iter().map(|t| (t.feature_id, t)).collect();
    let mut estimates = BTreeMap::new();
    for seed in seeds {
        if !tracks.contains_key(&seed.feature_id)
            || !seed.position.iter().all(|v| v.is_finite())
            || estimates.insert(seed.feature_id, seed.position).is_some()
        {
            return Err(ReconstructionError::Track {
                feature_id: seed.feature_id,
                reason: "unknown or repeated point seed, or nonfinite seed position",
            });
        }
    }
    let poses = fixed_structure::camera_poses(graph, cameras)?;
    result.unresolved_feature_ids.retain(|id| {
        let Some(&position) = estimates.get(id) else {
            return true;
        };
        let Some(track) = tracks.get(id) else {
            return true;
        };
        let observations: Vec<_> = track
            .observations
            .iter()
            .filter(|o| {
                poses.get(&o.camera_index).is_some_and(|&pose| {
                    reprojection::residual(
                        camera,
                        pose,
                        &reprojection::Point {
                            world: position,
                            pixel: o.pixel,
                        },
                    ) <= 12.0
                })
            })
            .cloned()
            .collect();
        if observations.len() < 2 {
            return true;
        }
        result.scene.points.push(LocalScenePoint {
            feature_id: *id,
            position,
            observations,
        });
        false
    });
    Ok(result)
}

#[cfg(test)]
mod tests;
