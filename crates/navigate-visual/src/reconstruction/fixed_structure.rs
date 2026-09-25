//! Recover conditional scene points without changing the supplied camera poses.
use super::{ImageTracks, Pose, ReconstructionError, structure, validation};
use crate::{CameraModel, CameraPose, LocalScene, LocalSceneCamera, LocalScenePoint};
use std::collections::BTreeMap;

/// Point estimates supported by image tracks and fixed camera estimates.
#[derive(Clone, Debug)]
pub struct SceneTriangulation {
    /// Conditional points and unchanged camera estimates in arbitrary scene units.
    /// The camera `fixed` flags are retained for a separate refinement request.
    pub scene: LocalScene,
    /// Features without sufficient parallax or consistent image support.
    pub unresolved_feature_ids: Vec<u64>,
}

/// Triangulate image tracks against one set of supplied camera estimates.
///
/// Camera order and source digests must match the graph. All camera positions
/// and orientations stay unchanged. Each point requires positive depth,
/// sufficient parallax, and support from at least three quarters of its images.
/// The result is conditional on estimated poses and calibration. It does not
/// measure uncertainty, add independent evidence, or accept a geographic pose.
///
/// # Errors
/// Rejects invalid graphs, invalid poses, or source identity mismatches.
pub fn triangulate_scene_tracks(
    camera: &CameraModel,
    graph: &ImageTracks,
    cameras: &[LocalSceneCamera],
) -> Result<SceneTriangulation, ReconstructionError> {
    validation::graph(camera, graph)?;
    let poses = camera_poses(graph, cameras)?;
    let mut points = Vec::new();
    let mut unresolved_feature_ids = Vec::new();
    for track in &graph.tracks {
        if let Some(position) = structure::triangulate(camera, track, &poses) {
            points.push(LocalScenePoint {
                feature_id: track.feature_id,
                position,
                observations: structure::observations(camera, track, &poses, position),
            });
        } else {
            unresolved_feature_ids.push(track.feature_id);
        }
    }
    Ok(SceneTriangulation {
        scene: LocalScene {
            cameras: cameras.to_vec(),
            points,
        },
        unresolved_feature_ids,
    })
}

pub(super) fn camera_poses(
    graph: &ImageTracks,
    cameras: &[LocalSceneCamera],
) -> Result<BTreeMap<usize, Pose>, ReconstructionError> {
    if cameras.len() != graph.observation_sha256.len() {
        return Err(ReconstructionError::CameraCount {
            supplied: cameras.len(),
            required: graph.observation_sha256.len(),
        });
    }
    cameras
        .iter()
        .enumerate()
        .map(|(index, c)| {
            if !c
                .observation_sha256
                .eq_ignore_ascii_case(&graph.observation_sha256[index])
            {
                return Err(ReconstructionError::Observation {
                    index,
                    reason: "camera digest does not match the source graph",
                });
            }
            CameraPose {
                position: c.pose.position,
                orientation: c.pose.orientation,
            }
            .validate()
            .map_err(|source| ReconstructionError::SceneCamera { index, source })?;
            Ok((index, Pose::from_scene(c.pose)))
        })
        .collect()
}

#[cfg(test)]
pub(super) mod tests;
