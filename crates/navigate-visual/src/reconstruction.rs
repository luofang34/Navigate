//! Incremental reconstruction from image links in one arbitrary-scale frame.
//!
//! A host supplies calibrated observations and an estimated seed. This path
//! supports general camera rotations and triangulated surfaces, including a
//! facade. It does not require a homography or rendered terrain depth.
//! Process long videos in bounded local groups. Each call accepts 2 to 129
//! cameras, up to 65,536 tracks, and up to 4,000,000 image observations.
//! The host must retain unresolved seeds, geographic alternatives, and source
//! provenance. This module does not select a global location or update a map.
use crate::local_scene::pose::Pose;
use crate::{CameraModel, ScenePointObservation};
use nalgebra::{Vector2, Vector3};
use std::collections::{BTreeMap, BTreeSet};
mod alignment;
mod builder;
mod dense_flow;
mod error;
mod five_point;
mod fixed_structure;
mod growth;
mod p3p;
mod pair_selection;
mod refinement;
mod registration;
mod reprojection;
mod resection;
mod seeded_structure;
mod seeds;
mod structure;
mod track_merge;
mod triangulation;
mod types;
mod validation;
pub use alignment::{
    AlignmentConfig, PointAlignmentConfig, PointAssociationConfig, SceneAlignment,
    SceneAlignmentError, ScenePointAlignment, ScenePointAlignmentProposals, ScenePointAssociation,
    SceneTransform, align_scenes, align_scenes_with_points, associate_scene_points,
};
pub use builder::DenseTrackBuilder;
pub use error::ReconstructionError;
pub use fixed_structure::{SceneTriangulation, triangulate_scene_tracks};
pub use pair_selection::candidate_seed_pairs;
pub use registration::{
    RegistrationConfig, SceneMapAssociation, SceneRegistration, SceneRegistrationError,
    SceneRegistrationProposals, propose_scene_registration,
};
pub use resection::{SceneCameraFit, ScenePointMatch, SceneResectionError, refit_scene_camera};
pub use seeded_structure::{ScenePointSeed, initialize_scene_tracks};
pub use seeds::propose_seeds;
pub use track_merge::{
    ImageTrackMerger, MergedTrackSelection, SceneTrackLinks, TrackMergeError, TrackSource,
};
pub use types::{ImageTrack, ImageTracks, Reconstruction, ReconstructionSeed, SeedProposal};
struct State {
    poses: BTreeMap<usize, Pose>,
    points: BTreeMap<usize, Vector3<f64>>,
}
fn pixel(track: &ImageTrack, camera: usize) -> Option<Vector2<f64>> {
    track
        .observations
        .iter()
        .find(|o| o.camera_index == camera)
        .map(|o| o.pixel)
}
/// Reconstruct one conditional scene. The input graph remains unchanged.
///
/// The first seed pose defines arbitrary coordinates. Its baseline length fixes
/// arbitrary scale; the second camera rotation and baseline direction stay free.
/// No pose, scene point, or calibration is treated as ground truth.
///
/// # Errors
/// Rejects invalid inputs, unsupported initialization, or failed refinement.
pub fn reconstruct(
    camera: &CameraModel,
    graph: &ImageTracks,
    seed: ReconstructionSeed,
) -> Result<Reconstruction, ReconstructionError> {
    validation::graph(camera, graph)?;
    validation::seed(graph, seed)?;
    let points = seeds::points(camera, graph, seed);
    if points.len() < 20 {
        return Err(ReconstructionError::Support {
            camera_indices: seed.camera_indices,
            points: points.len(),
        });
    }
    let [a, b] = seed.camera_indices;
    let mut state = State {
        poses: BTreeMap::from([
            (a, seeds::identity()),
            (b, Pose::from_scene(seed.second_pose)),
        ]),
        points,
    };
    refinement::run(camera, graph, &mut state, seed.camera_indices, 20)?;
    growth::run(camera, graph, &mut state, seed.camera_indices)?;
    refinement::run(camera, graph, &mut state, seed.camera_indices, 60)
}
#[cfg(test)]
mod tests;
