//! Observation links and conditional reconstruction results.
use crate::{
    LocalScene, LocalScenePose, SceneCoordinateGauge, ScenePointObservation, SceneRefinement,
};
/// One proposed physical feature across images. Associations remain estimates.
#[derive(Clone, Debug)]
pub struct ImageTrack {
    /// Stable identity assigned by the host or tracker, within this graph.
    pub feature_id: u64,
    /// Camera indices refer to [`ImageTracks::observation_sha256`].
    pub observations: Vec<ScenePointObservation>,
}
/// A bounded group of calibrated observations with proposed feature links.
///
/// Supply undistorted pixels with one common camera model. The host retains
/// source stream, calibration, matcher, and shared-evidence provenance.
#[derive(Clone, Debug)]
pub struct ImageTracks {
    /// Digests from [`crate::Frame::evidence_sha256`], in source camera order.
    pub observation_sha256: Vec<String>,
    /// Links from any image matcher or point tracker. No depth is required.
    pub tracks: Vec<ImageTrack>,
}
/// One conditional initialization. Alternative seeds require separate solves.
#[derive(Clone, Copy, Debug)]
pub struct ReconstructionSeed {
    /// Two distinct indices in the input graph. The first camera defines the
    /// scene origin with right, down, and forward axes.
    pub camera_indices: [usize; 2],
    /// Second camera pose in that scene frame. Translation sets arbitrary scale.
    /// Eye axes are right, up, and back, as for [`LocalScenePose`].
    pub second_pose: LocalScenePose,
}
impl ReconstructionSeed {
    /// Express two estimated poses in the first camera's local coordinate frame.
    /// This transfers an arbitrary scale; it does not establish metric accuracy.
    ///
    /// # Errors
    /// Rejects repeated indices, invalid poses, and a zero camera baseline.
    pub fn between(
        camera_indices: [usize; 2],
        poses: [LocalScenePose; 2],
    ) -> Result<Self, super::ReconstructionError> {
        if camera_indices[0] == camera_indices[1]
            || poses.iter().any(|p| {
                !p.position
                    .iter()
                    .chain(p.orientation.coords.iter())
                    .all(|v| v.is_finite())
                    || (p.orientation.norm_squared() - 1.0).abs() > 1e-8
            })
            || (poses[0].position - poses[1].position).norm() <= 1e-6
        {
            return Err(super::ReconstructionError::Seed {
                camera_indices,
                reason: "two finite estimated poses with a nonzero baseline are required",
            });
        }
        let first = crate::local_scene::pose::Pose::from_scene(poses[0]);
        Ok(Self {
            camera_indices,
            second_pose: LocalScenePose {
                position: first.r * poses[1].position + first.t,
                orientation: nalgebra::UnitQuaternion::from_matrix(&first.r) * poses[1].orientation,
            },
        })
    }
}
/// A camera-pair proposal with geometric support, not geographic acceptance.
#[derive(Clone, Copy, Debug)]
pub struct SeedProposal {
    /// A conditional pose that can initialize a separate reconstruction.
    pub seed: ReconstructionSeed,
    /// Points with positive depth, sufficient parallax, and small image error.
    /// This count is not a probability or evidence of a unique pose.
    pub triangulated_points: usize,
}
/// One local result. Missing cameras remain unresolved.
///
/// Scale, calibration error, geographic registration, and correlations remain
/// unknown. The scene is estimated geometry. It is not independently verified
/// scene data, a navigation fix, or a reference-data update.
#[derive(Clone, Debug)]
pub struct Reconstruction {
    /// Refined cameras and a bounded selection of their supporting points.
    /// The host retains the complete source graph.
    pub scene: LocalScene,
    /// For each scene camera, its index in the input graph.
    pub source_camera_indices: Vec<usize>,
    /// Input cameras without geometric support. Do not interpolate acceptance.
    pub unresolved_camera_indices: Vec<usize>,
    /// Arbitrary origin and baseline length used during joint refinement.
    pub coordinate_gauge: SceneCoordinateGauge,
    /// Final optimizer progress. This does not establish uncertainty.
    pub refinement: SceneRefinement,
}
