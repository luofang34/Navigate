//! Inputs and results for a conditional local scene refinement.
use nalgebra::{UnitQuaternion, Vector2, Vector3};

/// An estimated camera pose in one arbitrary-scale scene frame.
///
/// Scene units are not metres. A separate geographic registration is required.
/// Pose, calibration, and scene errors can be correlated and remain unknown.
#[derive(Clone, Copy, Debug)]
pub struct LocalScenePose {
    /// Camera centre in the common scene units.
    pub position: Vector3<f64>,
    /// Eye-to-scene rotation. Eye axes are right, up, and back.
    pub orientation: UnitQuaternion<f64>,
}
/// An image pose and its source identity in a local refinement problem.
#[derive(Clone, Debug)]
pub struct LocalSceneCamera {
    /// Digest returned by [`crate::Frame::evidence_sha256`].
    /// The host retains capture-stream identity and shared-evidence provenance.
    pub observation_sha256: String,
    /// Conditional camera estimate.
    pub pose: LocalScenePose,
    /// Hold this estimate constant during this refinement.
    /// This conditions the solve on an estimate; it does not make it ground truth.
    pub fixed: bool,
}
/// One observation of a persistent scene point.
#[derive(Clone, Debug)]
pub struct ScenePointObservation {
    /// Index in [`LocalScene::cameras`].
    pub camera_index: usize,
    /// Undistorted image coordinates. Pixel centres have integer coordinates.
    pub pixel: Vector2<f64>,
}
/// An estimated local point with its retained image observations.
#[derive(Clone, Debug)]
pub struct LocalScenePoint {
    /// Stable point identity within this scene. Repeated IDs are rejected.
    pub feature_id: u64,
    /// Position in the same scene units as the cameras.
    pub position: Vector3<f64>,
    /// Image support, including observations at fixed boundary cameras.
    /// Repeated observations from the same camera are rejected.
    pub observations: Vec<ScenePointObservation>,
}
/// One conditional local reconstruction candidate.
///
/// This is temporary scene geometry, not a map update or an accepted location.
/// Keep separate geographic alternatives in separate candidates. Refinement
/// cannot make repeated image evidence independent or establish covariance.
#[derive(Clone, Debug)]
pub struct LocalScene {
    /// Camera estimates and exact processing identities.
    pub cameras: Vec<LocalSceneCamera>,
    /// Scene estimates with their original observation links.
    pub points: Vec<LocalScenePoint>,
}
/// Numerical progress of a local refinement; not a pose acceptance result.
#[derive(Clone, Copy, Debug)]
pub struct SceneRefinement {
    /// Initial robust image residual cost, in the declared pixel coordinates.
    pub initial_cost: f64,
    /// Final cost from the same observations and loss function.
    pub final_cost: f64,
    /// Optimizer steps that reduced the objective.
    /// More steps do not add evidence or increase confidence.
    pub steps: usize,
}
