//! Scene refinement with a separate pixel grid for each camera.
use super::{LocalScene, LocalSceneError, SceneCoordinateGauge, SceneRefinement, refine};
use crate::CameraModel;

/// Refine poses and points with one fixed calibration for each scene camera.
///
/// Calibrations follow the camera order in `scene`. Each residual keeps the
/// pixel units of its source image. Intrinsics are not optimization variables.
/// A coordinate gauge selects an arbitrary origin and scale. Without a gauge,
/// each connected component needs an observed baseline between fixed cameras.
/// This produces conditional scene geometry, not an accepted navigation fix.
///
/// # Errors
/// Rejects a calibration count mismatch and the same invalid inputs as
/// [`super::refine_local_scene`]. Errors leave the scene unchanged.
pub fn refine_calibrated_local_scene(
    calibrations: &[CameraModel],
    scene: &mut LocalScene,
    max_iterations: usize,
    gauge: Option<SceneCoordinateGauge>,
) -> Result<SceneRefinement, LocalSceneError> {
    refine(calibrations, scene, max_iterations, gauge, &[])
}

#[cfg(test)]
mod tests;
