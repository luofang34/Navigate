//! Map constraints for conditional scene refinement.
use super::{LocalScene, LocalSceneError, SceneRefinement, refine};
use crate::CameraModel;
use nalgebra::Vector3;
use std::collections::BTreeMap;

/// An explicit position constraint for a scene point.
///
/// The solver adds half the squared distance divided by `scale` squared to
/// its robust image cost. This scale is a tuning parameter in scene units.
/// It is not a calibrated uncertainty or an independent measurement.
#[derive(Clone, Copy, Debug)]
pub struct ScenePointPrior {
    /// Identity of the constrained scene point.
    pub feature_id: u64,
    /// Constraint center in the scene coordinate system. Keep this unchanged.
    pub position: Vector3<f64>,
    /// Positive finite penalty scale in scene units.
    pub scale: f64,
}

/// Refine a calibrated scene with explicit map point constraints.
///
/// Each scene component still requires an observed baseline between fixed
/// boundary cameras. An arbitrary coordinate gauge is not supported because
/// constraint centers have fixed scene coordinates. Prior centers must come
/// from the input source, not from a previous fitted point position.
/// This does not accept a navigation fix or establish geographic accuracy.
///
/// # Errors
/// Rejects unknown or duplicate point identities, nonfinite centers, invalid
/// scales, and invalid scenes. Errors leave the scene unchanged.
pub fn refine_regularized_local_scene(
    calibrations: &[CameraModel],
    scene: &mut LocalScene,
    priors: &[ScenePointPrior],
    max_iterations: usize,
) -> Result<SceneRefinement, LocalSceneError> {
    refine(calibrations, scene, max_iterations, None, priors)
}

pub(super) fn validate(
    scene: &LocalScene,
    priors: &[ScenePointPrior],
) -> Result<BTreeMap<u64, ScenePointPrior>, LocalSceneError> {
    let identities: std::collections::BTreeSet<_> =
        scene.points.iter().map(|p| p.feature_id).collect();
    let mut result = BTreeMap::new();
    for prior in priors {
        if !identities.contains(&prior.feature_id)
            || !prior.position.iter().all(|v| v.is_finite())
            || !prior.scale.is_finite()
            || prior.scale <= 0.0
            || !prior.scale.recip().powi(2).is_normal()
            || result.insert(prior.feature_id, *prior).is_some()
        {
            return Err(LocalSceneError::Point {
                feature_id: prior.feature_id,
                reason: "invalid or duplicate position constraint",
            });
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
