//! Conditional registration from estimated local points to rendered map surfaces.
use super::SceneTransform;
use crate::{CameraModel, LocalScene, PixelMatch, ReferenceView};
mod consensus;
mod fit;
mod links;
mod types;
pub use types::{
    RegistrationConfig, SceneMapAssociation, SceneRegistration, SceneRegistrationError,
    SceneRegistrationProposals,
};
/// Propose scale, rotation, and translation from scene-to-map point associations.
///
/// Pixel matches can come from any image matcher. The reference must use the
/// supplied camera calibration. Only finite positive rendered depths contribute.
/// The scene and reference remain unchanged. This function does not enforce a
/// navigation prior or accept a geographic location. Inspect every returned
/// alternative and preserve the source and map identities when reprocessing.
///
/// # Errors
/// Rejects invalid inputs, repeated evidence, invalid reference data, and too few
/// distinct depth associations. Valid links without a supported transform return
/// an empty candidate list. This does not prove that the scene is unlocatable.
pub fn propose_scene_registration(
    camera: &CameraModel,
    scene: &LocalScene,
    observation_sha256: &str,
    reference: &ReferenceView,
    matches: &[PixelMatch],
    config: RegistrationConfig,
) -> Result<SceneRegistrationProposals, SceneRegistrationError> {
    validate_config(config)?;
    reference
        .validate_camera(camera)
        .map_err(|source| SceneRegistrationError::Reference { source })?;
    let associations = links::associate(
        camera,
        scene,
        observation_sha256,
        reference,
        matches,
        config,
    )?;
    if associations.len() < config.min_inliers {
        return Err(SceneRegistrationError::Support {
            found: associations.len(),
            required: config.min_inliers,
        });
    }
    let (candidates, candidate_budget_exhausted) =
        consensus::propose(camera, &associations, config);
    Ok(SceneRegistrationProposals {
        candidates,
        observation_sha256: observation_sha256.to_ascii_lowercase(),
        map: reference.map.clone(),
        frame: reference.frame,
        associations,
        candidate_budget_exhausted,
    })
}
fn validate_config(c: RegistrationConfig) -> Result<(), SceneRegistrationError> {
    if !c.association_radius_px.is_finite()
        || c.association_radius_px <= 0.0
        || !c.inlier_threshold_m.is_finite()
        || c.inlier_threshold_m <= 0.0
        || !(6..=4096).contains(&c.min_inliers)
        || !(1..=12).contains(&c.min_occupied_cells)
        || !(1..=16384).contains(&c.trials)
        || !(1..=32).contains(&c.max_candidates)
    {
        return Err(SceneRegistrationError::Config);
    }
    Ok(())
}
#[cfg(test)]
mod tests;
