//! Conditional refinement of temporary local scene geometry.
//!
//! A matcher or point tracker supplies observation links. This module refines
//! initialized poses and points. It does not find correspondences, initialize a
//! reconstruction, accept a navigation fix, or update reference data. Keep each
//! geographic alternative separate. Fixed boundary poses remain estimates.
use crate::CameraModel;
use std::collections::BTreeSet;
mod bundle;
mod error;
mod pose;
mod scale;
mod types;
mod validation;
pub use error::LocalSceneError;
pub use types::{
    LocalScene, LocalSceneCamera, LocalScenePoint, LocalScenePose, SceneCoordinateGauge,
    ScenePointObservation, SceneRefinement,
};

/// Refine one initialized scene candidate with a common estimated calibration.
///
/// Retain observations from boundary cameras when a local window moves. Each
/// connected component must contain an observed baseline between fixed cameras.
/// Those cameras condition the calculation; they are not treated as measurements
/// with zero error. The result has no covariance or independence claim.
///
/// All source identities and observation links remain unchanged. Only free
/// camera poses and point positions can change. Errors leave the input unchanged.
/// Run separate geometric and navigation acceptance checks on the result.
///
/// # Errors
/// Rejects invalid or repeated observations, unconstrained scene components,
/// more than 96 free cameras or 16,384 points, iteration counts outside 1 through
/// 100, and numerical failures. Fixed cameras without observations add no support.
pub fn refine_local_scene(
    camera: &CameraModel,
    scene: &mut LocalScene,
    max_iterations: usize,
) -> Result<SceneRefinement, LocalSceneError> {
    refine(camera, scene, max_iterations, None)
}

/// Refine one local scene under an arbitrary coordinate origin and scale.
///
/// The gauge camera pair must have a nonzero observed baseline. Only the origin
/// camera is fixed. The scale camera can rotate and change the baseline direction.
/// The baseline length defines scene units; it is not a measured distance.
/// Retain each alternative in a separate scene. The output has no geographic
/// acceptance, covariance, or independent-evidence claim.
///
/// # Errors
/// Rejects invalid observations, a disconnected or invalid gauge, more than 128
/// free cameras or 16,384 points, invalid iteration limits, or numerical failure.
/// Errors leave the scene unchanged.
pub fn refine_local_scene_with_gauge(
    camera: &CameraModel,
    scene: &mut LocalScene,
    gauge: SceneCoordinateGauge,
    max_iterations: usize,
) -> Result<SceneRefinement, LocalSceneError> {
    refine(camera, scene, max_iterations, Some(gauge))
}

fn refine(
    camera: &CameraModel,
    scene: &mut LocalScene,
    max_iterations: usize,
    gauge: Option<SceneCoordinateGauge>,
) -> Result<SceneRefinement, LocalSceneError> {
    validation::validate(camera, scene, max_iterations, gauge)?;
    let mut poses = scene
        .cameras
        .iter()
        .map(|c| pose::Pose::from_scene(c.pose))
        .collect::<Vec<_>>();
    let fixed: BTreeSet<_> = scene
        .cameras
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.fixed.then_some(i))
        .collect();
    let mut points = scene
        .points
        .iter()
        .map(|p| bundle::Landmark {
            world: p.position,
            observations: p
                .observations
                .iter()
                .map(|o| bundle::Observation {
                    frame: o.camera_index,
                    pixel: o.pixel,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let result = bundle::refine(
        camera,
        &mut poses,
        &mut points,
        &fixed,
        max_iterations,
        gauge,
    )
    .ok_or(LocalSceneError::Numerical {
        cameras: poses.len(),
        points: points.len(),
    })?;
    for (camera, pose) in scene.cameras.iter_mut().zip(poses) {
        if !camera.fixed {
            camera.pose = pose.to_scene()
        }
    }
    for (point, refined) in scene.points.iter_mut().zip(points) {
        point.position = refined.world
    }
    Ok(SceneRefinement {
        initial_cost: result.initial_cost,
        final_cost: result.final_cost,
        steps: result.steps,
    })
}
#[cfg(test)]
mod tests;
