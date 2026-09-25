//! Align conditional local groups through their shared image identities.
use crate::{LocalScene, LocalScenePose};
use nalgebra::{UnitQuaternion, Vector3};
mod fit;
mod points;
pub use points::{
    PointAlignmentConfig, PointAssociationConfig, ScenePointAlignment,
    ScenePointAlignmentProposals, ScenePointAssociation, align_scenes_with_points,
    associate_scene_points,
};
mod input;
mod types;
pub use types::{AlignmentConfig, SceneAlignment, SceneAlignmentError, SceneTransform};
struct Link {
    identity: String,
    source: LocalScenePose,
    target: LocalScenePose,
}
fn support(count: usize, config: AlignmentConfig) -> Result<(), SceneAlignmentError> {
    if count < config.min_shared_cameras {
        Err(SceneAlignmentError::Support {
            retained: count,
            required: config.min_shared_cameras,
        })
    } else {
        Ok(())
    }
}
fn rotation_error(link: &Link, transform: SceneTransform) -> f64 {
    transform
        .pose(link.source)
        .orientation
        .angle_to(&link.target.orientation)
}
fn position_error(link: &Link, transform: SceneTransform) -> f64 {
    (transform.point(link.source.position) - link.target.position).norm()
}
fn select(
    links: &[Link],
    transform: SceneTransform,
    baseline: f64,
    config: AlignmentConfig,
) -> Vec<usize> {
    links
        .iter()
        .enumerate()
        .filter_map(|(i, link)| {
            (rotation_error(link, transform) <= config.max_rotation_error_rad
                && position_error(link, transform) <= baseline * config.max_position_error_ratio)
                .then_some(i)
        })
        .collect()
}
fn rotation_seed(
    links: &[Link],
    config: AlignmentConfig,
) -> Result<(UnitQuaternion<f64>, Vec<usize>), SceneAlignmentError> {
    let rotations: Vec<_> = links
        .iter()
        .map(|p| p.target.orientation * p.source.orientation.inverse())
        .collect();
    let score = |q: &UnitQuaternion<f64>| {
        rotations
            .iter()
            .map(|r| q.angle_to(r).min(config.max_rotation_error_rad))
            .sum::<f64>()
    };
    let rotation = rotations
        .iter()
        .min_by(|a, b| score(a).total_cmp(&score(b)))
        .copied()
        .ok_or(SceneAlignmentError::Support {
            retained: 0,
            required: config.min_shared_cameras,
        })?;
    let ids: Vec<_> = rotations
        .iter()
        .enumerate()
        .filter_map(|(i, q)| (rotation.angle_to(q) <= config.max_rotation_error_rad).then_some(i))
        .collect();
    support(ids.len(), config)?;
    Ok((
        fit::rotation(links, &ids, rotation).ok_or(SceneAlignmentError::Baseline)?,
        ids,
    ))
}
fn span(scene: &LocalScene) -> f64 {
    scene
        .cameras
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            scene.cameras[i + 1..]
                .iter()
                .map(move |b| (a.pose.position - b.pose.position).norm())
        })
        .fold(0.0, f64::max)
}
/// Fit a conditional transform from a source group into a target group.
///
/// Camera identities must describe the same processed observations. This
/// operation aligns one pair of estimated alternatives. It does not average
/// geographic alternatives or provide a geographic acceptance decision.
///
/// # Errors
/// Rejects invalid inputs, weak shared motion, and inconsistent shared poses.
pub fn align_scenes(
    source: &LocalScene,
    target: &LocalScene,
    config: AlignmentConfig,
) -> Result<SceneAlignment, SceneAlignmentError> {
    let links = input::join(source, target, config)?;
    support(links.len(), config)?;
    let (rotation, ids) = rotation_seed(&links, config)?;
    let target_baseline = fit::baseline(&links, &ids, true).ok_or(SceneAlignmentError::Baseline)?;
    let source_baseline =
        fit::baseline(&links, &ids, false).ok_or(SceneAlignmentError::Baseline)?;
    if target_baseline <= 1e-8
        || source_baseline <= 1e-8
        || target_baseline < span(target) * config.min_shared_baseline_ratio
        || source_baseline < span(source) * config.min_shared_baseline_ratio
    {
        return Err(SceneAlignmentError::Baseline);
    }
    let mut transform =
        fit::initial(&links, &ids, rotation).ok_or(SceneAlignmentError::Baseline)?;
    let mut retained = select(&links, transform, target_baseline, config);
    for _ in 0..3 {
        support(retained.len(), config)?;
        transform = fit::refine(&links, &retained, transform.rotation)
            .ok_or(SceneAlignmentError::Baseline)?;
        let next = select(&links, transform, target_baseline, config);
        if next == retained {
            break;
        }
        retained = next;
    }
    support(retained.len(), config)?;
    let baseline = fit::baseline(&links, &retained, true).ok_or(SceneAlignmentError::Baseline)?;
    let count = retained.len() as f64;
    Ok(SceneAlignment {
        transform,
        observation_sha256: retained
            .iter()
            .map(|&i| links[i].identity.clone())
            .collect(),
        excluded_observation_sha256: links
            .iter()
            .enumerate()
            .filter(|(i, _)| !retained.contains(i))
            .map(|(_, p)| p.identity.clone())
            .collect(),
        position_rms_scene_units: (retained
            .iter()
            .map(|&i| position_error(&links[i], transform).powi(2))
            .sum::<f64>()
            / count)
            .sqrt(),
        rotation_rms_rad: (retained
            .iter()
            .map(|&i| rotation_error(&links[i], transform).powi(2))
            .sum::<f64>()
            / count)
            .sqrt(),
        target_baseline_scene_units: baseline,
    })
}
#[cfg(test)]
mod tests;
