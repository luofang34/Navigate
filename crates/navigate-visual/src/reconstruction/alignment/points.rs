//! Conditional scale from shared points when camera motion has little baseline.
use super::{
    Link, SceneAlignment, SceneAlignmentError, SceneTransform, fit, input, rotation_seed, support,
};
use crate::LocalScene;
use nalgebra::{UnitQuaternion, Vector3};
use std::collections::{BTreeMap, BTreeSet};
mod observations;
mod scale;
mod types;
pub use observations::associate_scene_points;
pub use types::{
    PointAlignmentConfig, PointAssociationConfig, ScenePointAlignment,
    ScenePointAlignmentProposals, ScenePointAssociation,
};
struct PointLink {
    association: ScenePointAssociation,
    source: Vector3<f64>,
    target: Vector3<f64>,
}
struct Frame {
    source: Vector3<f64>,
    target: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
}
fn validate(config: PointAlignmentConfig) -> Result<(), SceneAlignmentError> {
    observations::config(config.associations)?;
    if !(8..=16_384).contains(&config.min_shared_points)
        || !(1..=1024).contains(&config.max_scale_proposals)
        || !(1..=32).contains(&config.max_candidates)
        || [
            config.max_point_error_ratio,
            config.max_camera_position_error_ratio,
            config.min_point_spread_ratio,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(0.0..1.0).contains(v) || *v == 0.0)
    {
        return Err(SceneAlignmentError::Config);
    }
    Ok(())
}
fn point_links(
    scenes: [&LocalScene; 2],
    associations: &[ScenePointAssociation],
    ids: &BTreeSet<String>,
    frame: &Frame,
    config: PointAssociationConfig,
) -> Result<Vec<PointLink>, SceneAlignmentError> {
    let [source, target] = scenes;
    let source_points: BTreeMap<_, _> = source.points.iter().map(|p| (p.feature_id, p)).collect();
    let target_points: BTreeMap<_, _> = target.points.iter().map(|p| (p.feature_id, p)).collect();
    let (mut sources, mut targets) = (BTreeSet::new(), BTreeSet::new());
    let mut pixels = BTreeSet::new();
    let mut output = Vec::new();
    for (index, &association) in associations.iter().enumerate() {
        if !sources.insert(association.source_feature_id)
            || !targets.insert(association.target_feature_id)
        {
            return Err(SceneAlignmentError::Association {
                index,
                reason: "repeated source or target point",
            });
        }
        let a = source_points.get(&association.source_feature_id).ok_or(
            SceneAlignmentError::Association {
                index,
                reason: "source point is absent",
            },
        )?;
        let b = target_points.get(&association.target_feature_id).ok_or(
            SceneAlignmentError::Association {
                index,
                reason: "target point is absent",
            },
        )?;
        if !observations::shared(source, target, a, b, ids, config) {
            continue;
        }
        for (side, scene, point) in [(0, source, *a), (1, target, *b)] {
            for o in &point.observations {
                let id = scene.cameras[o.camera_index]
                    .observation_sha256
                    .to_ascii_lowercase();
                if ids.contains(&id)
                    && !pixels.insert((
                        side,
                        id,
                        canonical_bits(o.pixel.x),
                        canonical_bits(o.pixel.y),
                    ))
                {
                    return Err(SceneAlignmentError::Association {
                        index,
                        reason: "repeated image evidence under different point IDs",
                    });
                }
            }
        }
        let a = frame.rotation * (a.position - frame.source);
        let b = b.position - frame.target;
        if a.norm() > 1e-8 && b.norm() > 1e-8 {
            output.push(PointLink {
                association,
                source: a,
                target: b,
            });
        }
    }
    Ok(output)
}
fn canonical_bits(v: f64) -> u64 {
    if v == 0.0 { 0 } else { v.to_bits() }
}
fn point_support(retained: usize, config: PointAlignmentConfig) -> Result<(), SceneAlignmentError> {
    if retained < config.min_shared_points {
        Err(SceneAlignmentError::PointSupport {
            retained,
            required: config.min_shared_points,
        })
    } else {
        Ok(())
    }
}
/// Propose conditional transforms using camera rotations, shared origins, and
/// point depth. Scene point associations can come from any host algorithm.
///
/// The fit checks their shared image support before using them. It preserves
/// distinct supported scale proposals and reports bounded search. The returned
/// residuals describe agreement between correlated estimates, not accuracy.
/// Each call compares one source scene alternative with one target alternative.
///
/// # Errors
/// Rejects invalid or repeated evidence, insufficient point or camera support,
/// weak point extent, and inconsistent camera rotations or positions.
pub fn align_scenes_with_points(
    source: &LocalScene,
    target: &LocalScene,
    associations: &[ScenePointAssociation],
    config: PointAlignmentConfig,
) -> Result<ScenePointAlignmentProposals, SceneAlignmentError> {
    validate(config)?;
    observations::validate(source, "source")?;
    observations::validate(target, "target")?;
    let cameras = input::join(source, target, config.cameras)?;
    support(cameras.len(), config.cameras)?;
    let (rotation, retained) = rotation_seed(&cameras, config.cameras)?;
    let count = retained.len() as f64;
    let frame = Frame {
        rotation,
        source: retained
            .iter()
            .map(|&i| cameras[i].source.position)
            .sum::<Vector3<f64>>()
            / count,
        target: retained
            .iter()
            .map(|&i| cameras[i].target.position)
            .sum::<Vector3<f64>>()
            / count,
    };
    let ids = retained
        .iter()
        .map(|&i| cameras[i].identity.to_ascii_lowercase())
        .collect();
    let points = point_links(
        [source, target],
        associations,
        &ids,
        &frame,
        config.associations,
    )?;
    point_support(points.len(), config)?;
    let proposals = scale::propose(&points, config);
    let mut candidates = Vec::<ScenePointAlignment>::new();
    for (scale, indices) in proposals.candidates {
        let Some(result) = result(
            &cameras,
            &retained,
            &points,
            &indices,
            &frame,
            scale,
            (associations, config),
        ) else {
            continue;
        };
        if !candidates
            .iter()
            .any(|c| (c.alignment.transform.scale - scale).abs() <= scale * 1e-6)
        {
            candidates.push(result)
        }
    }
    candidates.sort_by(|a, b| {
        b.associations
            .len()
            .cmp(&a.associations.len())
            .then(a.point_rms_scene_units.total_cmp(&b.point_rms_scene_units))
    });
    if candidates.is_empty() {
        return Err(SceneAlignmentError::Baseline);
    }
    let exhausted = proposals.exhausted || candidates.len() > config.max_candidates;
    candidates.truncate(config.max_candidates);
    Ok(ScenePointAlignmentProposals {
        candidates,
        candidate_budget_exhausted: exhausted,
    })
}
fn result(
    cameras: &[Link],
    ids: &[usize],
    points: &[PointLink],
    indices: &[usize],
    frame: &Frame,
    scale: f64,
    settings: (&[ScenePointAssociation], PointAlignmentConfig),
) -> Option<ScenePointAlignment> {
    let (proposals, config) = settings;
    let distance = fit::median(indices.iter().map(|&i| points[i].target.norm()).collect())?;
    let transform = SceneTransform {
        scale,
        rotation: frame.rotation,
        translation: frame.target - frame.rotation * frame.source * scale,
    };
    if ids.iter().any(|&i| {
        super::position_error(&cameras[i], transform)
            > distance * config.max_camera_position_error_ratio
    }) {
        return None;
    }
    let count = ids.len() as f64;
    let associations: Vec<_> = indices.iter().map(|&i| points[i].association).collect();
    Some(ScenePointAlignment {
        excluded_associations: proposals
            .iter()
            .copied()
            .filter(|p| !associations.contains(p))
            .collect(),
        associations,
        point_rms_scene_units: (indices
            .iter()
            .map(|&i| (points[i].source * scale - points[i].target).norm_squared())
            .sum::<f64>()
            / indices.len() as f64)
            .sqrt(),
        target_point_distance_scene_units: distance,
        alignment: SceneAlignment {
            transform,
            observation_sha256: ids.iter().map(|&i| cameras[i].identity.clone()).collect(),
            excluded_observation_sha256: cameras
                .iter()
                .enumerate()
                .filter(|(i, _)| !ids.contains(i))
                .map(|(_, c)| c.identity.clone())
                .collect(),
            position_rms_scene_units: (ids
                .iter()
                .map(|&i| super::position_error(&cameras[i], transform).powi(2))
                .sum::<f64>()
                / count)
                .sqrt(),
            rotation_rms_rad: (ids
                .iter()
                .map(|&i| super::rotation_error(&cameras[i], transform).powi(2))
                .sum::<f64>()
                / count)
                .sqrt(),
            target_baseline_scene_units: fit::baseline(cameras, ids, true)?,
        },
    })
}
#[cfg(test)]
mod tests;
