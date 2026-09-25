//! Validate point evidence and join repeated image observations.
use super::super::{SceneAlignmentError, input};
use super::types::{PointAssociationConfig, ScenePointAssociation};
use crate::{LocalScene, LocalScenePoint};
use nalgebra::Vector2;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate(scene: &LocalScene, group: &'static str) -> Result<(), SceneAlignmentError> {
    input::validate(scene, group)?;
    if scene.points.len() > 16_384 {
        return Err(SceneAlignmentError::Point {
            group,
            feature_id: 0,
            reason: "more than 16384 scene points",
        });
    }
    let mut ids = BTreeSet::new();
    for p in &scene.points {
        let mut cameras = BTreeSet::new();
        if !ids.insert(p.feature_id) || !p.position.iter().all(|v| v.is_finite()) {
            return Err(SceneAlignmentError::Point {
                group,
                feature_id: p.feature_id,
                reason: "repeated ID or non-finite position",
            });
        }
        for o in &p.observations {
            if o.camera_index >= scene.cameras.len()
                || !cameras.insert(o.camera_index)
                || !o.pixel.iter().all(|v| v.is_finite() && v.abs() <= 1e9)
            {
                return Err(SceneAlignmentError::Point {
                    group,
                    feature_id: p.feature_id,
                    reason: "invalid or repeated image observation",
                });
            }
        }
    }
    Ok(())
}
pub(super) fn config(c: PointAssociationConfig) -> Result<(), SceneAlignmentError> {
    if !c.max_pixel_distance.is_finite()
        || !(1e-6..=16.0).contains(&c.max_pixel_distance)
        || !(3..=129).contains(&c.min_shared_observations)
    {
        return Err(SceneAlignmentError::Config);
    }
    Ok(())
}
fn pixels<'a>(scene: &'a LocalScene, point: &'a LocalScenePoint) -> BTreeMap<String, Vector2<f64>> {
    point
        .observations
        .iter()
        .map(|o| {
            (
                scene.cameras[o.camera_index]
                    .observation_sha256
                    .to_ascii_lowercase(),
                o.pixel,
            )
        })
        .collect()
}
pub(super) fn shared(
    source: &LocalScene,
    target: &LocalScene,
    a: &LocalScenePoint,
    b: &LocalScenePoint,
    allowed: &BTreeSet<String>,
    c: PointAssociationConfig,
) -> bool {
    let target = pixels(target, b);
    pixels(source, a)
        .iter()
        .filter(|(id, p)| {
            allowed.contains(*id)
                && target
                    .get(*id)
                    .is_some_and(|q| (*p - q).norm() <= c.max_pixel_distance)
        })
        .count()
        >= c.min_shared_observations
}
type Samples = Vec<(u64, Vector2<f64>)>;
fn by_image(scene: &LocalScene) -> BTreeMap<String, Samples> {
    let mut images = BTreeMap::<String, Samples>::new();
    for p in &scene.points {
        for o in &p.observations {
            images
                .entry(
                    scene.cameras[o.camera_index]
                        .observation_sha256
                        .to_ascii_lowercase(),
                )
                .or_default()
                .push((p.feature_id, o.pixel));
        }
    }
    images
}
fn nearest(source: &Samples, target: &Samples, radius: f64) -> BTreeMap<u64, u64> {
    let cell = |p: Vector2<f64>| ((p.x / radius).floor() as i64, (p.y / radius).floor() as i64);
    let mut grid = BTreeMap::<(i64, i64), Samples>::new();
    for &(id, p) in target {
        grid.entry(cell(p)).or_default().push((id, p));
    }
    let mut output = BTreeMap::new();
    for &(id, p) in source {
        let (x, y) = cell(p);
        let (mut best, mut distance, mut tied) = (None, radius * radius, false);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &(other, q) in grid.get(&(x + dx, y + dy)).into_iter().flatten() {
                    let d = (p - q).norm_squared();
                    if d <= distance {
                        tied = best.is_some() && (d - distance).abs() < 1e-16;
                        if !tied {
                            best = Some(other);
                            distance = d;
                        }
                    }
                }
            }
        }
        if !tied && let Some(other) = best {
            output.insert(id, other);
        }
    }
    output
}
/// Propose point pairs from mutual nearest pixels in the same source images.
/// This only finds associations. Call the separate geometric fit for each scene
/// alternative. Equal point IDs across scenes do not establish a correspondence.
///
/// # Errors
/// Rejects invalid scene identities, point records, and association limits.
pub fn associate_scene_points(
    source: &LocalScene,
    target: &LocalScene,
    limits: PointAssociationConfig,
) -> Result<Vec<ScenePointAssociation>, SceneAlignmentError> {
    config(limits)?;
    validate(source, "source")?;
    validate(target, "target")?;
    let source = by_image(source);
    let target = by_image(target);
    let mut votes = BTreeMap::<ScenePointAssociation, usize>::new();
    for (id, a) in &source {
        let Some(b) = target.get(id) else { continue };
        let forward = nearest(a, b, limits.max_pixel_distance);
        let reverse = nearest(b, a, limits.max_pixel_distance);
        for (source_feature_id, target_feature_id) in forward {
            if reverse.get(&target_feature_id) == Some(&source_feature_id) {
                let count = votes
                    .entry(ScenePointAssociation {
                        source_feature_id,
                        target_feature_id,
                    })
                    .or_default();
                *count = count.wrapping_add(1);
            }
        }
    }
    let pairs: Vec<_> = votes
        .into_iter()
        .filter_map(|(pair, count)| (count >= limits.min_shared_observations).then_some(pair))
        .collect();
    let mut sources = BTreeMap::<u64, usize>::new();
    let mut targets = BTreeMap::<u64, usize>::new();
    for p in &pairs {
        let s = sources.entry(p.source_feature_id).or_default();
        *s = s.wrapping_add(1);
        let t = targets.entry(p.target_feature_id).or_default();
        *t = t.wrapping_add(1);
    }
    Ok(pairs
        .into_iter()
        .filter(|p| sources[&p.source_feature_id] == 1 && targets[&p.target_feature_id] == 1)
        .collect())
}
