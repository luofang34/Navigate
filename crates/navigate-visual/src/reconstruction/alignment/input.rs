//! Validate and join groups by exact processing identities.
use super::*;
use std::collections::BTreeMap;
pub(super) fn join(
    source: &LocalScene,
    target: &LocalScene,
    config: AlignmentConfig,
) -> Result<Vec<Link>, SceneAlignmentError> {
    if config.min_shared_cameras < 3
        || config.min_shared_cameras > 129
        || !config.max_rotation_error_rad.is_finite()
        || !(0.0..std::f64::consts::PI).contains(&config.max_rotation_error_rad)
        || config.max_rotation_error_rad == 0.0
        || !config.max_position_error_ratio.is_finite()
        || !(0.0..1.0).contains(&config.max_position_error_ratio)
        || config.max_position_error_ratio == 0.0
        || !config.min_shared_baseline_ratio.is_finite()
        || !(0.0..1.0).contains(&config.min_shared_baseline_ratio)
        || config.min_shared_baseline_ratio == 0.0
    {
        return Err(SceneAlignmentError::Config);
    }
    validate(source, "source")?;
    validate(target, "target")?;
    let lookup: BTreeMap<_, _> = target
        .cameras
        .iter()
        .map(|c| (c.observation_sha256.to_ascii_lowercase(), c.pose))
        .collect();
    Ok(source
        .cameras
        .iter()
        .filter_map(|c| {
            Some(Link {
                identity: c.observation_sha256.clone(),
                source: c.pose,
                target: *lookup.get(&c.observation_sha256.to_ascii_lowercase())?,
            })
        })
        .collect())
}
pub(super) fn validate(scene: &LocalScene, group: &'static str) -> Result<(), SceneAlignmentError> {
    if scene.cameras.len() > 129 {
        return Err(SceneAlignmentError::Camera {
            group,
            index: 129,
            reason: "group has more than 129 cameras",
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    for (index, c) in scene.cameras.iter().enumerate() {
        let id = &c.observation_sha256;
        if id.len() != 64
            || !id.bytes().all(|v| v.is_ascii_hexdigit())
            || !seen.insert(id.to_ascii_lowercase())
        {
            return Err(SceneAlignmentError::Camera {
                group,
                index,
                reason: "invalid or repeated source digest",
            });
        }
        let p = c.pose;
        if !p
            .position
            .iter()
            .chain(p.orientation.coords.iter())
            .all(|v| v.is_finite())
            || (p.orientation.norm_squared() - 1.0).abs() > 1e-8
        {
            return Err(SceneAlignmentError::Camera {
                group,
                index,
                reason: "non-finite pose or non-unit rotation",
            });
        }
    }
    Ok(())
}
