//! Validate observation identity and the fixed support of each scene component.
use super::{LocalScene, LocalSceneError};
use crate::CameraModel;
use std::collections::{BTreeMap, BTreeSet};
pub(super) fn validate(
    camera: &CameraModel,
    scene: &LocalScene,
    iterations: usize,
) -> Result<(), LocalSceneError> {
    camera
        .validate()
        .map_err(|source| LocalSceneError::Camera { source })?;
    let free = scene.cameras.iter().filter(|c| !c.fixed).count();
    if free > 96
        || scene.points.is_empty()
        || scene.points.len() > 16384
        || !(1..=100).contains(&iterations)
    {
        return Err(LocalSceneError::Limits {
            cameras: free,
            points: scene.points.len(),
            iterations,
        });
    }
    cameras(scene)?;
    points(camera, scene)?;
    gauge(scene)
}
fn cameras(scene: &LocalScene) -> Result<(), LocalSceneError> {
    let mut identities = BTreeSet::new();
    for (index, camera) in scene.cameras.iter().enumerate() {
        let identity = &camera.observation_sha256;
        if identity.len() != 64
            || !identity.bytes().all(|c| c.is_ascii_hexdigit())
            || !identities.insert(identity)
        {
            return Err(LocalSceneError::Observation {
                index,
                reason: "invalid or repeated source digest",
            });
        }
        if !camera
            .pose
            .position
            .iter()
            .chain(camera.pose.orientation.coords.iter())
            .all(|x| x.is_finite())
            || (camera.pose.orientation.norm_squared() - 1.0).abs() > 1e-8
        {
            return Err(LocalSceneError::Observation {
                index,
                reason: "non-finite pose or non-unit rotation",
            });
        }
    }
    Ok(())
}
fn points(camera: &CameraModel, scene: &LocalScene) -> Result<(), LocalSceneError> {
    let mut identities = BTreeSet::new();
    for point in &scene.points {
        let fail = |reason| LocalSceneError::Point {
            feature_id: point.feature_id,
            reason,
        };
        if !identities.insert(point.feature_id) || !point.position.iter().all(|v| v.is_finite()) {
            return Err(fail("duplicate point identity or non-finite position"));
        }
        if point.observations.len() < 2 {
            return Err(fail("fewer than two image observations"));
        }
        let mut frames = BTreeSet::new();
        for observation in &point.observations {
            if observation.camera_index >= scene.cameras.len()
                || !frames.insert(observation.camera_index)
            {
                return Err(fail("missing or repeated camera observation"));
            }
            let p = observation.pixel;
            if !(0.0..f64::from(camera.width)).contains(&p.x)
                || !(0.0..f64::from(camera.height)).contains(&p.y)
            {
                return Err(fail("image coordinates are outside the calibrated image"));
            }
        }
    }
    Ok(())
}
fn root(parents: &[usize], mut index: usize) -> usize {
    while parents[index] != index {
        index = parents[index]
    }
    index
}
fn gauge(scene: &LocalScene) -> Result<(), LocalSceneError> {
    let mut parents: Vec<_> = (0..scene.cameras.len()).collect();
    let mut used = BTreeSet::new();
    for point in &scene.points {
        let first = point.observations[0].camera_index;
        used.insert(first);
        for observation in &point.observations[1..] {
            used.insert(observation.camera_index);
            let a = root(&parents, first);
            let b = root(&parents, observation.camera_index);
            parents[b] = a;
        }
    }
    if let Some((index, _)) = scene
        .cameras
        .iter()
        .enumerate()
        .find(|(i, c)| !c.fixed && !used.contains(i))
    {
        return Err(LocalSceneError::Unconstrained {
            cameras: vec![index],
        });
    }
    let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &frame in &used {
        components
            .entry(root(&parents, frame))
            .or_default()
            .push(frame);
    }
    for component in components.values() {
        let anchors: Vec<_> = component
            .iter()
            .filter(|&&i| scene.cameras[i].fixed)
            .collect();
        let baseline = anchors.iter().any(|&&a| {
            anchors.iter().any(|&&b| {
                (scene.cameras[a].pose.position - scene.cameras[b].pose.position).norm() > 1e-6
            })
        });
        if !baseline {
            return Err(LocalSceneError::Unconstrained {
                cameras: component.clone(),
            });
        }
    }
    Ok(())
}
