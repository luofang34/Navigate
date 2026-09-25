//! Associate visible scene points with unique reference-depth correspondences.
use super::*;
use crate::local_scene::pose::Pose;
use nalgebra::Vector2;
use std::collections::{BTreeMap, BTreeSet};
struct Point {
    feature_id: u64,
    world: nalgebra::Vector3<f64>,
    pixel: Vector2<f64>,
}
fn invalid(index: usize, reason: &'static str) -> SceneRegistrationError {
    SceneRegistrationError::Input { index, reason }
}
fn inside(p: Vector2<f64>, camera: &CameraModel) -> bool {
    p.iter().all(|v| v.is_finite())
        && p.x >= 0.0
        && p.y >= 0.0
        && p.x < f64::from(camera.width)
        && p.y < f64::from(camera.height)
}
fn points(
    camera: &CameraModel,
    scene: &LocalScene,
    digest: &str,
) -> Result<Vec<Point>, SceneRegistrationError> {
    if scene.cameras.is_empty() || scene.cameras.len() > 129 || scene.points.len() > 65536 {
        return Err(invalid(0, "scene exceeds registration resource bounds"));
    }
    let mut identities = BTreeSet::new();
    for (i, c) in scene.cameras.iter().enumerate() {
        if c.observation_sha256.len() != 64
            || !c.observation_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || !identities.insert(c.observation_sha256.to_ascii_lowercase())
            || !c
                .pose
                .position
                .iter()
                .chain(c.pose.orientation.coords.iter())
                .all(|v| v.is_finite())
            || (c.pose.orientation.norm_squared() - 1.0).abs() > 1e-8
        {
            return Err(invalid(
                i,
                "invalid or repeated scene camera identity or pose",
            ));
        }
    }
    let index = scene
        .cameras
        .iter()
        .position(|c| c.observation_sha256.eq_ignore_ascii_case(digest))
        .ok_or_else(|| SceneRegistrationError::Observation {
            observation_sha256: digest.to_owned(),
        })?;
    let pose = Pose::from_scene(scene.cameras[index].pose);
    let mut features = BTreeSet::new();
    let mut selected = Vec::new();
    for (i, point) in scene.points.iter().enumerate() {
        if !features.insert(point.feature_id)
            || !point.position.iter().all(|v| v.is_finite())
            || point.observations.len() > 129
        {
            return Err(invalid(i, "invalid or repeated scene point"));
        }
        let mut cameras = BTreeSet::new();
        for o in &point.observations {
            if o.camera_index >= scene.cameras.len()
                || !cameras.insert(o.camera_index)
                || !inside(o.pixel, camera)
            {
                return Err(invalid(i, "invalid or repeated point observation"));
            }
        }
        if point.observations.len() < 2 {
            continue;
        }
        let Some(o) = point.observations.iter().find(|o| o.camera_index == index) else {
            continue;
        };
        if pose
            .project(camera, point.position)
            .is_some_and(|p| (p - o.pixel).norm() <= 3.0)
        {
            selected.push(Point {
                feature_id: point.feature_id,
                world: point.position,
                pixel: o.pixel,
            });
        }
    }
    Ok(selected)
}
fn cell(pixel: Vector2<f64>, size: f64) -> (i64, i64) {
    (
        (pixel.x / size).floor() as i64,
        (pixel.y / size).floor() as i64,
    )
}
fn nearest(
    points: &[Point],
    grid: &BTreeMap<(i64, i64), Vec<usize>>,
    query: Vector2<f64>,
    radius: f64,
) -> Option<(usize, f64)> {
    let key = cell(query, radius);
    let mut found = Vec::new();
    for dx in -1..=1 {
        for dy in -1..=1 {
            if let Some(ids) = grid.get(&(key.0.saturating_add(dx), key.1.saturating_add(dy))) {
                for &i in ids {
                    let distance = (points[i].pixel - query).norm();
                    if distance <= radius {
                        found.push((i, distance));
                    }
                }
            }
        }
    }
    found.sort_by(|a, b| a.1.total_cmp(&b.1));
    let first = *found.first()?;
    if found.get(1).is_some_and(|next| next.1 - first.1 < 1e-6) {
        return None;
    }
    Some(first)
}
pub(super) fn associate(
    camera: &CameraModel,
    scene: &LocalScene,
    digest: &str,
    reference: &ReferenceView,
    matches: &[PixelMatch],
    config: RegistrationConfig,
) -> Result<Vec<SceneMapAssociation>, SceneRegistrationError> {
    if matches.len() > 4096 {
        return Err(invalid(matches.len(), "too many pixel matches"));
    }
    let points = points(camera, scene, digest)?;
    let mut grid = BTreeMap::<_, Vec<_>>::new();
    for (i, p) in points.iter().enumerate() {
        grid.entry(cell(p.pixel, config.association_radius_px))
            .or_default()
            .push(i);
    }
    let mut linked = Vec::new();
    for (match_index, pair) in matches.iter().enumerate() {
        if !inside(pair.query, camera) || !inside(pair.reference, camera) {
            continue;
        }
        let [x, y] = [
            pair.reference.x.round() as usize,
            pair.reference.y.round() as usize,
        ];
        if x >= camera.width as usize || y >= camera.height as usize {
            continue;
        }
        let depth = f64::from(reference.depth_m[y * camera.width as usize + x]);
        if !depth.is_finite() || depth <= 0.0 {
            continue;
        }
        let Some((i, distance)) = nearest(&points, &grid, pair.query, config.association_radius_px)
        else {
            continue;
        };
        let world = camera.unproject(&reference.pose, pair.reference, depth);
        if !world.iter().all(|v| v.is_finite()) {
            continue;
        }
        linked.push(SceneMapAssociation {
            feature_id: points[i].feature_id,
            match_index,
            scene_point: points[i].world,
            map_point: world,
            query_pixel: pair.query,
            reference_pixel: pair.reference,
            association_distance_px: distance,
        });
    }
    linked.sort_by(|a, b| {
        a.association_distance_px
            .total_cmp(&b.association_distance_px)
            .then_with(|| a.match_index.cmp(&b.match_index))
    });
    let mut features = BTreeSet::new();
    let mut pixels = BTreeSet::new();
    linked.retain(|p| {
        let pixel = (
            p.reference_pixel.x.round() as u32,
            p.reference_pixel.y.round() as u32,
        );
        if features.contains(&p.feature_id) || pixels.contains(&pixel) {
            return false;
        }
        features.insert(p.feature_id);
        pixels.insert(pixel);
        true
    });
    Ok(linked)
}
