//! Spatial holdouts measure conditional map agreement, not geographic truth.
use navigate_visual::{CameraModel, CameraPose, PixelMatch, ReferenceView};
use serde_json::{Value, json};

pub(super) fn partition(pair: &PixelMatch, fold: u32) -> Option<bool> {
    let p = pair.reference;
    if !p.iter().all(|v| v.is_finite() && (0.0..8192.0).contains(v)) {
        return None;
    }
    let interior = |v: f64| (4.0..60.0).contains(&(v % 64.0));
    if !interior(p.x) || !interior(p.y) {
        return None;
    }
    Some(((p.x / 64.0) as u32 + 3 * (p.y / 64.0) as u32) % 5 == fold)
}

pub(super) fn calibrated(pair: PixelMatch, camera: &CameraModel, scale: f64) -> PixelMatch {
    PixelMatch {
        reference: pair.reference,
        query: nalgebra::Vector2::new(
            camera.cx + (pair.query.x - camera.cx) / scale,
            camera.cy + (pair.query.y - camera.cy) / scale,
        ),
    }
}

pub(super) fn residual(
    pair: PixelMatch,
    camera: &CameraModel,
    reference: (&ReferenceView, &CameraModel),
    pose: &CameraPose,
) -> Option<f64> {
    let (reference, reference_camera) = reference;
    let p = pair.reference;
    if !p.iter().chain(pair.query.iter()).all(|v| v.is_finite())
        || p.x < 0.0
        || p.y < 0.0
        || p.x >= f64::from(reference_camera.width - 1)
        || p.y >= f64::from(reference_camera.height - 1)
    {
        return None;
    }
    let index = p.y.round() as usize * reference_camera.width as usize + p.x.round() as usize;
    let depth = f64::from(*reference.depth_m.get(index)?);
    if !depth.is_finite() || depth <= 0.0 {
        return None;
    }
    let world = reference_camera.unproject(&reference.pose, p, depth);
    Some(
        camera
            .project(pose, world)
            .map_or(f64::INFINITY, |q| (q - pair.query).norm()),
    )
}

pub(super) fn score(
    pairs: &[PixelMatch],
    camera: &CameraModel,
    reference: (&ReferenceView, &CameraModel),
    pose: &CameraPose,
    scale: f64,
) -> Value {
    let mut errors: Vec<_> = pairs
        .iter()
        .filter_map(|p| residual(calibrated(*p, camera, scale), camera, reference, pose))
        .collect();
    errors.sort_by(f64::total_cmp);
    let below = errors.iter().filter(|e| **e <= 2.5).count();
    let quantile = |q: f64| {
        errors
            .get(((errors.len().saturating_sub(1) as f64) * q).ceil() as usize)
            .copied()
            .filter(|e| e.is_finite())
    };
    json!({"matches":pairs.len(),"valid_depth":errors.len(),
        "behind_camera":errors.iter().filter(|e| !e.is_finite()).count(),
        "within_2_5_px":below,"median_px":quantile(0.5),"p90_px":quantile(0.9),
        "scope":"All withheld matches with valid depth; no pose-based outlier removal. Matcher and map errors remain shared."})
}

#[cfg(test)]
mod tests;
