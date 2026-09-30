//! Limit repeated local image support before counting evidence or fitting covariance.
use crate::{CameraModel, CameraPose, pose_solver::Correspondence};
use nalgebra::Vector2;

pub(super) fn separated(
    camera: &CameraModel,
    reference: &CameraPose,
    points: &[Correspondence],
    minimum_distance: f64,
) -> (Vec<Correspondence>, usize) {
    let mut ordered: Vec<_> = points.iter().collect();
    ordered.sort_by(|a, b| {
        a.pixel
            .x
            .total_cmp(&b.pixel.x)
            .then(a.pixel.y.total_cmp(&b.pixel.y))
            .then(a.world.x.total_cmp(&b.world.x))
            .then(a.world.y.total_cmp(&b.world.y))
            .then(a.world.z.total_cmp(&b.world.z))
    });
    let mut selected: Vec<Correspondence> = Vec::new();
    let mut reference_pixels: Vec<Vector2<f64>> = Vec::new();
    let mut cells = [false; 12];
    let radius_squared = minimum_distance * minimum_distance;
    for point in ordered {
        let Some(pixel) = camera.project(reference, point.world) else {
            continue;
        };
        if selected
            .iter()
            .any(|p| (p.pixel - point.pixel).norm_squared() < radius_squared)
            || reference_pixels
                .iter()
                .any(|p| (p - pixel).norm_squared() < radius_squared)
        {
            continue;
        }
        let x = (pixel.x * 4.0 / f64::from(camera.width)).clamp(0.0, 3.0) as usize;
        let y = (pixel.y * 3.0 / f64::from(camera.height)).clamp(0.0, 2.0) as usize;
        cells[y * 4 + x] = true;
        reference_pixels.push(pixel);
        selected.push(point.clone());
    }
    (
        selected,
        cells.into_iter().filter(|occupied| *occupied).count(),
    )
}

#[cfg(test)]
mod tests;
