//! Five-point proposals behind Navigate-owned observations and poses.
use super::triangulation::Rays;
use super::*;
use kornia_3d::pose::{RansacParams, ransac_essential_5pt};
use kornia_algebra::{Mat3F64, Vec2F64};
pub(super) fn estimate(
    camera: &CameraModel,
    rays: &[Rays],
    camera_indices: [usize; 2],
) -> Result<(nalgebra::Matrix3<f64>, Vec<usize>), ReconstructionError> {
    let first: Vec<_> = rays
        .iter()
        .map(|p| Vec2F64::new(p.a.x * camera.fx + camera.cx, p.a.y * camera.fy + camera.cy))
        .collect();
    let second: Vec<_> = rays
        .iter()
        .map(|p| Vec2F64::new(p.b.x * camera.fx + camera.cx, p.b.y * camera.fy + camera.cy))
        .collect();
    let intrinsics = Mat3F64::from_cols_array(&[
        camera.fx, 0.0, 0.0, 0.0, camera.fy, 0.0, camera.cx, camera.cy, 1.0,
    ]);
    let config = RansacParams {
        max_iterations: 2000,
        threshold: 1.0,
        min_inliers: 20,
        random_seed: Some(0),
        refit: false,
    };
    let result = ransac_essential_5pt(&first, &second, &intrinsics, &intrinsics, &config).map_err(
        |source| ReconstructionError::TwoView {
            camera_indices,
            source,
        },
    )?;
    Ok((
        nalgebra::Matrix3::from_column_slice(&result.model.to_cols_array()),
        result
            .inliers
            .iter()
            .enumerate()
            .filter_map(|(i, &yes)| yes.then_some(i))
            .collect(),
    ))
}
