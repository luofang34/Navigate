//! Similarity fitting with positive scale and a proper rotation.
use super::*;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
pub(super) fn fit(points: &[SceneMapAssociation], ids: &[usize]) -> Option<SceneTransform> {
    if ids.len() < 3 {
        return None;
    }
    let n = ids.len() as f64;
    let source = ids
        .iter()
        .fold(Vector3::zeros(), |sum, &i| sum + points[i].scene_point / n);
    let target = ids
        .iter()
        .fold(Vector3::zeros(), |sum, &i| sum + points[i].map_point / n);
    let mut covariance = Matrix3::zeros();
    let mut variance = 0.0;
    for &i in ids {
        let x = points[i].scene_point - source;
        let y = points[i].map_point - target;
        covariance += y * x.transpose();
        variance += x.norm_squared();
    }
    if !covariance.iter().all(|v| v.is_finite()) || !variance.is_finite() || variance <= 1e-12 {
        return None;
    }
    let svd = covariance.svd(true, true);
    if svd.singular_values[1] <= svd.singular_values[0] * 1e-8 {
        return None;
    }
    let u = svd.u?;
    let v = svd.v_t?;
    let sign = (u * v).determinant().signum();
    let scale = (svd.singular_values[0] + svd.singular_values[1] + sign * svd.singular_values[2])
        / variance;
    let rotation = UnitQuaternion::from_matrix(
        &(u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, sign)) * v),
    );
    let translation = target - rotation * source * scale;
    if !scale.is_finite() || scale <= 0.0 || !translation.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(SceneTransform {
        scale,
        rotation,
        translation,
    })
}
