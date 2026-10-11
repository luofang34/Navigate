//! Symmetric point-to-epipolar-line distances in the observed pixel grid.
use nalgebra::{Matrix3, Vector2, Vector3};
use navigate_visual::{CameraModel, CameraPose};
use serde_json::{Value, json};

pub(super) fn score(
    camera: CameraModel,
    a: CameraPose,
    b: CameraPose,
    pairs: &[(Vector2<f64>, Vector2<f64>)],
) -> Value {
    let r = (b.orientation.inverse() * a.orientation).to_rotation_matrix();
    let t = b.orientation.inverse() * (a.position - b.position);
    let baseline = t.norm();
    let rays = Matrix3::new(
        1.0 / camera.fx,
        0.0,
        -camera.cx / camera.fx,
        0.0,
        -1.0 / camera.fy,
        camera.cy / camera.fy,
        0.0,
        0.0,
        -1.0,
    );
    let f = (baseline > 1e-9).then(|| {
        let v = t / baseline;
        rays.transpose()
            * Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
            * r.matrix()
            * rays
    });
    let mut epipolar = Vec::new();
    let mut rotation = Vec::new();
    for (p, q) in pairs {
        let x = Vector3::new(p.x, p.y, 1.0);
        let y = Vector3::new(q.x, q.y, 1.0);
        if let Some(f) = f {
            let l = f * x;
            let m = f.transpose() * y;
            let aa = l.x * l.x + l.y * l.y;
            let bb = m.x * m.x + m.y * m.y;
            if aa > 1e-24 && bb > 1e-24 {
                epipolar.push((0.5 * y.dot(&l).powi(2) * (1.0 / aa + 1.0 / bb)).sqrt());
            }
        }
        let eye = r.matrix() * rays * x;
        if eye.z < -1e-9 {
            let predicted = Vector2::new(
                camera.cx - camera.fx * eye.x / eye.z,
                camera.cy + camera.fy * eye.y / eye.z,
            );
            rotation.push((predicted - q).norm());
        }
    }
    json!({"available":true,"tracks":pairs.len(),"estimated_baseline_m":baseline,"epipolar":distribution(epipolar),"rotation_only":distribution(rotation)})
}
fn distribution(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let percentile = |q: f64| {
        values
            .get(((values.len().saturating_sub(1) as f64) * q).ceil() as usize)
            .copied()
    };
    json!({"count":values.len(),"median_px":percentile(0.5),"p90_px":percentile(0.9),"within_2_5_px":values.iter().filter(|v|**v<=2.5).count()})
}
#[cfg(test)]
mod tests;
