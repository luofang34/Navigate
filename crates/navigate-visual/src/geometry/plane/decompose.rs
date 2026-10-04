//! Homography decomposition into rotation, scaled translation, and plane normal.
//!
//! The method follows Ma, Soatto, Košecká and Sastry, "An Invitation to 3-D
//! Vision", section 5.3. With `X2 = R X1 + T` and the plane `N·X1 = d`, a
//! homography normalized by its middle singular value is `R + (T/d) Nᵀ`.

use super::{PlaneMotionConfig, PlaneSolution};
use crate::VisualError;
use nalgebra::{Matrix3, Rotation3, Unit, UnitQuaternion, Vector2, Vector3};

/// One algebraic solution `(R, N, T/d)` in optical-axis-forward axes.
pub(super) type Algebraic = (Matrix3<f64>, Vector3<f64>, Vector3<f64>);

/// Optical-axis-forward coordinates to eye axes.
const FLIP: Matrix3<f64> = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);

/// The physical solutions that keep the points in front of both cameras,
/// with their normals in optical-axis-forward axes.
pub(super) fn visible_solutions(
    h: &Matrix3<f64>,
    points: &[(Vector2<f64>, Vector2<f64>)],
    config: &PlaneMotionConfig,
) -> Result<Vec<(PlaneSolution, Vector3<f64>)>, VisualError> {
    let candidates = algebraic(h, points).ok_or(VisualError::DegenerateGeometry)?;
    let mut visible: Vec<Algebraic> = Vec::new();
    for (r, n, t) in candidates {
        if t.norm() < config.min_parallax || !in_front(&r, &n, &t, points, config) {
            continue;
        }
        // Motion along the normal makes two candidates equal.
        let repeated = visible
            .iter()
            .any(|(vr, vn, _)| (vn - n).norm() < 1e-6 && (vr - r).norm() < 1e-6);
        if !repeated {
            visible.push((r, n, t));
        }
    }
    if visible.is_empty() || visible.len() > 2 {
        return Err(VisualError::DegenerateGeometry);
    }
    Ok(visible
        .iter()
        .map(|(r, n, t)| (to_eye(r, n, t), *n))
        .collect())
}

/// The scaled and signed homography and its four algebraic decompositions.
pub(super) fn algebraic(
    h: &Matrix3<f64>,
    points: &[(Vector2<f64>, Vector2<f64>)],
) -> Option<Vec<Algebraic>> {
    candidates(&normalize(h, points)?)
}

/// Points must be in front of the earlier camera (`N x1 > 0`) and of the
/// later camera: its plane distance `1 + (R N) t` is positive and `(R N) x2 > 0`.
fn in_front(
    r: &Matrix3<f64>,
    n: &Vector3<f64>,
    t: &Vector3<f64>,
    points: &[(Vector2<f64>, Vector2<f64>)],
    config: &PlaneMotionConfig,
) -> bool {
    let later = r * n;
    if 1.0 + later.dot(t) <= 0.0 {
        return false;
    }
    let front = points
        .iter()
        .filter(|(p, q)| {
            n.dot(&Vector3::new(p.x, p.y, 1.0)) > 0.0
                && later.dot(&Vector3::new(q.x, q.y, 1.0)) > 0.0
        })
        .count();
    front as f64 >= config.min_visible_fraction * points.len() as f64
}

/// Scale by the middle singular value and choose the sign with positive depth.
fn normalize(h: &Matrix3<f64>, points: &[(Vector2<f64>, Vector2<f64>)]) -> Option<Matrix3<f64>> {
    let mut values: Vec<f64> = h.singular_values().iter().copied().collect();
    values.sort_by(f64::total_cmp);
    let middle = *values.get(1)?;
    if !(middle.is_finite() && middle > 1e-12) {
        return None;
    }
    let scaled = h / middle;
    let positive = points
        .iter()
        .filter(|(p, q)| {
            Vector3::new(q.x, q.y, 1.0).dot(&(scaled * Vector3::new(p.x, p.y, 1.0))) > 0.0
        })
        .count();
    Some(if 2 * positive >= points.len() {
        scaled
    } else {
        -scaled
    })
}

/// The four algebraic solutions `(R, N, T/d)` in optical-axis-forward axes.
fn candidates(h: &Matrix3<f64>) -> Option<Vec<Algebraic>> {
    let eigen = (h.transpose() * h).symmetric_eigen();
    let mut pairs: Vec<(f64, Vector3<f64>)> = eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
        .map(|(l, v)| (*l, v.into_owned()))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let [(l1, v1), (_, v2), (l3, v3)] = pairs.as_slice() else {
        return None;
    };
    // Without translation the homography is a rotation and has no plane information.
    let spread = l1 - l3;
    if !(spread.is_finite() && spread > 1e-10) {
        return None;
    }
    let (a, b) = ((1.0 - l3).max(0.0).sqrt(), (l1 - 1.0).max(0.0).sqrt());
    let mut solutions = Vec::with_capacity(4);
    for u in [
        (v1 * a + v3 * b) / spread.sqrt(),
        (v1 * a - v3 * b) / spread.sqrt(),
    ] {
        let basis = Matrix3::from_columns(&[*v2, u, v2.cross(&u)]);
        let (hv, hu) = (h * v2, h * u);
        let image = Matrix3::from_columns(&[hv, hu, hv.cross(&hu)]);
        let r = image * basis.transpose();
        let n = v2.cross(&u);
        let t = (h - r) * n;
        solutions.push((r, n, t));
        solutions.push((r, -n, -t));
    }
    Some(solutions)
}

/// Later-camera pose in the earlier camera frame, and the plane normal from
/// the plane toward the earlier camera, both in eye axes.
fn to_eye(r: &Matrix3<f64>, n: &Vector3<f64>, t: &Vector3<f64>) -> PlaneSolution {
    let orientation = FLIP * r.transpose() * FLIP;
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(&orientation));
    PlaneSolution {
        rotation,
        translation_per_distance: FLIP * (-r.transpose() * t),
        normal: Unit::new_normalize(-(FLIP * n)),
    }
}
