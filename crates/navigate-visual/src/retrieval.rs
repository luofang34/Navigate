//! Local planar retrieval proposals. Surface verification controls acceptance.
use crate::{CameraModel, CameraPose, VisualError};
use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector2, Vector3};

/// A pixel associated with a terrain sample in a declared map frame.
#[derive(Clone, Debug)]
pub struct GroundCorrespondence {
    /// East, north, up in metres. This is map evidence, not certified scene geometry.
    pub world: Vector3<f64>,
    /// Observed pixel centre.
    pub query: Vector2<f64>,
}

/// A retrieval proposal which still requires geometric and prior checks.
#[derive(Clone, Debug)]
pub struct RetrievalProposal {
    /// Arbitrarily oriented candidate camera pose.
    pub pose: CameraPose,
    /// Number of local planar inliers. This is not geographic confidence.
    pub inliers: usize,
}

/// Propose a camera from approximately planar terrain correspondences.
///
/// This method is one retrieval implementation. It can fail for oblique views
/// with substantial depth variation. Final surface verification is separate.
///
/// # Errors
/// Rejects invalid calibration, nonfinite evidence, or more than 4096 matches.
pub fn planar_proposal(
    camera: &CameraModel,
    pairs: &[GroundCorrespondence],
) -> Result<Option<RetrievalProposal>, VisualError> {
    camera.validate()?;
    if pairs.len() > 4096
        || pairs
            .iter()
            .any(|p| p.world.iter().chain(p.query.iter()).any(|v| !v.is_finite()))
    {
        return Err(VisualError::Invalid {
            field: "retrieval correspondences",
        });
    }
    Ok(proposals(*camera, pairs))
}
fn proposals(camera: CameraModel, pairs: &[GroundCorrespondence]) -> Option<RetrievalProposal> {
    if pairs.len() < 8 {
        return None;
    }
    let origin = pairs
        .iter()
        .fold(Vector3::zeros(), |sum, p| sum + Vector3::from(p.world))
        / pairs.len() as f64;
    let normalized: Vec<_> = pairs
        .iter()
        .map(|p| {
            (
                (Vector2::new(p.world[0], p.world[1]) - Vector2::new(origin.x, origin.y)) / 100.0,
                Vector2::new(
                    (p.query[0] - camera.cx) / camera.fx,
                    (p.query[1] - camera.cy) / camera.fy,
                ),
            )
        })
        .collect();
    let best = planar_consensus(camera, &normalized);
    if best.len() < 8 {
        return None;
    }
    let h = fit(&normalized, &best)?;
    let pose = decompose(h, origin)?;
    Some(RetrievalProposal {
        pose: CameraPose {
            position: pose.0,
            orientation: pose.1,
        },
        inliers: best.len(),
    })
}
fn planar_consensus(
    camera: CameraModel,
    normalized: &[(Vector2<f64>, Vector2<f64>)],
) -> Vec<usize> {
    let mut seed = 0x915a_31d7_u64;
    let mut best = Vec::new();
    let mut budget = 4096_usize;
    let mut trial = 0_usize;
    while trial < budget {
        trial = trial.wrapping_add(1);
        let mut sample = Vec::new();
        while sample.len() < 4 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let i = (seed >> 32) as usize % normalized.len();
            if !sample.contains(&i) {
                sample.push(i);
            }
        }
        if let Some(h) = fit(normalized, &sample) {
            let inliers: Vec<_> = normalized
                .iter()
                .enumerate()
                .filter_map(|(i, (p, q))| {
                    let r = h * Vector3::new(p.x, p.y, 1.0);
                    (r.z.abs() > 1e-10
                        && ((Vector2::new(r.x / r.z, r.y / r.z) - q)
                            .component_mul(&Vector2::new(camera.fx, camera.fy)))
                        .norm()
                            < 4.0)
                        .then_some(i)
                })
                .collect();
            if inliers.len() > best.len() {
                best = inliers;
                // This bound controls compute. It is not geographic confidence.
                let fraction = best.len() as f64 / normalized.len() as f64;
                let needed = (0.001_f64.ln() / (1.0 - fraction.powi(4)).ln()).ceil() as usize;
                budget = budget.min(needed.max(64));
            }
        }
    }
    best
}
fn fit(points: &[(Vector2<f64>, Vector2<f64>)], indices: &[usize]) -> Option<Matrix3<f64>> {
    let mut normal = SMatrix::<f64, 8, 8>::zeros();
    let mut rhs = SVector::<f64, 8>::zeros();
    for &i in indices {
        let (p, q) = points[i];
        let a = SVector::<f64, 8>::from_row_slice(&[
            p.x,
            p.y,
            1.0,
            0.0,
            0.0,
            0.0,
            -q.x * p.x,
            -q.x * p.y,
        ]);
        let b = SVector::<f64, 8>::from_row_slice(&[
            0.0,
            0.0,
            0.0,
            p.x,
            p.y,
            1.0,
            -q.y * p.x,
            -q.y * p.y,
        ]);
        normal += a * a.transpose() + b * b.transpose();
        rhs += a * q.x + b * q.y;
    }
    let h = normal.lu().solve(&rhs)?;
    Some(Matrix3::new(
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0,
    ))
}
fn decompose(h: Matrix3<f64>, origin: Vector3<f64>) -> Option<(Vector3<f64>, UnitQuaternion<f64>)> {
    let a = h.column(0).into_owned() / 100.0;
    let b = h.column(1).into_owned() / 100.0;
    let scale = 2.0 / (a.norm() + b.norm());
    for sign in [1.0, -1.0] {
        let r1 = (a * sign).try_normalize(1e-10)?;
        let b = b * sign;
        let r2 = (b - r1 * r1.dot(&b)).try_normalize(1e-10)?;
        let r3 = r1.cross(&r2);
        let r = Matrix3::from_columns(&[r1, r2, r3]);
        let t = h.column(2).into_owned() * scale * sign;
        let position = origin - r.transpose() * t;
        if position.z > origin.z + 5.0 && position.z < origin.z + 10000.0 {
            let rotation = r.transpose() * Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
            return Some((position, UnitQuaternion::from_matrix(&rotation)));
        }
    }
    None
}
#[cfg(test)]
mod tests;
