//! Near-nadir similarity seeds. Rendered surface checks control acceptance.
use super::{GroundCorrespondence, RetrievalProposal};
use crate::{CameraModel, CameraPose, VisualError};
use nalgebra::{UnitQuaternion, Vector2, Vector3};

/// Propose a near-nadir camera from a robust image-to-map similarity fit.
///
/// This optional initializer estimates scale, yaw, and horizontal translation.
/// It does not verify scene geometry or restrict a subsequent pose solver.
/// `max_error_px` bounds seed residuals, not final pose acceptance or accuracy.
/// Terrain heights supply the vertical origin. They are not scene ground truth.
///
/// # Errors
/// Rejects invalid calibration, nonfinite evidence, more than 4096 matches,
/// or a residual threshold outside `(0, 64]` pixels.
pub fn nadir_similarity_proposal(
    camera: &CameraModel,
    pairs: &[GroundCorrespondence],
    max_error_px: f64,
) -> Result<Option<RetrievalProposal>, VisualError> {
    camera.validate()?;
    if !max_error_px.is_finite() || max_error_px <= 0.0 || max_error_px > 64.0 {
        return Err(VisualError::Invalid {
            field: "similarity seed threshold",
        });
    }
    if pairs.len() > 4096
        || pairs
            .iter()
            .any(|p| p.world.iter().chain(p.query.iter()).any(|v| !v.is_finite()))
    {
        return Err(VisualError::Invalid {
            field: "retrieval correspondences",
        });
    }
    if pairs.len() < 8 {
        return Ok(None);
    }
    let origin = pairs.iter().map(|p| p.world.xy()).sum::<Vector2<f64>>() / pairs.len() as f64;
    let points: Vec<_> = pairs
        .iter()
        .map(|p| {
            (
                p.world.xy() - origin,
                Vector2::new(
                    (p.query.x - camera.cx) / camera.fx,
                    (camera.cy - p.query.y) / camera.fy,
                ),
            )
        })
        .collect();
    let mut indices = consensus(camera, &points, max_error_px);
    let mut transform = None;
    for _ in 0..3 {
        if indices.len() < 8 {
            return Ok(None);
        }
        let Some(fitted) = fit(&points, &indices) else {
            return Ok(None);
        };
        indices = inliers(camera, &points, fitted, max_error_px);
        transform = Some(fitted);
    }
    Ok(transform.and_then(|t| pose(t, origin, pairs, &indices)))
}

#[derive(Clone, Copy)]
struct Similarity {
    coefficient: Vector2<f64>,
    translation: Vector2<f64>,
}

fn multiply(a: Vector2<f64>, b: Vector2<f64>) -> Vector2<f64> {
    Vector2::new(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y)
}

fn fit(points: &[(Vector2<f64>, Vector2<f64>)], indices: &[usize]) -> Option<Similarity> {
    let (world, query) = indices
        .iter()
        .fold((Vector2::zeros(), Vector2::zeros()), |(w, q), &i| {
            (w + points[i].0, q + points[i].1)
        });
    let world = world / indices.len() as f64;
    let query = query / indices.len() as f64;
    let (numerator, denominator) = indices.iter().fold((Vector2::zeros(), 0.0), |(n, d), &i| {
        let w = points[i].0 - world;
        (
            n + multiply(Vector2::new(w.x, -w.y), points[i].1 - query),
            d + w.norm_squared(),
        )
    });
    if denominator < 1e-8 {
        return None;
    }
    let coefficient = numerator / denominator;
    if coefficient.norm_squared() < 1e-16 {
        return None;
    }
    Some(Similarity {
        coefficient,
        translation: query - multiply(coefficient, world),
    })
}

fn inliers(
    camera: &CameraModel,
    points: &[(Vector2<f64>, Vector2<f64>)],
    t: Similarity,
    threshold: f64,
) -> Vec<usize> {
    points
        .iter()
        .enumerate()
        .filter_map(|(i, (world, query))| {
            let r = multiply(t.coefficient, *world) + t.translation - query;
            ((r.x * camera.fx).hypot(r.y * camera.fy) < threshold).then_some(i)
        })
        .collect()
}

fn consensus(
    camera: &CameraModel,
    points: &[(Vector2<f64>, Vector2<f64>)],
    threshold: f64,
) -> Vec<usize> {
    let mut state = 0x915a_31d7_u64;
    let mut best = Vec::new();
    for _ in 0..1024 {
        let mut sample = [0; 2];
        for index in &mut sample {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *index = (state >> 32) as usize % points.len();
        }
        // Nearly coincident map points do not constrain scale and yaw.
        if (points[sample[0]].0 - points[sample[1]].0).norm() < 2.0 {
            continue;
        }
        if let Some(t) = fit(points, &sample) {
            let selected = inliers(camera, points, t, threshold);
            if selected.len() > best.len() {
                best = selected;
            }
        }
    }
    best
}

fn pose(
    t: Similarity,
    origin: Vector2<f64>,
    pairs: &[GroundCorrespondence],
    indices: &[usize],
) -> Option<RetrievalProposal> {
    if indices.len() < 8 {
        return None;
    }
    let height = 1.0 / t.coefficient.norm();
    if !(5.0..10000.0).contains(&height) {
        return None;
    }
    let inverse = Vector2::new(t.coefficient.x, -t.coefficient.y) / t.coefficient.norm_squared();
    let center = origin - multiply(inverse, t.translation);
    let terrain = indices.iter().map(|&i| pairs[i].world.z).sum::<f64>() / indices.len() as f64;
    let position = Vector3::new(center.x, center.y, terrain + height);
    if position.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(RetrievalProposal {
        pose: CameraPose {
            position,
            orientation: UnitQuaternion::from_euler_angles(
                0.0,
                0.0,
                -t.coefficient.y.atan2(t.coefficient.x),
            ),
        },
        inliers: indices.len(),
        separated_inliers: super::separated_support(pairs, indices),
    })
}

#[cfg(test)]
mod tests;
