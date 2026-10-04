//! First-order propagation of pixel noise to the plane normal.
//!
//! The normal depends on every match through the homography fit and its
//! decomposition. The derivative for each match coordinate comes from a
//! small step: only that match changes its term in the normal equations,
//! so each step needs one 8 by 8 solve. With independent pixel noise of
//! `sigma_px`, the normal variance is the sum of the squared derivatives.

use super::decompose::algebraic;
use crate::retrieval::{fit_terms, solve_fit};
use nalgebra::{SMatrix, SVector, Vector2, Vector3};

/// Step of one match coordinate, in pixels.
const STEP_PX: f64 = 0.05;

/// One-sigma direction error of each nominal normal (optical-axis-forward axes).
pub(super) fn normal_sigmas(
    points: &[(Vector2<f64>, Vector2<f64>)],
    normals: &[Vector3<f64>],
    focal_px: [f64; 2],
    sigma_px: f64,
) -> Vec<f64> {
    let mut normal = SMatrix::<f64, 8, 8>::zeros();
    let mut rhs = SVector::<f64, 8>::zeros();
    for (p, q) in points {
        let (n, r) = fit_terms(p, q);
        normal += n;
        rhs += r;
    }
    let mut sums = vec![0.0; normals.len()];
    let steps = [
        (Vector2::new(STEP_PX / focal_px[0], 0.0), Vector2::zeros()),
        (Vector2::new(0.0, STEP_PX / focal_px[1]), Vector2::zeros()),
        (Vector2::zeros(), Vector2::new(STEP_PX / focal_px[0], 0.0)),
        (Vector2::zeros(), Vector2::new(0.0, STEP_PX / focal_px[1])),
    ];
    for (p, q) in points {
        let (n0, r0) = fit_terms(p, q);
        for (dp, dq) in &steps {
            let (n1, r1) = fit_terms(&(p + dp), &(q + dq));
            let Some(h) = solve_fit(&(normal - n0 + n1), &(rhs - r0 + r1)) else {
                return vec![f64::INFINITY; normals.len()];
            };
            let Some(candidates) = algebraic(&h, points) else {
                return vec![f64::INFINITY; normals.len()];
            };
            for (sum, nominal) in sums.iter_mut().zip(normals) {
                let nearest = candidates
                    .iter()
                    .map(|(_, n, _)| n)
                    .max_by(|a, b| a.dot(nominal).total_cmp(&b.dot(nominal)));
                if let Some(moved) = nearest {
                    *sum += ((moved - nominal) / STEP_PX).norm_squared();
                }
            }
        }
    }
    sums.into_iter().map(|s| sigma_px * s.sqrt()).collect()
}
