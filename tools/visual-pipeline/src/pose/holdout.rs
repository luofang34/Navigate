//! Mask-aware spatial holdout partitioning and ray domain conditioning.

use super::point_jacobian;
use nalgebra::{Matrix3, Matrix6, Vector3};

/// Deterministic 25% spatial holdout partitioner across occupied ground sub-cells.
#[derive(Debug, Clone)]
pub struct MaskAwareSpatialHoldout {
    /// Grid cell subdivisions across width and height [cols, rows].
    pub subdivisions: [usize; 2],
    /// Holdout percentage fraction (nominal 0.25).
    pub holdout_fraction: f64,
}

impl Default for MaskAwareSpatialHoldout {
    fn default() -> Self {
        Self {
            subdivisions: [4, 4],
            holdout_fraction: 0.25,
        }
    }
}

impl MaskAwareSpatialHoldout {
    /// Partitions candidate 2D point indices into fit and holdout sets.
    #[must_use]
    pub fn partition(
        &self,
        points: &[[f64; 2]],
        image_size: [u32; 2],
        mask: Option<&[bool]>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut fit_indices = Vec::new();
        let mut holdout_indices = Vec::new();
        let w = f64::from(image_size[0].max(1));
        let h = f64::from(image_size[1].max(1));
        let cell_w = w / self.subdivisions[0] as f64;
        let cell_h = h / self.subdivisions[1] as f64;

        let mut cell_buckets = vec![Vec::new(); self.subdivisions[0] * self.subdivisions[1]];
        for (i, pt) in points.iter().enumerate() {
            if mask.is_some_and(|m| i < m.len() && !m[i]) {
                continue;
            }
            let col = ((pt[0] / cell_w).floor() as usize).min(self.subdivisions[0] - 1);
            let row = ((pt[1] / cell_h).floor() as usize).min(self.subdivisions[1] - 1);
            let cell_idx = row * self.subdivisions[0] + col;
            cell_buckets[cell_idx].push(i);
        }

        for bucket in cell_buckets {
            let n_holdout = ((bucket.len() as f64) * self.holdout_fraction).round() as usize;
            for (idx_in_cell, &pt_idx) in bucket.iter().enumerate() {
                if idx_in_cell < n_holdout {
                    holdout_indices.push(pt_idx);
                } else {
                    fit_indices.push(pt_idx);
                }
            }
        }

        (fit_indices, holdout_indices)
    }
}

/// Analysis of query rays across angular span and normal-matrix conditioning.
#[derive(Debug, Clone)]
pub struct QueryRayDomain {
    /// Total occupied ground subcells.
    pub occupied_subcells: usize,
    /// Maximum ray-angle span in radians.
    pub ray_angle_span_rad: f64,
    /// Condition number of 6-DoF normal matrix.
    pub condition_number: f64,
    /// Minimum eigenvalue of normal matrix.
    pub min_eigenvalue: f64,
}

impl QueryRayDomain {
    /// Computes ray domain metrics from 3D world points, 2D pixels, pose, and intrinsics.
    #[must_use]
    pub fn analyze(
        world_points: &[[f64; 3]],
        pixels: &[[f64; 2]],
        eye: [f64; 3],
        rot: &Matrix3<f64>,
        k: &Matrix3<f64>,
        image_size: [u32; 2],
    ) -> Option<Self> {
        if world_points.len() != pixels.len() || world_points.len() < 6 {
            return None;
        }

        // Count occupied subcells in 4x4 grid
        let w = f64::from(image_size[0].max(1));
        let h = f64::from(image_size[1].max(1));
        let mut occupied = [false; 16];
        for p in pixels {
            let col = ((p[0] / (w / 4.0)).floor() as usize).min(3);
            let row = ((p[1] / (h / 4.0)).floor() as usize).min(3);
            occupied[row * 4 + col] = true;
        }
        let occupied_subcells = occupied.iter().filter(|&&v| v).count();

        // Compute ray angle span
        let fx = k[(0, 0)];
        let fy = k[(1, 1)];
        let cx = k[(0, 2)];
        let cy = k[(1, 2)];
        let mut rays: Vec<Vector3<f64>> = Vec::with_capacity(pixels.len());
        for p in pixels {
            let rx = (p[0] - cx) / fx;
            let ry = (p[1] - cy) / fy;
            let r = Vector3::new(rx, ry, 1.0).normalize();
            rays.push(r);
        }

        let mut max_span = 0.0_f64;
        for i in 0..rays.len() {
            for j in (i + 1)..rays.len() {
                let dot = rays[i].dot(&rays[j]).clamp(-1.0, 1.0);
                let angle = dot.acos();
                if angle > max_span {
                    max_span = angle;
                }
            }
        }

        // Compute 6-DoF normal matrix via point_jacobian
        let eye_v = Vector3::from(eye);
        let mut normal_matrix = Matrix6::<f64>::zeros();
        let mut valid_count = 0usize;
        for i in 0..world_points.len() {
            if let Some((_, _, ju, jv)) = point_jacobian(world_points[i], pixels[i], eye_v, rot, k)
            {
                normal_matrix += ju * ju.transpose() + jv * jv.transpose();
                valid_count += 1;
            }
        }

        if valid_count < 6 {
            return None;
        }

        let eigen = normal_matrix.symmetric_eigen();
        let min_ev = eigen.eigenvalues.min();
        let max_ev = eigen.eigenvalues.max();
        let cond = if min_ev > 1e-12 {
            max_ev / min_ev
        } else {
            f64::INFINITY
        };

        Some(Self {
            occupied_subcells,
            ray_angle_span_rad: max_span,
            condition_number: cond,
            min_eigenvalue: min_ev,
        })
    }

    /// Verifies if conditioning satisfies non-degeneracy criteria.
    #[must_use]
    pub fn is_well_conditioned(&self) -> bool {
        self.occupied_subcells >= 4 && self.min_eigenvalue >= 1e-6 && self.condition_number <= 1e10
    }
}
