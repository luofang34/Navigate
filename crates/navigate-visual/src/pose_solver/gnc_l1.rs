//! Deterministic L1 IRLS and graduated non-convexity camera pose solver.

use super::{Matrix6, Vector6, point_jacobian};
use crate::{CameraModel, CameraPose, PosePrior, VisualError};
use nalgebra::{Vector2, Vector3};

/// Configuration for deterministic L1 IRLS and graduated non-convexity pose estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GncPoseConfig {
    /// Maximum reprojection residual in pixels for an accepted inlier.
    pub inlier_threshold_px: f64,
    /// Minimum residual floor in pixels for Stage 1 isotropic L1 weights.
    pub l1_floor_px: f64,
    /// Maximum damped Cholesky steps in Stage 1 isotropic L1 IRLS.
    pub l1_iterations: usize,
    /// Number of continuation stages in Stage 2 graduated non-convexity.
    pub gnc_stages: usize,
    /// Minimum accepted inlier count.
    pub min_inliers: usize,
}

impl Default for GncPoseConfig {
    fn default() -> Self {
        Self {
            inlier_threshold_px: 3.5,
            l1_floor_px: 0.5,
            l1_iterations: 6,
            gnc_stages: 5,
            min_inliers: 12,
        }
    }
}

impl GncPoseConfig {
    fn validate(&self) -> Result<(), VisualError> {
        let checks = [
            (
                self.inlier_threshold_px.is_finite() && self.inlier_threshold_px > 0.0,
                "inlier_threshold_px",
            ),
            (
                self.l1_floor_px.is_finite() && self.l1_floor_px > 0.0,
                "l1_floor_px",
            ),
            (self.l1_iterations > 0, "l1_iterations"),
            (self.gnc_stages > 0, "gnc_stages"),
            (self.min_inliers >= 6, "min_inliers"),
        ];
        for (ok, field) in checks {
            if !ok {
                return Err(VisualError::Invalid { field });
            }
        }
        Ok(())
    }
}

/// Accepted camera pose and support summary from [`solve_pose_l1_gnc`].
#[derive(Clone, Debug)]
pub struct GncPoseSolution {
    /// Camera pose in the local east, north, up frame.
    pub pose: CameraPose,
    /// Inlier flag for each input correspondence.
    pub inlier_mask: Vec<bool>,
    /// Count of accepted inliers.
    pub inlier_count: usize,
    /// Root-mean-square reprojection residual in pixels on accepted inliers.
    pub inlier_rms_px: f64,
    /// Count of `6 x 6` Cholesky factorizations.
    pub cholesky_solves: usize,
}

#[derive(Clone, Copy)]
enum WeightKernel {
    L1 { floor_px: f64, behind_px: f64 },
    Gnc { scale_sq: f64 },
    Tls { threshold_px: f64 },
}

impl WeightKernel {
    fn weight(self, r: f64) -> f64 {
        match self {
            Self::L1 { floor_px, .. } => 1.0 / r.max(floor_px),
            Self::Gnc { scale_sq } => (scale_sq / (r * r + scale_sq)).powi(2),
            Self::Tls { threshold_px } => f64::from(u8::from(r <= threshold_px)),
        }
    }

    fn loss(self, residual_px: Option<f64>) -> f64 {
        match (self, residual_px) {
            (Self::L1 { behind_px, .. }, None) => behind_px,
            (Self::L1 { floor_px, .. }, Some(r)) if r < floor_px => {
                0.5 * r * r / floor_px + 0.5 * floor_px
            }
            (Self::L1 { .. }, Some(r)) => r,
            (Self::Gnc { scale_sq }, None) => 0.5 * scale_sq,
            (Self::Gnc { scale_sq }, Some(r)) => 0.5 * scale_sq * r * r / (r * r + scale_sq),
            (Self::Tls { threshold_px }, None) => 0.5 * threshold_px * threshold_px,
            (Self::Tls { threshold_px }, Some(r)) => 0.5 * r.min(threshold_px).powi(2),
        }
    }
}

struct SolverState<'a> {
    camera: &'a CameraModel,
    world: &'a [Vector3<f64>],
    pixels: &'a [Vector2<f64>],
    pose: CameraPose,
    damping: f64,
    cholesky_solves: usize,
}

impl SolverState<'_> {
    fn step(&mut self, kernel: WeightKernel) -> Result<f64, VisualError> {
        let mut h = Matrix6::zeros();
        let mut b = Vector6::zeros();
        let mut active = 0usize;
        let world_to_eye = self.pose.orientation.inverse().to_rotation_matrix();
        for (&world, &pixel) in self.world.iter().zip(self.pixels.iter()) {
            let Some((err, jac)) =
                point_jacobian(self.camera, &world_to_eye, self.pose.position, world, pixel)
            else {
                continue;
            };
            let w = kernel.weight(err.norm());
            if w > 0.0 {
                active += 1;
                h += jac.transpose() * jac * w;
                b += jac.transpose() * err * w;
            }
        }
        if active < 6 {
            return Err(VisualError::DegenerateGeometry);
        }
        if b.norm() < 1e-9 {
            return Ok(0.0);
        }
        let reg = h + Matrix6::from_diagonal(&h.diagonal().map(|v| self.damping * v.max(1e-9)));
        self.cholesky_solves += 1;
        let delta = reg
            .cholesky()
            .ok_or(VisualError::DegenerateGeometry)?
            .solve(&b);
        if !delta.iter().all(|v| v.is_finite()) {
            return Err(VisualError::DegenerateGeometry);
        }
        let step_norm = delta.norm();
        if step_norm < 1e-6 {
            return Ok(step_norm);
        }
        let current_cost = self.cost(&self.pose, kernel);
        let candidate = self.pose.increment(&delta);
        if self.cost(&candidate, kernel) < current_cost {
            self.pose = candidate;
            self.damping = (self.damping * 0.5).max(1e-6);
        } else {
            self.damping = (self.damping * 8.0).min(1e6);
        }
        Ok(step_norm)
    }

    fn cost(&self, pose: &CameraPose, kernel: WeightKernel) -> f64 {
        self.world
            .iter()
            .zip(self.pixels.iter())
            .map(|(&w, &px)| kernel.loss(self.camera.project(pose, w).map(|p| (px - p).norm())))
            .sum()
    }

    fn max_and_count(&self, cap_px: f64, threshold_px: f64) -> (f64, usize) {
        let mut max_r = 0.0_f64;
        let mut within = 0usize;
        for (&w, &px) in self.world.iter().zip(self.pixels.iter()) {
            if let Some(p) = self.camera.project(&self.pose, w) {
                let r = (px - p).norm();
                max_r = max_r.max(r.min(cap_px));
                within += usize::from(r <= threshold_px);
            }
        }
        (max_r, within)
    }
}

/// Solve a 6-DoF camera pose deterministically with L1 IRLS and graduated non-convexity.
///
/// Stage 1 runs isotropic L1 IRLS from `prior.pose` without random sampling.
/// Stage 2 decreases the Geman-McClure continuation parameter down to
/// `config.inlier_threshold_px`. A final step refines the pose on the accepted inliers.
///
/// # Errors
///
/// Rejects invalid parameters, correspondence count mismatches, insufficient inliers,
/// degenerate camera geometry, and poses outside `prior`.
pub fn solve_pose_l1_gnc(
    camera: &CameraModel,
    world_points: &[Vector3<f64>],
    image_pixels: &[Vector2<f64>],
    prior: &PosePrior,
    config: &GncPoseConfig,
) -> Result<GncPoseSolution, VisualError> {
    camera.validate()?;
    prior.validate()?;
    config.validate()?;
    if world_points.len() != image_pixels.len() {
        return Err(VisualError::Invalid {
            field: "correspondence length",
        });
    }
    if world_points
        .iter()
        .any(|p| !p.iter().all(|v| v.is_finite()))
        || image_pixels
            .iter()
            .any(|uv| !uv.iter().all(|v| v.is_finite()))
    {
        return Err(VisualError::Invalid {
            field: "correspondence coordinates",
        });
    }
    if world_points.len() < config.min_inliers {
        return Err(VisualError::InsufficientMatches {
            found: world_points.len(),
            required: config.min_inliers,
        });
    }

    let mut state = SolverState {
        camera,
        world: world_points,
        pixels: image_pixels,
        pose: prior.pose,
        damping: 1e-3,
        cholesky_solves: 0,
    };
    let span = 2.0 * f64::from(camera.width.max(camera.height));
    let l1 = WeightKernel::L1 {
        floor_px: config.l1_floor_px,
        behind_px: 2.0 * span,
    };
    for _ in 0..config.l1_iterations {
        if state.step(l1)? < 1e-5 {
            break;
        }
    }

    let c = config.inlier_threshold_px;
    let mu_start = (state.max_and_count(span, c).0 / c).powi(2).max(4.0);
    for stage in 0..config.gnc_stages {
        let mu = if config.gnc_stages == 1 {
            1.0
        } else {
            mu_start.powf(1.0 - (stage as f64) / ((config.gnc_stages - 1) as f64))
        };
        let gnc = WeightKernel::Gnc {
            scale_sq: mu * c * c,
        };
        state.damping = 1e-3;
        if stage + 1 < config.gnc_stages {
            if state.step(gnc)? >= 1e-3 {
                let _second = state.step(gnc)?;
            }
        } else {
            let _first = state.step(gnc)?;
            if state.max_and_count(span, c).1 >= config.min_inliers {
                state.damping = 1e-4;
                let _polish = state.step(WeightKernel::Tls { threshold_px: c })?;
            }
        }
    }

    state.pose.validate()?;
    let meters = (state.pose.position - prior.pose.position).norm();
    let radians = state.pose.orientation.angle_to(&prior.pose.orientation);
    if !meters.is_finite() || !radians.is_finite() {
        return Err(VisualError::DegenerateGeometry);
    }
    if meters > prior.position_radius_m || radians > prior.attitude_radius_rad {
        return Err(VisualError::OutsidePrior { meters, radians });
    }

    let mut inlier_mask = vec![false; world_points.len()];
    let (mut inlier_count, mut sq_sum, mut h_inliers) = (0usize, 0.0, Matrix6::zeros());
    let w2e = state.pose.orientation.inverse().to_rotation_matrix();
    for (i, (&w, &px)) in world_points.iter().zip(image_pixels.iter()).enumerate() {
        if let Some((err, jac)) = point_jacobian(camera, &w2e, state.pose.position, w, px)
            && err.norm_squared() <= c * c
        {
            inlier_mask[i] = true;
            inlier_count += 1;
            sq_sum += err.norm_squared();
            h_inliers += jac.transpose() * jac;
        }
    }
    if inlier_count < config.min_inliers {
        return Err(VisualError::InsufficientMatches {
            found: inlier_count,
            required: config.min_inliers,
        });
    }

    let eig = h_inliers.symmetric_eigen().eigenvalues;
    let cond = eig.max() / eig.min();
    if eig.min() <= 1e-10 || !cond.is_finite() || cond > 1e10 {
        return Err(VisualError::DegenerateGeometry);
    }

    Ok(GncPoseSolution {
        pose: state.pose,
        inlier_mask,
        inlier_count,
        inlier_rms_px: (sq_sum / (inlier_count as f64)).sqrt(),
        cholesky_solves: state.cholesky_solves,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Unit, UnitQuaternion};

    fn test_camera() -> CameraModel {
        CameraModel {
            width: 960,
            height: 720,
            fx: 760.0,
            fy: 760.0,
            cx: 479.5,
            cy: 359.5,
        }
    }

    fn synthetic_correspondences(
        camera: &CameraModel,
        truth: &CameraPose,
    ) -> (Vec<Vector3<f64>>, Vec<Vector2<f64>>, Vec<bool>) {
        let mut world_points = Vec::with_capacity(64);
        let mut image_pixels = Vec::with_capacity(64);
        for row in 0..8 {
            for col in 0..8 {
                let pixel =
                    Vector2::new(80.0 + f64::from(col) * 112.0, 60.0 + f64::from(row) * 84.0);
                let depth = 850.0 + 160.0 * (f64::from(row) * 0.9 + f64::from(col) * 0.6).sin();
                world_points.push(camera.unproject(truth, pixel, depth));
                image_pixels.push(pixel);
            }
        }
        let mut expected_inliers = vec![true; 64];
        let mut corrupted = 0usize;
        for idx in 0..64 {
            if ((idx * 7 + 3) % 16 < 7 || idx >= 60) && corrupted < 29 {
                expected_inliers[idx] = false;
                let mag = 100.0 + (corrupted as f64) * (200.0 / 28.0);
                let angle = 0.35 + (corrupted as f64) * 0.19;
                let offset = Vector2::new(mag * angle.cos(), mag * angle.sin());
                assert!((99.999..=300.001).contains(&offset.norm()));
                image_pixels[idx] += offset;
                corrupted += 1;
            }
        }
        assert_eq!(corrupted, 29);
        (world_points, image_pixels, expected_inliers)
    }

    #[test]
    fn recovers_exact_pose_under_45_percent_gross_adversarial_outliers() {
        let cam = test_camera();
        let truth = CameraPose {
            position: Vector3::new(140.0, -95.0, 980.0),
            orientation: UnitQuaternion::from_euler_angles(0.22, -0.14, 0.48),
        };
        let (pts, pxs, expected) = synthetic_correspondences(&cam, &truth);

        let offset_pos = Vector3::new(64.0_f64, -38.4, 28.8);
        assert!((offset_pos.norm() - 80.0).abs() < 1e-12);
        let axis = Unit::new_normalize(Vector3::new(1.0, -0.7, 0.5));
        let rot_err = UnitQuaternion::from_axis_angle(&axis, 6.0_f64.to_radians());
        let initial = CameraPose {
            position: truth.position + offset_pos,
            orientation: truth.orientation * rot_err,
        };
        let prior = PosePrior {
            pose: initial,
            position_radius_m: 150.0,
            attitude_radius_rad: 0.25,
        };

        let cfg = GncPoseConfig::default();
        let a = solve_pose_l1_gnc(&cam, &pts, &pxs, &prior, &cfg).expect("must converge");
        let b = solve_pose_l1_gnc(&cam, &pts, &pxs, &prior, &cfg).expect("must converge");

        let pos_err = (a.pose.position - truth.position).norm();
        let att_err = a.pose.orientation.angle_to(&truth.orientation).to_degrees();
        assert!(pos_err < 0.05, "position error {pos_err} m");
        assert!(att_err < 0.01, "attitude error {att_err} deg");
        assert_eq!(a.inlier_count, 35);
        assert_eq!(a.inlier_mask, expected);
        assert!(a.cholesky_solves <= 16, "solves {}", a.cholesky_solves);

        assert_eq!(a.pose.position, b.pose.position);
        assert_eq!(a.pose.orientation.coords, b.pose.orientation.coords);
        assert_eq!(a.inlier_mask, b.inlier_mask);
        assert_eq!(a.inlier_rms_px.to_bits(), b.inlier_rms_px.to_bits());
        assert_eq!(a.cholesky_solves, b.cholesky_solves);
    }

    #[test]
    fn rejects_insufficient_correspondences_and_outside_prior() {
        let cam = test_camera();
        let truth = CameraPose {
            position: Vector3::new(0.0, 0.0, 900.0),
            orientation: UnitQuaternion::identity(),
        };
        let (pts, pxs, _) = synthetic_correspondences(&cam, &truth);
        let cfg = GncPoseConfig::default();

        let ok_prior = PosePrior {
            pose: truth,
            position_radius_m: 100.0,
            attitude_radius_rad: 0.2,
        };
        let few = solve_pose_l1_gnc(&cam, &pts[..11], &pxs[..11], &ok_prior, &cfg);
        assert!(matches!(
            few,
            Err(VisualError::InsufficientMatches {
                found: 11,
                required: 12
            })
        ));

        let shifted = CameraPose {
            position: truth.position + Vector3::new(80.0, 0.0, 0.0),
            orientation: truth.orientation,
        };
        let tight = PosePrior {
            pose: shifted,
            position_radius_m: 25.0,
            attitude_radius_rad: 0.2,
        };
        let out = solve_pose_l1_gnc(&cam, &pts, &pxs, &tight, &cfg);
        assert!(matches!(out, Err(VisualError::OutsidePrior { .. })));
    }
}
