//! Physical plausibility validation, Fisher conditioning, and prior bounding for corridor anchors.

use nalgebra::{Matrix3, Matrix6, Vector3};
use navigate_visual::{
    LocalFrame, SearchPrior,
    holdout_pnp::{PoseCandidate, point_jacobian, reproj_error, spatial_holdout},
};
use navigate_visual_render::TerrainAtlas;

/// Read-only terrain surface adapter for physical plausibility checks and seed projection.
pub trait CorridorTerrain {
    /// Convert local `[east_m, north_m, up_m]` coordinates into WGS84 `[lat_deg, lon_deg, alt_m]`.
    fn geodetic(&self, local_enu: [f64; 3]) -> [f64; 3];

    /// Convert WGS84 `[lat_deg, lon_deg, alt_m]` into local `[east_m, north_m, up_m]` coordinates.
    fn local(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3];

    /// Sample bare-earth terrain elevation in meters at `(lat_deg, lon_deg)`.
    fn sample_dem(&self, lat_deg: f64, lon_deg: f64) -> Option<f64>;

    /// Build a world-to-camera rotation matrix at `(lat_deg, lon_deg)` for `(heading, down, roll)` in degrees.
    fn camera_rotation(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        heading_deg: f64,
        down_deg: f64,
        roll_deg: f64,
    ) -> Matrix3<f64>;
}

impl CorridorTerrain for (LocalFrame, &TerrainAtlas) {
    fn geodetic(&self, local_enu: [f64; 3]) -> [f64; 3] {
        self.0
            .geodetic(Vector3::new(local_enu[0], local_enu[1], local_enu[2]))
    }

    fn local(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
        self.0
            .local([lat_deg, lon_deg, alt_m])
            .map_or([0.0, 0.0, alt_m], |v| [v.x, v.y, v.z])
    }

    fn sample_dem(&self, lat_deg: f64, lon_deg: f64) -> Option<f64> {
        crate::scene::height_at(self.1, [lat_deg, lon_deg])
    }

    fn camera_rotation(
        &self,
        _lat_deg: f64,
        _lon_deg: f64,
        heading_deg: f64,
        down_deg: f64,
        roll_deg: f64,
    ) -> Matrix3<f64> {
        local_camera_rotation(heading_deg, down_deg, roll_deg)
    }
}

/// Build a local East-North-Up world-to-camera rotation matrix from angles in degrees.
#[must_use]
pub fn local_camera_rotation(heading_deg: f64, down_deg: f64, roll_deg: f64) -> Matrix3<f64> {
    let h = heading_deg.to_radians();
    let d = down_deg.to_radians();
    let r = roll_deg.to_radians();
    let fwd = Vector3::new(h.sin() * d.cos(), h.cos() * d.cos(), -d.sin());
    let right0 = Vector3::new(h.cos(), -h.sin(), 0.0);
    let down0 = fwd.cross(&right0).normalize();
    let right = right0 * r.cos() + down0 * r.sin();
    let down = -right0 * r.sin() + down0 * r.cos();
    Matrix3::from_rows(&[right.transpose(), down.transpose(), fwd.transpose()])
}

/// Failure reasons when physical plausibility validation rejects a candidate pose.
#[derive(Debug, Clone, PartialEq)]
pub enum PlausibilityError {
    /// Clearance above terrain is below the minimum required height.
    SubSurfaceClearance {
        /// Observed above-ground clearance in meters.
        agl_m: f64,
    },
    /// Fraction of correspondences in front of the camera is below the minimum limit.
    LowCheiralityFraction {
        /// Observed positive-depth fraction.
        fraction: f64,
    },
    /// Minimum eigenvalue of the 6-DoF Fisher information matrix is below the minimum limit.
    IllConditionedFisher {
        /// Observed minimum eigenvalue.
        min_eigenvalue: f64,
    },
    /// Holdout reprojection root-mean-square error exceeds the maximum limit.
    ExcessiveReprojectionError {
        /// Observed root-mean-square error in pixels.
        rmse_px: f64,
    },
    /// Candidate pose coordinates, rotation, or verification metrics contain non-finite values.
    NonFinitePose,
}

/// Configuration parameters for physical plausibility validation.
#[derive(Debug, Clone)]
pub struct PhysicalPlausibilityConfig {
    /// Minimum clearance above the digital elevation model in meters (no upper ceiling).
    pub min_agl_m: f64,
    /// Minimum fraction of correspondences located in front of the camera.
    pub min_positive_depth_fraction: f64,
    /// Minimum acceptable eigenvalue of the 6-DoF Fisher information matrix.
    pub min_fisher_eigenvalue: f64,
    /// Maximum allowable holdout reprojection root-mean-square error in pixels.
    pub max_reprojection_rmse_px: f64,
    /// Standard-deviation multiplier for prior covariance bounding when the prior is trusted.
    pub prior_sigma_scale: f64,
}

impl Default for PhysicalPlausibilityConfig {
    fn default() -> Self {
        Self {
            min_agl_m: 5.0,
            min_positive_depth_fraction: 0.98,
            min_fisher_eigenvalue: 1e-6,
            max_reprojection_rmse_px: 4.0,
            prior_sigma_scale: 3.0,
        }
    }
}

/// Verify scalar physical plausibility criteria without a prior constraint.
///
/// # Errors
/// Returns [`PlausibilityError`] if clearance, cheirality, conditioning, or reprojection error fails.
pub fn verify_physical_plausibility(
    eye_msl_m: f64,
    dem_msl_m: f64,
    positive_depth_fraction: f64,
    min_fisher_eigenvalue: f64,
    reprojection_rmse_px: f64,
    config: &PhysicalPlausibilityConfig,
) -> Result<(), PlausibilityError> {
    if !eye_msl_m.is_finite()
        || !dem_msl_m.is_finite()
        || !positive_depth_fraction.is_finite()
        || !min_fisher_eigenvalue.is_finite()
        || !reprojection_rmse_px.is_finite()
    {
        return Err(PlausibilityError::NonFinitePose);
    }
    let agl_m = eye_msl_m - dem_msl_m;
    if agl_m < config.min_agl_m {
        return Err(PlausibilityError::SubSurfaceClearance { agl_m });
    }
    if positive_depth_fraction < config.min_positive_depth_fraction {
        return Err(PlausibilityError::LowCheiralityFraction {
            fraction: positive_depth_fraction,
        });
    }
    if min_fisher_eigenvalue < config.min_fisher_eigenvalue {
        return Err(PlausibilityError::IllConditionedFisher {
            min_eigenvalue: min_fisher_eigenvalue,
        });
    }
    if reprojection_rmse_px > config.max_reprojection_rmse_px {
        return Err(PlausibilityError::ExcessiveReprojectionError {
            rmse_px: reprojection_rmse_px,
        });
    }
    Ok(())
}

/// Compute the minimum eigenvalue of the 6-DoF Fisher information matrix.
#[must_use]
pub fn compute_fisher_min_eigenvalue(cand: &PoseCandidate) -> f64 {
    let eye_v = Vector3::from(cand.eye);
    let mut h = Matrix6::<f64>::zeros();
    let mut inlier_count = 0usize;

    for (i, &is_inlier) in cand.inliers.iter().enumerate() {
        if !is_inlier || i >= cand.points.len() || i >= cand.pixels.len() {
            continue;
        }
        if let Some((_, _, ju, jv)) = point_jacobian(
            cand.points[i],
            cand.pixels[i],
            eye_v,
            &cand.rotation,
            &cand.k,
        ) {
            h += ju * ju.transpose() + jv * jv.transpose();
            inlier_count += 1;
        }
    }

    if inlier_count < 6 {
        return 0.0;
    }
    h.symmetric_eigen().eigenvalues.min()
}

/// Compute a symmetric positive-definite `6 x 6` Fisher geometry covariance `[pos_m, rot_rad]`.
#[must_use]
pub fn compute_fisher_geometry_covariance(cand: &PoseCandidate) -> Matrix6<f64> {
    let eye_v = Vector3::from(cand.eye);
    let mut h = Matrix6::<f64>::zeros();
    let mut inlier_count = 0usize;
    let rot_world_from_cam = cand.rotation.transpose();

    for (i, &is_inlier) in cand.inliers.iter().enumerate() {
        if !is_inlier || i >= cand.points.len() || i >= cand.pixels.len() {
            continue;
        }
        if let Some((_, _, ju, jv)) = point_jacobian(
            cand.points[i],
            cand.pixels[i],
            eye_v,
            &cand.rotation,
            &cand.k,
        ) {
            let jr_u = rot_world_from_cam * Vector3::new(ju[0], ju[1], ju[2]);
            let jr_v = rot_world_from_cam * Vector3::new(jv[0], jv[1], jv[2]);
            let mut gu = nalgebra::Vector6::new(ju[3], ju[4], ju[5], jr_u.x, jr_u.y, jr_u.z);
            let mut gv = nalgebra::Vector6::new(jv[3], jv[4], jv[5], jr_v.x, jr_v.y, jr_v.z);
            for idx in 3..6 {
                gu[idx] *= 1e-3;
                gv[idx] *= 1e-3;
            }
            h += gu * gu.transpose() + gv * gv.transpose();
            inlier_count += 1;
        }
    }

    let sigma_px_sq = cand.score.train_rms_px.clamp(0.75, 12.0).powi(2);
    if inlier_count >= 6 {
        let mut reg = h;
        for d in 0..6 {
            reg[(d, d)] += 1e-6;
        }
        if let Some(inv_scaled) = reg.try_inverse() {
            let mut cov = inv_scaled * sigma_px_sq;
            for r in 0..6 {
                for c in 0..6 {
                    let scale = if r >= 3 { 1e-3 } else { 1.0 } * if c >= 3 { 1e-3 } else { 1.0 };
                    cov[(r, c)] *= scale;
                }
            }
            let sym = (cov + cov.transpose()) * 0.5;
            if sym.iter().all(|v| v.is_finite()) && sym.symmetric_eigen().eigenvalues.min() > 1e-12
            {
                return sym;
            }
        }
    }

    let mut fallback = Matrix6::zeros();
    fallback[(0, 0)] = 9.0;
    fallback[(1, 1)] = 9.0;
    fallback[(2, 2)] = 25.0;
    fallback[(3, 3)] = 0.0025;
    fallback[(4, 4)] = 0.0025;
    fallback[(5, 5)] = 0.0025;
    fallback
}

/// Compute the root-mean-square reprojection error on held-out geometric inliers.
#[must_use]
pub fn compute_holdout_rmse_px(cand: &PoseCandidate) -> f64 {
    let held = spatial_holdout(&cand.pixels, cand.size);
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for (i, &is_held) in held.iter().enumerate() {
        let is_inlier = cand.inliers.get(i).copied().unwrap_or(false);
        if !is_held || !is_inlier || i >= cand.points.len() || i >= cand.pixels.len() {
            continue;
        }
        if let Some(err) = reproj_error(
            cand.points[i],
            cand.pixels[i],
            cand.eye,
            &cand.rotation,
            &cand.k,
        ) {
            sum_sq += err * err;
            count += 1;
        }
    }

    if count > 0 {
        (sum_sq / count as f64).sqrt()
    } else {
        cand.score.train_rms_px
    }
}

/// Compute the fraction of geometric inliers located in front of the camera.
#[must_use]
pub fn compute_inlier_positive_depth_fraction(cand: &PoseCandidate) -> f64 {
    if cand.inliers.is_empty() {
        return cand.score.positive_depth_fraction;
    }
    let eye_v = Vector3::from(cand.eye);
    let mut inlier_total = 0usize;
    let mut pos_depth = 0usize;
    for (i, &is_inlier) in cand.inliers.iter().enumerate() {
        if !is_inlier || i >= cand.points.len() {
            continue;
        }
        inlier_total += 1;
        let p_cam = cand.rotation * (Vector3::from(cand.points[i]) - eye_v);
        if p_cam.z > 0.1 {
            pos_depth += 1;
        }
    }
    if inlier_total > 0 {
        (pos_depth as f64) / (inlier_total as f64)
    } else {
        cand.score.positive_depth_fraction
    }
}

/// Check intermediate seed sanity from pose finiteness, terrain clearance, and inlier count.
#[must_use]
pub fn is_sane_seed_candidate<S: CorridorTerrain + ?Sized>(
    cand: &PoseCandidate,
    surface: &S,
    min_agl_m: f64,
) -> bool {
    if !cand.eye.iter().all(|v| v.is_finite()) || !cand.rotation.iter().all(|v| v.is_finite()) {
        return false;
    }
    if cand.score.inliers < 6 {
        return false;
    }
    let geodetic = surface.geodetic(cand.eye);
    if !geodetic.iter().all(|v| v.is_finite()) {
        return false;
    }
    let dem_msl_m = surface.sample_dem(geodetic[0], geodetic[1]).unwrap_or(0.0);
    (geodetic[2] - dem_msl_m) >= min_agl_m
}

/// Verify that a candidate pose satisfies all physical plausibility requirements.
///
/// # Errors
/// Returns [`PlausibilityError`] if the candidate violates physical plausibility constraints.
pub fn verify_candidate_plausibility<S: CorridorTerrain + ?Sized>(
    cand: &PoseCandidate,
    surface: &S,
    config: &PhysicalPlausibilityConfig,
) -> Result<(), PlausibilityError> {
    if !cand.eye.iter().all(|v| v.is_finite()) || !cand.rotation.iter().all(|v| v.is_finite()) {
        return Err(PlausibilityError::NonFinitePose);
    }

    let geodetic = surface.geodetic(cand.eye);
    let (lat, lon, eye_msl_m) = (geodetic[0], geodetic[1], geodetic[2]);
    let dem_msl_m = surface.sample_dem(lat, lon).unwrap_or(0.0);

    let inlier_pos_depth = compute_inlier_positive_depth_fraction(cand);
    let min_fisher_eigenvalue = compute_fisher_min_eigenvalue(cand);
    let reprojection_rmse_px = compute_holdout_rmse_px(cand);
    let res_scale = (f64::from(cand.size[0].max(cand.size[1])) / 640.0).max(1.0);
    let mut scaled_cfg = config.clone();
    scaled_cfg.max_reprojection_rmse_px *= res_scale;

    verify_physical_plausibility(
        eye_msl_m,
        dem_msl_m,
        inlier_pos_depth,
        min_fisher_eigenvalue,
        reprojection_rmse_px,
        &scaled_cfg,
    )
}

/// Check candidate position against a prior covariance bound when the prior is trusted.
#[must_use]
pub fn verify_prior_bounding(
    cand_eye: [f64; 3],
    prior: Option<&SearchPrior>,
    trust_prior: bool,
    sigma_scale: f64,
) -> bool {
    if !trust_prior {
        return true;
    }
    let Some(prior) = prior else {
        return true;
    };

    let diff = Vector3::new(
        cand_eye[0] - prior.center.position.x,
        cand_eye[1] - prior.center.position.y,
        cand_eye[2] - prior.center.position.z,
    );

    if let Some(inv_cov) = prior.position_covariance_m2.try_inverse() {
        let mahalanobis_sq = (diff.transpose() * inv_cov * diff)[(0, 0)];
        mahalanobis_sq <= sigma_scale * sigma_scale
    } else {
        diff.norm() <= sigma_scale * 100.0
    }
}

/// Check physical plausibility, optional trusted prior bounding, and spatial support.
#[must_use]
pub fn is_plausible_candidate<S: CorridorTerrain + ?Sized>(
    cand: &PoseCandidate,
    surface: &S,
    prior: Option<&SearchPrior>,
    trust_prior: bool,
    config: &PhysicalPlausibilityConfig,
) -> bool {
    if verify_candidate_plausibility(cand, surface, config).is_err() {
        return false;
    }
    if !verify_prior_bounding(cand.eye, prior, trust_prior, config.prior_sigma_scale) {
        return false;
    }
    cand.score.image_support_fraction >= 0.025
}

#[cfg(test)]
mod tests;
