//! Decoupled physical plausibility validation and prior bounding for corridor anchors.

use crate::{
    pose::{PoseCandidate, point_jacobian, reproj_error, spatial_holdout},
    surface::Surface,
};
use nalgebra::{Matrix6, Vector3};
use navigate_visual::SearchPrior;

/// Reasons that cause physical plausibility validation to reject a candidate pose.
#[derive(Debug, Clone, PartialEq)]
pub enum PlausibilityError {
    /// Clearance above terrain is less than the minimum required clearance.
    SubSurfaceClearance {
        /// Actual above-ground clearance in meters.
        agl_m: f64,
    },
    /// Fraction of correspondences with positive depth is below the threshold.
    LowCheiralityFraction {
        /// Observed positive depth fraction.
        fraction: f64,
    },
    /// Minimal eigenvalue of the 6-DoF Fisher information matrix is below the threshold.
    IllConditionedFisher {
        /// Minimum observed eigenvalue.
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
    /// Sigma multiplier for prior covariance bounding when trusted.
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

/// Verify scalar physical plausibility criteria decoupled from any prior.
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

/// Calculate the minimum eigenvalue of the 6-DoF Fisher information matrix.
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

/// Calculate the root-mean-square reprojection error on held-out correspondences.
#[must_use]
pub fn compute_holdout_rmse_px(cand: &PoseCandidate) -> f64 {
    let held = spatial_holdout(&cand.pixels, cand.size);
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for (i, &is_held) in held.iter().enumerate() {
        if !is_held || i >= cand.points.len() || i >= cand.pixels.len() {
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

/// Verify that a candidate pose satisfies all physical plausibility requirements.
///
/// # Errors
/// Returns [`PlausibilityError`] if the candidate violates physical plausibility constraints.
pub fn verify_candidate_plausibility(
    cand: &PoseCandidate,
    surface: &Surface,
    config: &PhysicalPlausibilityConfig,
) -> Result<(), PlausibilityError> {
    if !cand.eye.iter().all(|v| v.is_finite()) || !cand.rotation.iter().all(|v| v.is_finite()) {
        return Err(PlausibilityError::NonFinitePose);
    }

    let geodetic = surface.geodetic(cand.eye);
    let (lat, lon, eye_msl_m) = (geodetic[0], geodetic[1], geodetic[2]);
    let dem_msl_m = surface.sample_dem(lat, lon).unwrap_or(0.0);

    let min_fisher_eigenvalue = compute_fisher_min_eigenvalue(cand);
    let reprojection_rmse_px = compute_holdout_rmse_px(cand);

    verify_physical_plausibility(
        eye_msl_m,
        dem_msl_m,
        cand.score.positive_depth_fraction,
        min_fisher_eigenvalue,
        reprojection_rmse_px,
        config,
    )
}

/// Check candidate position against a prior covariance bounding only when trusted.
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

/// Combined gate: checks physical plausibility and trusted prior bounding.
#[must_use]
pub fn is_plausible_candidate(
    cand: &PoseCandidate,
    surface: &Surface,
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
mod tests {
    use super::*;

    #[test]
    fn test_verify_physical_plausibility_agl_bounds() {
        let cfg = PhysicalPlausibilityConfig::default();
        assert!(verify_physical_plausibility(505.0, 500.0, 0.99, 1e-4, 1.0, &cfg).is_ok());
        assert!(matches!(
            verify_physical_plausibility(504.0, 500.0, 0.99, 1e-4, 1.0, &cfg),
            Err(PlausibilityError::SubSurfaceClearance { .. })
        ));
    }

    #[test]
    fn test_stratospheric_and_orbital_clearance() {
        let cfg = PhysicalPlausibilityConfig::default();
        assert!(verify_physical_plausibility(25_000.0, 100.0, 0.99, 1e-4, 1.0, &cfg).is_ok());
        assert!(verify_physical_plausibility(400_000.0, 0.0, 0.99, 1e-4, 1.0, &cfg).is_ok());
    }

    #[test]
    fn test_cheirality_and_fisher_rejections() {
        let cfg = PhysicalPlausibilityConfig::default();
        assert!(matches!(
            verify_physical_plausibility(600.0, 500.0, 0.95, 1e-4, 1.0, &cfg),
            Err(PlausibilityError::LowCheiralityFraction { .. })
        ));
        assert!(matches!(
            verify_physical_plausibility(600.0, 500.0, 0.99, 1e-7, 1.0, &cfg),
            Err(PlausibilityError::IllConditionedFisher { .. })
        ));
        assert!(matches!(
            verify_physical_plausibility(600.0, 500.0, 0.99, 1e-4, 5.0, &cfg),
            Err(PlausibilityError::ExcessiveReprojectionError { .. })
        ));
    }

    #[test]
    fn test_nan_and_inf_metrics_rejection() {
        let cfg = PhysicalPlausibilityConfig::default();
        assert_eq!(
            verify_physical_plausibility(500.0, 0.0, f64::NAN, 1e-4, 1.0, &cfg),
            Err(PlausibilityError::NonFinitePose)
        );
        assert_eq!(
            verify_physical_plausibility(500.0, 0.0, 0.99, f64::NAN, 1.0, &cfg),
            Err(PlausibilityError::NonFinitePose)
        );
        assert_eq!(
            verify_physical_plausibility(500.0, 0.0, 0.99, 1e-4, f64::NAN, &cfg),
            Err(PlausibilityError::NonFinitePose)
        );
        assert_eq!(
            verify_physical_plausibility(500.0, 0.0, f64::INFINITY, 1e-4, 1.0, &cfg),
            Err(PlausibilityError::NonFinitePose)
        );
        assert_eq!(
            verify_physical_plausibility(500.0, 0.0, 0.99, f64::INFINITY, 1.0, &cfg),
            Err(PlausibilityError::NonFinitePose)
        );
        assert_eq!(
            verify_physical_plausibility(500.0, 0.0, 0.99, 1e-4, f64::INFINITY, &cfg),
            Err(PlausibilityError::NonFinitePose)
        );
    }

    #[test]
    fn test_untrusted_prior_bypasses_bounding() {
        let cand_eye = [10000.0, 0.0, 500.0];
        assert!(verify_prior_bounding(cand_eye, None, false, 3.0));
    }
}
