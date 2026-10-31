//! Physical plausibility validation, multi-tier search mapping, and spatial holdout.

/// Physical plausibility error reasons.
#[derive(Debug, Clone, PartialEq)]
pub enum PlausibilityError {
    /// Clearance above terrain is less than 5.0 meters.
    SubSurfaceClearance {
        /// Actual above-ground clearance in meters.
        agl_m: f64,
    },
    /// Positive cheirality fraction is below 0.98.
    LowCheiralityFraction {
        /// Observed positive depth fraction.
        fraction: f64,
    },
    /// Fisher information matrix has minimal eigenvalue below 1e-6.
    IllConditionedFisher {
        /// Minimum observed eigenvalue.
        min_eigenvalue: f64,
    },
    /// Reprojection root-mean-square error exceeds 4.0 pixels.
    ExcessiveReprojectionError {
        /// Observed RMSE in pixels.
        rmse_px: f64,
    },
    /// Candidate pose contains NaN or infinite coordinates.
    NonFinitePose,
}

/// Physical plausibility configuration parameters.
#[derive(Debug, Clone)]
pub struct PhysicalPlausibilityConfig {
    /// Minimum clearance above DEM surface in meters (no upper ceiling).
    pub min_agl_m: f64,
    /// Minimum fraction of correspondences in front of camera.
    pub min_positive_depth_fraction: f64,
    /// Minimum acceptable Fisher information matrix eigenvalue.
    pub min_fisher_eigenvalue: f64,
    /// Maximum allowable holdout reprojection error in pixels.
    pub max_reprojection_rmse_px: f64,
}

impl Default for PhysicalPlausibilityConfig {
    fn default() -> Self {
        Self {
            min_agl_m: 5.0,
            min_positive_depth_fraction: 0.98,
            min_fisher_eigenvalue: 1e-6,
            max_reprojection_rmse_px: 4.0,
        }
    }
}

/// Verifies physical plausibility decoupled from prior bounds.
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

/// Search tier for candidate pose acquisition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchTier {
    /// Local perturbation around predicted pose.
    Local,
    /// Elliptical search grid across region prior.
    Region,
    /// Global retrieval via DINOv2 place recognition.
    Global,
}

/// Outcome of search tier evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchTierOutcome {
    /// Solved at local tier.
    LocalSolved([f64; 3]),
    /// Solved at region tier.
    RegionSolved([f64; 3]),
    /// Global fallback needed with masked ground crop.
    GlobalFallbackNeeded {
        /// Center coordinate of rough prior if available.
        rough_prior: Option<[f64; 3]>,
    },
}

/// Maps prior uncertainty radius into the appropriate search tier.
pub fn resolve_search_tier(
    prior_radius_m: Option<f64>,
    local_limit_m: f64,
    region_limit_m: f64,
) -> SearchTier {
    match prior_radius_m {
        None => SearchTier::Global,
        Some(r) if r <= local_limit_m => SearchTier::Local,
        Some(r) if r <= region_limit_m => SearchTier::Region,
        Some(_) => SearchTier::Global,
    }
}

/// Deterministic 25% spatial holdout partitioner across ground sub-cells.
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
    pub fn partition(
        &self,
        points: &[[f64; 2]],
        image_size: [u32; 2],
        mask: Option<&[bool]>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut fit_indices = Vec::new();
        let mut holdout_indices = Vec::new();
        let w = image_size[0] as f64;
        let h = image_size[1] as f64;
        let cell_w = w / self.subdivisions[0] as f64;
        let cell_h = h / self.subdivisions[1] as f64;

        // Group points by sub-cell
        let mut cell_buckets = vec![Vec::new(); self.subdivisions[0] * self.subdivisions[1]];
        for (i, pt) in points.iter().enumerate() {
            if mask.is_some_and(|m| i < m.len() && !m[i]) {
                continue; // Skip masked (sky/airframe) points
            }
            let col = ((pt[0] / cell_w).floor() as usize).min(self.subdivisions[0] - 1);
            let row = ((pt[1] / cell_h).floor() as usize).min(self.subdivisions[1] - 1);
            let cell_idx = row * self.subdivisions[0] + col;
            cell_buckets[cell_idx].push(i);
        }

        // Take holdout fraction deterministically from each cell
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
