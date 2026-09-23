//! Where to look for the camera, from the navigation estimate (ADR-0009).
//!
//! A [`SearchPrior`] narrows the candidate poses that the host renders and
//! matches. It never admits a pose. Admission uses a [`crate::PosePrior`]
//! that the host builds from evidence independent of this estimate, so a
//! wrong estimate cannot confirm itself.

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

use crate::{CameraPose, VisualError};

/// Thresholds that choose a search tier and bound the candidate count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    /// Sigma multiple of the horizontal error ellipse that the search covers.
    pub sigma_scale: f64,
    /// Search radius up to which one refinement from the center suffices, in metres.
    pub local_radius_m: f64,
    /// Search radius above which the estimate does not narrow the search, in metres.
    pub region_radius_m: f64,
    /// Horizontal spacing of region candidates, in metres.
    pub candidate_spacing_m: f64,
    /// Upper bound on region candidates.
    pub max_candidates: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            sigma_scale: 3.0,
            local_radius_m: 30.0,
            region_radius_m: 2_000.0,
            candidate_spacing_m: 40.0,
            max_candidates: 64,
        }
    }
}

/// How much the estimate narrows the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchTier {
    /// Refine from the projected pose only.
    Local,
    /// Render and match candidates across the error ellipse.
    Region,
    /// The estimate is too uncertain. Use global retrieval.
    Global,
}

/// The navigation estimate projected to one frame's capture time.
#[derive(Clone, Copy, Debug)]
pub struct SearchPrior {
    /// Projected camera pose in the reference frame.
    pub center: CameraPose,
    /// Position covariance along the reference frame axes (east, north, up), in metres².
    pub position_covariance_m2: Matrix3<f64>,
    /// One-sigma attitude uncertainty, in radians.
    pub attitude_sigma_rad: f64,
}

impl SearchPrior {
    /// Check that the pose is valid and the covariance is finite and positive definite.
    ///
    /// # Errors
    /// Rejects an invalid pose, covariance, or attitude uncertainty.
    pub fn validate(&self) -> Result<(), VisualError> {
        self.center.validate()?;
        let finite = self.position_covariance_m2.iter().all(|v| v.is_finite());
        let positive = self
            .position_covariance_m2
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .all(|v| *v > 0.0);
        if !finite
            || !positive
            || !(self.attitude_sigma_rad.is_finite() && self.attitude_sigma_rad >= 0.0)
        {
            return Err(VisualError::Invalid {
                field: "search prior",
            });
        }
        Ok(())
    }

    /// Horizontal search radius: the sigma multiple of the major semi-axis, in metres.
    pub fn horizontal_radius_m(&self, config: &SearchConfig) -> f64 {
        let major = self
            .horizontal_covariance()
            .symmetric_eigen()
            .eigenvalues
            .max();
        config.sigma_scale * major.max(0.0).sqrt()
    }

    /// The tier that this estimate supports.
    pub fn tier(&self, config: &SearchConfig) -> SearchTier {
        let radius = self.horizontal_radius_m(config);
        if radius <= config.local_radius_m {
            SearchTier::Local
        } else if radius <= config.region_radius_m {
            SearchTier::Region
        } else {
            SearchTier::Global
        }
    }

    /// Candidate poses, most likely first, bounded by `config.max_candidates`.
    ///
    /// A local tier gives the center only. A region tier gives the center and
    /// grid points inside the sigma ellipse, ordered by Mahalanobis distance. A
    /// global tier gives no candidates.
    pub fn candidates(&self, config: &SearchConfig) -> Vec<CameraPose> {
        match self.tier(config) {
            SearchTier::Local => vec![self.center],
            SearchTier::Global => Vec::new(),
            SearchTier::Region => self.region_candidates(config),
        }
    }

    fn horizontal_covariance(&self) -> Matrix2<f64> {
        self.position_covariance_m2
            .fixed_view::<2, 2>(0, 0)
            .into_owned()
    }

    fn region_candidates(&self, config: &SearchConfig) -> Vec<CameraPose> {
        let Some(inverse) = self.horizontal_covariance().try_inverse() else {
            return vec![self.center];
        };
        let spacing = config.candidate_spacing_m.max(f64::EPSILON);
        let radius = self.horizontal_radius_m(config);
        let steps = (radius / spacing).ceil() as i64;
        let limit = config.sigma_scale * config.sigma_scale;
        let mut points: Vec<(f64, Vector2<f64>)> = Vec::new();
        for i in -steps..=steps {
            for j in -steps..=steps {
                let offset = Vector2::new(i as f64 * spacing, j as f64 * spacing);
                let distance = (offset.transpose() * inverse * offset)[(0, 0)];
                if distance <= limit {
                    points.push((distance, offset));
                }
            }
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        points
            .into_iter()
            .take(config.max_candidates.max(1))
            .map(|(_, offset)| CameraPose {
                position: self.center.position + Vector3::new(offset.x, offset.y, 0.0),
                orientation: self.center.orientation,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
