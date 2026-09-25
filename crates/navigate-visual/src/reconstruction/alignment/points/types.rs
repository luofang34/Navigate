//! Inputs and fit evidence for alignment through shared scene points.
use super::super::{AlignmentConfig, SceneAlignment};
/// One proposed point identity across two estimated scenes.
/// Point IDs are local to each source scene, not global map identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScenePointAssociation {
    /// Point ID in the source scene.
    pub source_feature_id: u64,
    /// Point ID in the target scene.
    pub target_feature_id: u64,
}
/// Limits for point proposals from repeated pixel observations.
#[derive(Clone, Copy, Debug)]
pub struct PointAssociationConfig {
    /// Maximum pixel distance in the same processed image.
    pub max_pixel_distance: f64,
    /// Minimum distinct shared images for each proposed point pair.
    pub min_shared_observations: usize,
}
impl Default for PointAssociationConfig {
    fn default() -> Self {
        Self {
            max_pixel_distance: 0.5,
            min_shared_observations: 3,
        }
    }
}
/// Numerical limits for a conditional point-supported alignment.
/// These limits are not probability, covariance, or an accuracy bound.
#[derive(Clone, Copy, Debug)]
pub struct PointAlignmentConfig {
    /// Shared camera count and rotation limits. Camera baseline limits do not
    /// determine scale in this calculation.
    pub cameras: AlignmentConfig,
    /// Required support in the shared processed images.
    pub associations: PointAssociationConfig,
    /// Maximum distinct initial scale proposals examined.
    pub max_scale_proposals: usize,
    /// Maximum returned scale alternatives. Truncation is reported.
    pub max_candidates: usize,
    /// Minimum distinct retained point pairs.
    pub min_shared_points: usize,
    /// Maximum point residual divided by its distance from the target cameras.
    pub max_point_error_ratio: f64,
    /// Maximum camera residual divided by the median retained point distance.
    pub max_camera_position_error_ratio: f64,
    /// Minimum point extent divided by median point distance in each scene.
    pub min_point_spread_ratio: f64,
}
impl Default for PointAlignmentConfig {
    fn default() -> Self {
        Self {
            cameras: AlignmentConfig::default(),
            associations: PointAssociationConfig::default(),
            min_shared_points: 8,
            max_scale_proposals: 256,
            max_candidates: 8,
            max_point_error_ratio: 0.05,
            max_camera_position_error_ratio: 0.01,
            min_point_spread_ratio: 0.05,
        }
    }
}
/// Conditional fit supported by shared cameras and estimated points.
/// Shared images remain correlated. Reprocessing does not add evidence.
#[derive(Clone, Debug)]
pub struct ScenePointAlignment {
    /// Transform and shared-camera residuals. Its camera baseline can be zero.
    pub alignment: SceneAlignment,
    /// Point proposals retained by both image and scene consistency checks.
    pub associations: Vec<ScenePointAssociation>,
    /// Point proposals excluded by those checks.
    pub excluded_associations: Vec<ScenePointAssociation>,
    /// Point fit residual in target scene units, not geographic accuracy.
    pub point_rms_scene_units: f64,
    /// Median point distance used to normalize camera position residuals.
    pub target_point_distance_scene_units: f64,
}

/// Supported conditional scale alternatives from one pair of scene estimates.
#[derive(Clone, Debug)]
pub struct ScenePointAlignmentProposals {
    /// Distinct supported fits. No geographic alternative is accepted here.
    pub candidates: Vec<ScenePointAlignment>,
    /// A work or output limit deferred other scale proposals.
    pub candidate_budget_exhausted: bool,
}
