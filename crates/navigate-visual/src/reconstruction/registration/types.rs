//! Conditional correspondence and transform results for geographic registration.
use super::super::SceneTransform;
use crate::{LocalFrame, MapRevision, VisualError};
use nalgebra::{Vector2, Vector3};
/// Limits for one bounded scene-to-reference geometric search.
/// These thresholds do not express an accuracy bound or a probability.
#[derive(Clone, Copy, Debug)]
pub struct RegistrationConfig {
    /// Maximum distance between the two observations of an associated feature.
    pub association_radius_px: f64,
    /// Maximum scene-to-reference point residual in map units.
    pub inlier_threshold_m: f64,
    /// Required distinct scene points. Must be at least six.
    pub min_inliers: usize,
    /// Required occupied cells in the query image's four-by-three grid.
    pub min_occupied_cells: usize,
    /// Deterministic three-point hypothesis budget, at most 16,384.
    pub trials: usize,
    /// Maximum retained geometric alternatives, at most 32.
    pub max_candidates: usize,
}
impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            association_radius_px: 3.0,
            inlier_threshold_m: 2.0,
            min_inliers: 20,
            min_occupied_cells: 6,
            trials: 4096,
            max_candidates: 8,
        }
    }
}
/// One estimated image association between local geometry and a reference surface.
#[derive(Clone, Debug)]
pub struct SceneMapAssociation {
    /// Stable local point identity from the supplied scene.
    pub feature_id: u64,
    /// Index in the supplied pixel matches.
    pub match_index: usize,
    /// Estimated local point in arbitrary scene units.
    pub scene_point: Vector3<f64>,
    /// Reference surface point in the declared map frame.
    pub map_point: Vector3<f64>,
    /// Query pixel from the supplied image correspondence.
    pub query_pixel: Vector2<f64>,
    /// Reference pixel with valid rendered optical depth.
    pub reference_pixel: Vector2<f64>,
    /// Distance from the query pixel to the local point observation.
    pub association_distance_px: f64,
}
/// One geometric transform. It is not an accepted geographic location.
#[derive(Clone, Debug)]
pub struct SceneRegistration {
    /// Transform from local scene units into the reference map frame.
    pub transform: SceneTransform,
    /// Indices of retained distinct associations in the proposal result.
    pub inlier_indices: Vec<usize>,
    /// Point fit error in map units, not absolute geographic accuracy.
    pub inlier_rms_m: f64,
    /// Query image cells containing retained associations.
    pub occupied_cells: usize,
}
/// Bounded scene registration alternatives with exact source and map identities.
///
/// Calibration, scene geometry, rendered depth, and geographic registration are
/// estimates with unknown errors and correlations. Repeated calls add no evidence.
/// A navigation prior and final acceptance policy must be evaluated separately.
#[derive(Clone, Debug)]
pub struct SceneRegistrationProposals {
    /// Separate geometric transforms. No poses or locations are averaged.
    pub candidates: Vec<SceneRegistration>,
    /// Processing identity of the query observation used for these links.
    pub observation_sha256: String,
    /// Versioned map data used to render the supplied reference.
    pub map: MapRevision,
    /// Frame and vertical datum of the map points.
    pub frame: LocalFrame,
    /// Estimated associations shared by the candidate transforms.
    pub associations: Vec<SceneMapAssociation>,
    /// More supported candidates were found than the retention budget permits.
    /// False does not establish uniqueness: the geometric search is bounded.
    pub candidate_budget_exhausted: bool,
}
/// A scene registration input lacks valid, traceable geometry.
#[derive(Debug, thiserror::Error)]
pub enum SceneRegistrationError {
    /// Invalid map surface or camera calibration.
    #[error("scene registration reference: {source}")]
    Reference {
        /// Original validation failure.
        #[source]
        source: VisualError,
    },
    /// Invalid search limits.
    #[error("invalid scene registration configuration")]
    Config,
    /// Invalid or repeated source evidence.
    #[error("scene registration input {index}: {reason}")]
    Input {
        /// Point or camera index.
        index: usize,
        /// Validation failure.
        reason: &'static str,
    },
    /// The requested observation is absent from the supplied scene.
    #[error("scene has no observation {observation_sha256}")]
    Observation {
        /// Requested processing identity.
        observation_sha256: String,
    },
    /// Missing surface data or weak association support.
    #[error("scene registration has {found} unique valid depth associations; requires {required}")]
    Support {
        /// Distinct associations found.
        found: usize,
        /// Required association count.
        required: usize,
    },
}
