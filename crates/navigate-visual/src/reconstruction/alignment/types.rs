//! Conditional coordinate transforms between estimated scene groups.
use crate::LocalScenePose;
use nalgebra::{UnitQuaternion, Vector3};
/// Coordinate conversion from one estimated scene to another.
/// This is not a navigation measurement or an accuracy bound.
#[derive(Clone, Copy, Debug)]
pub struct SceneTransform {
    /// Positive ratio of target units to source units.
    pub scale: f64,
    /// Source axes to target axes.
    pub rotation: UnitQuaternion<f64>,
    /// Target coordinates of the source origin.
    pub translation: Vector3<f64>,
}
impl SceneTransform {
    /// Express an estimated source point in the target coordinate frame.
    pub fn point(&self, point: Vector3<f64>) -> Vector3<f64> {
        self.rotation * point * self.scale + self.translation
    }
    /// Express an estimated source camera in the target coordinate frame.
    pub fn pose(&self, pose: LocalScenePose) -> LocalScenePose {
        LocalScenePose {
            position: self.point(pose.position),
            orientation: self.rotation * pose.orientation,
        }
    }
}
/// Numerical consistency limits for two conditional reconstruction groups.
/// These limits do not express covariance, probability, or geographic accuracy.
#[derive(Clone, Copy, Debug)]
pub struct AlignmentConfig {
    /// Minimum retained shared observations. Must be at least three.
    pub min_shared_cameras: usize,
    /// Maximum difference between aligned camera rotations, in radians.
    pub max_rotation_error_rad: f64,
    /// Maximum camera position residual divided by the median target baseline.
    pub max_position_error_ratio: f64,
    /// Minimum shared baseline relative to each complete group's camera span.
    /// This prevents near-stationary overlap from fixing an unstable scale.
    pub min_shared_baseline_ratio: f64,
}
impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            min_shared_cameras: 6,
            max_rotation_error_rad: 5.0_f64.to_radians(),
            max_position_error_ratio: 0.1,
            min_shared_baseline_ratio: 0.005,
        }
    }
}
/// A conditional alignment supported by shared image identities.
///
/// Shared camera estimates use correlated image evidence. This transform does
/// not create independent measurements, remove geographic alternatives, or
/// establish metric scale. Apply it within one reconstruction alternative only.
#[derive(Clone, Debug)]
pub struct SceneAlignment {
    /// Coordinate conversion into the target group's frame.
    pub transform: SceneTransform,
    /// Exact source digests retained in this fit.
    pub observation_sha256: Vec<String>,
    /// Shared observations that failed the consistency limits.
    pub excluded_observation_sha256: Vec<String>,
    /// Position fit residual in target scene units, not absolute accuracy.
    pub position_rms_scene_units: f64,
    /// Rotation fit residual in radians, not an attitude accuracy bound.
    pub rotation_rms_rad: f64,
    /// Median retained target-camera baseline, in target scene units.
    pub target_baseline_scene_units: f64,
}
/// Group alignment lacks valid inputs or consistent shared support.
#[derive(Debug, thiserror::Error)]
pub enum SceneAlignmentError {
    /// A consistency limit is invalid.
    #[error("invalid scene alignment configuration")]
    Config,
    /// A source or target group is invalid.
    #[error("invalid {group} scene camera {index}: {reason}")]
    Camera {
        /// Group role.
        group: &'static str,
        /// Camera index.
        index: usize,
        /// Validation failure.
        reason: &'static str,
    },
    /// Too few shared estimates remain. No transform is returned.
    #[error("scene alignment has {retained} consistent shared cameras; requires {required}")]
    Support {
        /// Retained source observations.
        retained: usize,
        /// Required count.
        required: usize,
    },
    /// A scene point or its observation support is invalid.
    #[error("invalid {group} scene point {feature_id}: {reason}")]
    Point {
        /// Group role.
        group: &'static str,
        /// Point identity within that scene.
        feature_id: u64,
        /// Validation failure.
        reason: &'static str,
    },
    /// A supplied point association is invalid or reuses point evidence.
    #[error("invalid scene point association {index}: {reason}")]
    Association {
        /// Index in the supplied proposals.
        index: usize,
        /// Validation failure.
        reason: &'static str,
    },
    /// Too few point proposals pass image and scene consistency checks.
    #[error("scene alignment has {retained} consistent points; requires {required}")]
    PointSupport {
        /// Retained distinct point pairs.
        retained: usize,
        /// Required count.
        required: usize,
    },
    /// Shared motion does not constrain a stable positive scale.
    #[error("scene alignment baseline is insufficient or inconsistent")]
    Baseline,
}
