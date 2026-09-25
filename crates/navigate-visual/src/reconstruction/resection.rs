//! Conditional camera fitting against one estimated scene.
use super::{Pose, reprojection};
use crate::{CameraModel, CameraPose, LocalScenePose, VisualError};
use nalgebra::{Vector2, Vector3};
use std::collections::BTreeSet;

/// One proposed image observation of an estimated scene point.
#[derive(Clone, Debug)]
pub struct ScenePointMatch {
    /// Feature identity within the supplied scene.
    pub feature_id: u64,
    /// Point in the scene's arbitrary coordinate frame and units.
    pub position: Vector3<f64>,
    /// Pixel in the calibrated observation.
    pub pixel: Vector2<f64>,
}

/// A conditional pose fit. This is not a geographic acceptance decision.
#[derive(Clone, Debug)]
pub struct SceneCameraFit {
    /// Host-supplied digest of the exact scene used for this fit.
    pub scene_sha256: String,
    /// Digest of the source image, calibration, and capture identity.
    pub observation_sha256: String,
    /// Estimated camera in the supplied scene coordinates.
    pub pose: LocalScenePose,
    /// Distinct features supporting the fitted pose.
    pub inlier_feature_ids: Vec<u64>,
    /// Root mean square image error for the retained features.
    pub reprojection_rms_px: f64,
}

/// Invalid inputs to conditional scene-camera fitting.
#[derive(Debug, thiserror::Error)]
pub enum SceneResectionError {
    /// Camera calibration is invalid.
    #[error("invalid scene-camera calibration: {source}")]
    Camera {
        /// Calibration error.
        #[source]
        source: VisualError,
    },
    /// The initial pose is invalid.
    #[error("invalid scene-camera initial pose: {source}")]
    Pose {
        /// Pose error.
        #[source]
        source: VisualError,
    },
    /// An evidence digest is invalid.
    #[error("invalid scene-camera {kind} digest")]
    Identity {
        /// The invalid identity field.
        kind: &'static str,
    },
    /// The correspondence array exceeds the resource bound.
    #[error("scene-camera fit has {count} matches; maximum is 65536")]
    Limit {
        /// Supplied correspondence count.
        count: usize,
    },
    /// A correspondence is invalid or duplicates evidence.
    #[error("invalid scene-camera feature {feature_id}: {reason}")]
    Match {
        /// Feature identity.
        feature_id: u64,
        /// Validation failure.
        reason: &'static str,
    },
}

/// Fit one camera to proposed 2D-to-3D links without changing the scene.
///
/// All orientations are supported. The initial pose, scene points, calibration,
/// and links remain estimates. The scene is held fixed only for this calculation.
/// Retain each scene alternative separately. A caller can supply links from any
/// matcher or tracker. The result has no covariance or independence claim.
/// Reprocessing the same input does not add evidence. The caller must check the
/// source digests and apply geometric and navigation acceptance separately.
///
/// Returns `None` when this fitter cannot find at least 20 inliers within 2.5
/// pixels. That result does not establish that the observation is unlocatable.
///
/// # Errors
/// Rejects invalid calibration, pose, identities, repeated features or pixels,
/// nonfinite points, pixels outside the image, and more than 65,536 links.
pub fn refit_scene_camera(
    camera: &CameraModel,
    scene_sha256: &str,
    observation_sha256: &str,
    initial: LocalScenePose,
    matches: &[ScenePointMatch],
) -> Result<Option<SceneCameraFit>, SceneResectionError> {
    validate(camera, scene_sha256, observation_sha256, initial, matches)?;
    let points: Vec<_> = matches
        .iter()
        .map(|p| reprojection::Point {
            world: p.position,
            pixel: p.pixel,
        })
        .collect();
    let Some((pose, indices)) = reprojection::fit(camera, &points, Pose::from_scene(initial))
    else {
        return Ok(None);
    };
    let squared_error: f64 = indices
        .iter()
        .map(|&i| reprojection::residual(camera, pose, &points[i]).powi(2))
        .sum();
    Ok(Some(SceneCameraFit {
        scene_sha256: scene_sha256.to_ascii_lowercase(),
        observation_sha256: observation_sha256.to_ascii_lowercase(),
        pose: pose.to_scene(),
        reprojection_rms_px: (squared_error / indices.len() as f64).sqrt(),
        inlier_feature_ids: indices.into_iter().map(|i| matches[i].feature_id).collect(),
    }))
}

fn validate(
    camera: &CameraModel,
    scene: &str,
    observation: &str,
    initial: LocalScenePose,
    matches: &[ScenePointMatch],
) -> Result<(), SceneResectionError> {
    camera
        .validate()
        .map_err(|source| SceneResectionError::Camera { source })?;
    CameraPose {
        position: initial.position,
        orientation: initial.orientation,
    }
    .validate()
    .map_err(|source| SceneResectionError::Pose { source })?;
    for (kind, id) in [("scene", scene), ("observation", observation)] {
        if id.len() != 64 || !id.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(SceneResectionError::Identity { kind });
        }
    }
    if matches.len() > 65536 {
        return Err(SceneResectionError::Limit {
            count: matches.len(),
        });
    }
    let mut features = BTreeSet::new();
    let mut pixels = BTreeSet::new();
    for p in matches {
        let invalid = |reason| SceneResectionError::Match {
            feature_id: p.feature_id,
            reason,
        };
        if !features.insert(p.feature_id) {
            return Err(invalid("repeated feature identity"));
        }
        if !p.position.iter().all(|v| v.is_finite())
            || !(0.0..f64::from(camera.width)).contains(&p.pixel.x)
            || !(0.0..f64::from(camera.height)).contains(&p.pixel.y)
        {
            return Err(invalid(
                "nonfinite point or pixel outside the calibrated image",
            ));
        }
        let bits = |v: f64| if v == 0.0 { 0 } else { v.to_bits() };
        if !pixels.insert((bits(p.pixel.x), bits(p.pixel.y))) {
            return Err(invalid("repeated image pixel"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
