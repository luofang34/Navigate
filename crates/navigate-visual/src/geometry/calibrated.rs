//! Preserve each image's calibration when matching a reference from another camera.
use super::{FitSupport, PoseVerifier};
use crate::pose_solver::{Correspondence, Motion};
use crate::{CameraModel, Estimate, Frame, PixelMatch, PosePrior, ReferenceView, VisualError};
use nalgebra::Vector2;
use std::collections::BTreeSet;

impl PoseVerifier {
    /// Fit a query against a reference with its own image size and intrinsics.
    ///
    /// Reference depth uses the reference camera's optical axis. Query residuals
    /// use the query camera. Both images keep their original pixel coordinates.
    /// This permits cached references without a query-specific image resampling.
    /// An estimated photo anchor remains conditional evidence. The host must
    /// retain its source lineage and map uncertainty when using this result.
    ///
    /// # Errors
    /// Rejects invalid calibration, dimensions, more than 16,384 matches, weak
    /// support, missing depth, and poses outside the supplied prior. It uses the
    /// same spatial support, residual, and conditioning limits as [`Self::verify`].
    pub fn verify_calibrated_reference(
        &self,
        frame: &Frame,
        reference: &ReferenceView,
        reference_camera: &CameraModel,
        prior: &PosePrior,
        matches: &[PixelMatch],
        matcher_identity: &str,
    ) -> Result<Estimate, VisualError> {
        frame.camera.validate()?;
        reference.validate_camera(reference_camera)?;
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height)
            || matches.len() > 16384
            || matcher_identity.is_empty()
        {
            return Err(VisualError::Invalid {
                field: "calibrated reference observation",
            });
        }
        let points = correspondences(frame, reference, reference_camera, matches);
        let (pose, inliers, depth_matches) =
            self.fit_points(frame, reference.pose, prior, points, Motion::Free)?;
        let support = FitSupport::calibrated(
            &frame.camera,
            reference_camera,
            &reference.pose,
            &inliers,
            self.config.support_separation_px,
        );
        let (quality, geometry_covariance) =
            self.assess_support(frame, &pose, &support, depth_matches)?;
        Ok(Estimate {
            stamp: frame.stamp,
            observation_sha256: frame.evidence_sha256(),
            map: reference.map.clone(),
            frame: reference.frame,
            pose,
            quality,
            geometry_covariance,
            backend: matcher_identity.into(),
        })
    }
}

fn correspondences(
    frame: &Frame,
    reference: &ReferenceView,
    camera: &CameraModel,
    matches: &[PixelMatch],
) -> Vec<Correspondence> {
    let mut reference_pixels = BTreeSet::new();
    let mut query_pixels = BTreeSet::new();
    matches
        .iter()
        .filter_map(|pair| {
            if !inside(pair.reference, camera) || !inside(pair.query, &frame.camera) {
                return None;
            }
            let x = pair.reference.x.round() as usize;
            let y = pair.reference.y.round() as usize;
            let depth = f64::from(reference.depth_m[y * camera.width as usize + x]);
            let query_key = (pair.query.x.round() as u32, pair.query.y.round() as u32);
            if !depth.is_finite()
                || depth <= 0.0
                || reference_pixels.contains(&(x, y))
                || query_pixels.contains(&query_key)
            {
                return None;
            }
            reference_pixels.insert((x, y));
            query_pixels.insert(query_key);
            Some(Correspondence {
                world: camera.unproject(&reference.pose, pair.reference, depth),
                pixel: pair.query,
            })
        })
        .collect()
}
fn inside(pixel: Vector2<f64>, camera: &CameraModel) -> bool {
    pixel.iter().all(|v| v.is_finite())
        && pixel.x >= 0.0
        && pixel.y >= 0.0
        && pixel.x < f64::from(camera.width - 1)
        && pixel.y < f64::from(camera.height - 1)
}
#[cfg(test)]
mod tests;
