//! Solve map matches without a query-specific perspective reference render.
use super::{FitSupport, PoseVerifier, support};
use crate::pose_solver::{Correspondence, Motion};
use crate::{
    Frame, OrthographicEstimate, OrthographicReference, PixelMatch, PosePrior, VisualError,
};
use std::collections::BTreeSet;

impl PoseVerifier {
    /// Fit camera pose from matches to an orthographic surface raster.
    ///
    /// The prior initializes fitting and supplies admission bounds. No reference
    /// camera or query ground truth enters this path. Inlier, spatial support,
    /// conditioning, pixel noise, and prior checks match perspective verification.
    ///
    /// # Errors
    /// Rejects invalid inputs, over 16,384 matches, weak geometry, missing heights,
    /// and poses outside the prior.
    pub fn verify_orthographic(
        &self,
        frame: &Frame,
        reference: &OrthographicReference,
        prior: &PosePrior,
        matches: &[PixelMatch],
        matcher_identity: &str,
    ) -> Result<OrthographicEstimate, VisualError> {
        frame.camera.validate()?;
        reference.validate()?;
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height)
            || matches.len() > 16384
            || matcher_identity.is_empty()
        {
            return Err(VisualError::Invalid {
                field: "orthographic observation",
            });
        }
        let points = correspondences(frame, reference, matches);
        let (pose, inliers, depth_matches) =
            self.fit_points(frame, prior.pose, prior, points, Motion::Free)?;
        let (points, reference_cells) = support::separated_projected(
            [reference.image.width(), reference.image.height()],
            |world| reference.geometry.project(world),
            &inliers,
            self.config.support_separation_px,
        );
        let support =
            FitSupport::from_separated(&frame.camera, points, inliers.len(), reference_cells);
        let (quality, geometry_covariance) =
            self.assess_support(frame, &pose, &support, depth_matches)?;
        Ok(OrthographicEstimate {
            pose,
            quality,
            geometry_covariance,
            observation_sha256: frame.evidence_sha256(),
            map: reference.map.clone(),
            backend: matcher_identity.into(),
        })
    }
}

pub(super) fn correspondences(
    frame: &Frame,
    reference: &OrthographicReference,
    matches: &[PixelMatch],
) -> Vec<Correspondence> {
    let mut reference_pixels = BTreeSet::new();
    let mut query_pixels = BTreeSet::new();
    matches
        .iter()
        .filter_map(|pair| {
            let q = pair.query;
            if !q.iter().all(|v| v.is_finite())
                || q.x < 0.0
                || q.y < 0.0
                || q.x >= f64::from(frame.camera.width - 1)
                || q.y >= f64::from(frame.camera.height - 1)
            {
                return None;
            }
            let world = reference.world_at(pair.reference)?;
            let reference_key = (
                pair.reference.x.round() as u32,
                pair.reference.y.round() as u32,
            );
            let query_key = (q.x.round() as u32, q.y.round() as u32);
            if reference_pixels.contains(&reference_key) || query_pixels.contains(&query_key) {
                return None;
            }
            reference_pixels.insert(reference_key);
            query_pixels.insert(query_key);
            Some(Correspondence { world, pixel: q })
        })
        .collect()
}
