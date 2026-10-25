//! Fit physical ellipsoid points and retain the source image support.
use super::{FitSupport, PoseVerifier, calibrated, support};
use crate::{
    CameraModel, CameraPose, Estimate, Frame, LocalFrame, PixelMatch, PosePrior, ReferenceView,
    VisualError, pose_solver::Motion,
};
use std::collections::BTreeMap;

/// A spherical reference whose source heights are WGS84 ellipsoid heights.
///
/// The caller must establish the height datum of the map release. This type does
/// not infer a datum from pixels or convert orthometric heights. It retains the
/// source reference and its map identity. Query and reference pixels do not change.
pub struct EllipsoidHeightReference<'a> {
    reference: &'a ReferenceView,
    camera: &'a CameraModel,
    target: LocalFrame,
}
impl<'a> EllipsoidHeightReference<'a> {
    /// Assert that the spherical reference uses ellipsoid heights.
    ///
    /// # Errors
    /// Rejects a non-spherical frame or invalid reference dimensions and calibration.
    pub fn from_ellipsoid_heights(
        reference: &'a ReferenceView,
        camera: &'a CameraModel,
    ) -> Result<Self, VisualError> {
        reference.validate_camera(camera)?;
        if !matches!(reference.frame, LocalFrame::SphericalEnu { .. }) {
            return Err(VisualError::Invalid {
                field: "ellipsoid-height spherical reference",
            });
        }
        let [lat, lon] = reference.frame.anchor();
        Ok(Self {
            reference,
            camera,
            target: LocalFrame::wgs84_enu(lat, lon)?,
        })
    }
    /// WGS84 frame required for the navigation prior and returned estimate.
    pub fn frame(&self) -> LocalFrame {
        self.target
    }

    /// Convert a spherical pose at the same geographic position to WGS84.
    ///
    /// This converts the pose only. A radius or covariance needs its own transport.
    /// The shared anchor has the same physical east, north and up axes in both models.
    ///
    /// # Errors
    /// Rejects invalid poses and positions with an unusable WGS84 coordinate conversion.
    pub fn convert_pose(&self, source: CameraPose) -> Result<CameraPose, VisualError> {
        source.validate()?;
        Ok(CameraPose {
            position: self
                .target
                .local(self.reference.frame.geodetic(source.position))?,
            orientation: source.orientation,
        })
    }
}
impl PoseVerifier {
    /// Fit WGS84 points reconstructed from an ellipsoid-height spherical reference.
    ///
    /// The prior is in [`EllipsoidHeightReference::frame`]. The returned pose and
    /// geometry covariance use that WGS84 frame. Spatial support uses the original
    /// source pixels. Residual, conditioning and admission thresholds do not change.
    /// The source map identity is retained. Map and calibration errors remain external.
    ///
    /// # Errors
    /// Rejects invalid observations, insufficient support and poses outside the prior.
    pub fn verify_wgs84_reference(
        &self,
        frame: &Frame,
        source: &EllipsoidHeightReference<'_>,
        prior: &PosePrior,
        matches: &[PixelMatch],
        matcher_identity: &str,
    ) -> Result<Estimate, VisualError> {
        frame.camera.validate()?;
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height)
            || matches.len() > 16384
            || matcher_identity.is_empty()
        {
            return Err(VisualError::Invalid {
                field: "WGS84 reference observation",
            });
        }
        let mut pixels = BTreeMap::new();
        let points = calibrated::observations(frame, source.reference, source.camera, matches)
            .into_iter()
            .map(|(mut point, pixel)| {
                pixels.insert((point.pixel.x.to_bits(), point.pixel.y.to_bits()), pixel);
                point.world = source
                    .target
                    .local(source.reference.frame.geodetic(point.world))?;
                Ok(point)
            })
            .collect::<Result<Vec<_>, VisualError>>()?;
        let (pose, inliers, depth_matches) = self.fit_points(
            frame,
            source.convert_pose(source.reference.pose)?,
            prior,
            points,
            Motion::Free,
        )?;
        let (points, cells) = support::separated_pixels(
            [source.camera.width, source.camera.height],
            |point| {
                pixels
                    .get(&(point.pixel.x.to_bits(), point.pixel.y.to_bits()))
                    .copied()
            },
            &inliers,
            self.config.support_separation_px,
        );
        let support = FitSupport::from_separated(&frame.camera, points, inliers.len(), cells);
        let (quality, geometry_covariance) =
            self.assess_support(frame, &pose, &support, depth_matches)?;
        Ok(Estimate {
            stamp: frame.stamp,
            observation_sha256: frame.evidence_sha256(),
            map: source.reference.map.clone(),
            frame: source.target,
            pose,
            quality,
            geometry_covariance,
            backend: matcher_identity.into(),
        })
    }
}
