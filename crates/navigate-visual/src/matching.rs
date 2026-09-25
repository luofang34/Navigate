//! Image correspondences with a replaceable compute backend.

use crate::VisualError;
use image::GrayImage;
use nalgebra::Vector2;

pub(crate) mod pyramid;
mod tracking;

/// One correspondence between reference and query pixels.
#[derive(Clone, Copy, Debug)]
pub struct PixelMatch {
    /// Reference pixel centre.
    pub reference: Vector2<f64>,
    /// Query pixel centre.
    pub query: Vector2<f64>,
}

/// Backend boundary for CPU features or hardware model inference.
pub trait ImageMatcher {
    /// Stable backend and model identity for reports and derived caches.
    fn identity(&self) -> &str;
    /// Match two undistorted grayscale images with the same intrinsics.
    ///
    /// The call may wait for an accelerator. Backends must report their actual
    /// identity; they must not silently substitute a different implementation.
    /// Return pixel correspondences only. The shared pose solver owns geometry
    /// checks. Neural models and NPU runtimes belong behind this boundary.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible images or a backend failure. Empty
    /// matches are valid when no usable visual evidence exists.
    fn match_images_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError>;
}

/// CPU patch tracking for reference views close to the camera pose prior.
///
/// This backend uses image pyramids and checks tracking in both directions.
/// It requires overlapping views with similar appearance. It does not perform
/// global image retrieval or neural model inference.
#[derive(Default)]
pub struct PyramidalMatcher;

/// Backend for feature identities that persist across camera frames.
///
/// Feature selection and pixel tracking belong to the adapter. The caller owns
/// world geometry, pose validation, and the lifetime of each feature identity.
pub trait PointTracker: ImageMatcher {
    /// Select initial image locations without assigning world geometry.
    ///
    /// # Errors
    /// Returns an adapter error when feature selection fails.
    fn features_blocking(&mut self, image: &GrayImage) -> Result<Vec<Vector2<f64>>, VisualError>;
    /// Track each supplied reference location. Output order matches input order.
    ///
    /// A lost feature returns `None`. Locations must not be replaced by unrelated
    /// corners. Geometric acceptance remains separate from this operation.
    ///
    /// # Errors
    /// Returns invalid input or execution errors.
    fn track_points_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
        points: &[Vector2<f64>],
    ) -> Result<Vec<Option<Vector2<f64>>>, VisualError>;
}

impl ImageMatcher for PyramidalMatcher {
    fn identity(&self) -> &str {
        "cpu-pyramidal-lk-v1"
    }
    fn match_images_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        let points = pyramid::corners(reference);
        let found = self.track_points_blocking(reference, query, &points)?;
        Ok(points
            .into_iter()
            .zip(found)
            .filter_map(|(reference, query)| query.map(|query| PixelMatch { reference, query }))
            .collect())
    }
}
impl PointTracker for PyramidalMatcher {
    fn features_blocking(&mut self, image: &GrayImage) -> Result<Vec<Vector2<f64>>, VisualError> {
        Ok(pyramid::corners(image))
    }
    fn track_points_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
        points: &[Vector2<f64>],
    ) -> Result<Vec<Option<Vector2<f64>>>, VisualError> {
        validate_points(reference, query, points)?;
        let source = pyramid::build(reference);
        let target = pyramid::build(query);
        Ok(points
            .iter()
            .map(|&point| {
                let found = tracking::track(&source, &target, point, point)?;
                let back = tracking::track(&target, &source, found, point)?;
                ((back - point).norm() <= 1.0).then_some(found)
            })
            .collect())
    }
}

pub(crate) fn validate_points(
    reference: &GrayImage,
    query: &GrayImage,
    points: &[Vector2<f64>],
) -> Result<(), VisualError> {
    if reference.dimensions() != query.dimensions() {
        return Err(VisualError::Dimensions {
            query: query.dimensions(),
            reference: reference.dimensions(),
        });
    }
    if points.len() > 16384
        || points.iter().any(|p| {
            !p.iter().all(|v| v.is_finite())
                || p.x < 0.0
                || p.y < 0.0
                || p.x >= f64::from(reference.width())
                || p.y >= f64::from(reference.height())
        })
    {
        return Err(VisualError::Invalid {
            field: "tracked feature locations",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
