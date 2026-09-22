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

impl ImageMatcher for PyramidalMatcher {
    fn identity(&self) -> &str {
        "cpu-pyramidal-lk-v1"
    }

    fn match_images_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        if reference.dimensions() != query.dimensions() {
            return Err(VisualError::Dimensions {
                query: query.dimensions(),
                reference: reference.dimensions(),
            });
        }
        let source = pyramid::build(reference);
        let target = pyramid::build(query);
        let mut matches = Vec::new();
        for point in pyramid::corners(reference) {
            let Some(found) = tracking::track(&source, &target, point, point) else {
                continue;
            };
            let Some(back) = tracking::track(&target, &source, found, point) else {
                continue;
            };
            if (back - point).norm() <= 1.0 {
                matches.push(PixelMatch {
                    reference: point,
                    query: found,
                });
            }
        }
        Ok(matches)
    }
}

#[cfg(test)]
mod tests;
