//! Feature handles can refer to host memory or device memory.

use image::GrayImage;
use navigate_visual::{PixelMatch, VisualError};

/// Bounds and source identity for one feature handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureMetadata {
    /// Bytes retained by the feature handle, including host and device copies.
    pub resident_bytes: usize,
    /// Selected feature count.
    pub count: usize,
}

/// Resident extraction and matching stages with explicit feature reuse.
///
/// The handle need not contain host tensors. A device backend can retain its
/// buffers between calls. Exporting a map package is a separate host operation.
pub trait FeatureBackend {
    /// Immutable features owned by this backend.
    type Features;

    /// Model and execution identity for result records.
    fn identity(&self) -> &str;

    /// Extract one image. This call can wait for the compute device.
    ///
    /// # Errors
    /// Returns invalid input, model, or device errors.
    fn extract_features_blocking(
        &mut self,
        image: &GrayImage,
    ) -> Result<Self::Features, VisualError>;

    /// Check that a handle belongs to this encoder and these image pixels.
    ///
    /// # Errors
    /// Rejects incompatible feature contracts or source images.
    fn validate_features(
        &self,
        features: &Self::Features,
        image: &GrayImage,
    ) -> Result<FeatureMetadata, VisualError>;

    /// Match two resident feature handles in their source image coordinates.
    ///
    /// # Errors
    /// Returns incompatible feature contracts or execution errors.
    fn match_features_blocking(
        &mut self,
        reference: &Self::Features,
        query: &Self::Features,
    ) -> Result<Vec<PixelMatch>, VisualError>;
}
