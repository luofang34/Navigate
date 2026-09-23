//! GPU patch alignment with CPU image preparation and geometric validation.

mod dispatch;
mod runtime;

use crate::{ImageMatcher, PixelMatch, VisualError, matching::pyramid};
use image::GrayImage;
use nalgebra::Vector2;

/// Optional compute-shader implementation of pyramidal patch matching.
///
/// Enable the `gpu` Cargo feature. Image pyramids and corners are prepared on
/// the CPU. Coarse search, iterative alignment, and reverse tracking execute
/// on the GPU. The shared Rust pose solver validates the returned matches.
///
/// The backend uses wgpu compute shaders. It can select Metal, Vulkan, or DX12
/// where available. It is not an ANE, CUDA, or NPU inference implementation.
/// A neural matcher requires a model and its own [`ImageMatcher`] backend.
/// No CPU fallback occurs when this backend is selected.
pub struct GpuPyramidalMatcher {
    compute: runtime::Compute,
    identity: String,
}

impl GpuPyramidalMatcher {
    /// Create a hardware compute device and compile the matching kernel.
    ///
    /// Reuse the matcher across frames to retain the device and pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`VisualError::Backend`] if no hardware adapter is available,
    /// device creation fails, or the shader cannot compile.
    pub async fn new() -> Result<Self, VisualError> {
        let (compute, adapter) = runtime::Compute::new().await.map_err(runtime::failure)?;
        Ok(Self {
            compute,
            identity: format!("wgpu-pyramidal-lk-v1/{adapter}"),
        })
    }
}

impl ImageMatcher for GpuPyramidalMatcher {
    fn identity(&self) -> &str {
        &self.identity
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
        if reference.width() > 4096 || reference.height() > 4096 {
            return Err(VisualError::Invalid {
                field: "GPU image dimensions exceed 4096",
            });
        }
        let points = pyramid::corners(reference);
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let source = pyramid::build(reference);
        let target = pyramid::build(query);
        let aligned = self
            .compute
            .align_blocking(&source, &target, &points)
            .map_err(runtime::failure)?;
        Ok(points
            .into_iter()
            .zip(aligned)
            .filter_map(|(reference, q)| {
                (q[2] > 0.5 && q[0].is_finite() && q[1].is_finite()).then_some(PixelMatch {
                    reference,
                    query: Vector2::new(f64::from(q[0]), f64::from(q[1])),
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests;
