//! GPU patch alignment with CPU image preparation and geometric validation.

mod dispatch;
mod profile;
mod runtime;
mod upload;

use crate::{ImageMatcher, PixelMatch, PointTracker, VisualError, matching::pyramid};
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
    previous_pyramid: Option<Vec<GrayImage>>,
    previous_gpu: Option<wgpu::Buffer>,
    reuse_gpu: bool,
}

impl GpuPyramidalMatcher {
    /// Create a hardware compute device and compile the matching kernel.
    ///
    /// Reuse the matcher across frames to retain the device and pipeline.
    /// The last query pyramid is retained for an exactly equal next reference.
    /// This uses fewer than twice the query image bytes between calls.
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
            previous_pyramid: None,
            previous_gpu: None,
            reuse_gpu: false,
        })
    }

    /// Enable exact-content reuse of the last query GPU image. Disabled by default.
    ///
    /// One packed f32 pyramid remains resident between tracking calls. Refinement,
    /// empty tracking, invalid input and failed alignment discard that buffer.
    /// This setting does not change matching or geometric acceptance.
    pub fn set_gpu_image_reuse(&mut self, enabled: bool) {
        self.previous_gpu = None;
        self.reuse_gpu = enabled;
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
        let points = pyramid::corners(reference);
        let found = self.track_points_blocking(reference, query, &points)?;
        Ok(points
            .into_iter()
            .zip(found)
            .filter_map(|(reference, query)| query.map(|query| PixelMatch { reference, query }))
            .collect())
    }
}
impl PointTracker for GpuPyramidalMatcher {
    fn refine_points_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
        points: &[Vector2<f64>],
        initial: &[Vector2<f64>],
    ) -> Result<Vec<Option<Vector2<f64>>>, VisualError> {
        self.previous_gpu = None;
        crate::matching::validate_refinement(reference, query, points, initial)?;
        if reference.width() > 4096 || reference.height() > 4096 {
            return Err(VisualError::Invalid {
                field: "GPU image dimensions exceed 4096",
            });
        }
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let mut profile = profile::Profile::new();
        let aligned = self
            .compute
            .align_blocking(
                std::slice::from_ref(reference),
                std::slice::from_ref(query),
                points,
                Some(initial),
                None,
                &mut profile,
            )
            .map_err(runtime::failure)?;
        profile.finish("refine");
        Ok(locations(aligned.points))
    }
    fn features_blocking(&mut self, image: &GrayImage) -> Result<Vec<Vector2<f64>>, VisualError> {
        Ok(pyramid::corners(image))
    }
    fn track_points_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
        points: &[Vector2<f64>],
    ) -> Result<Vec<Option<Vector2<f64>>>, VisualError> {
        let previous_gpu = self.previous_gpu.take();
        crate::matching::validate_points(reference, query, points)?;
        if reference.width() > 4096 || reference.height() > 4096 {
            return Err(VisualError::Invalid {
                field: "GPU image dimensions exceed 4096",
            });
        }
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let mut profile = profile::Profile::new();
        let total = profile.timer();
        let (source, hit) =
            reference_pyramid_cached(&mut self.previous_pyramid, reference, &mut profile);
        let resident = previous_gpu.filter(|_| hit);
        let preparation = profile.timer();
        let target = pyramid::build(query);
        profile.record(preparation, "query_pyramid", 0, 0);
        let aligned = self
            .compute
            .align_blocking(&source, &target, points, None, resident, &mut profile)
            .map_err(runtime::failure)?;
        self.previous_pyramid = Some(target);
        let retention = profile.timer();
        self.previous_gpu = self.reuse_gpu.then_some(aligned.query);
        profile.record(
            retention,
            "query_image_resident",
            self.previous_gpu
                .as_ref()
                .map_or(0, |buffer| buffer.size() as usize),
            0,
        );
        let result = locations(aligned.points);
        profile.record(total, "track_total", 0, 0);
        profile.finish("track");
        Ok(result)
    }
}

fn locations(aligned: Vec<[f32; 4]>) -> Vec<Option<Vector2<f64>>> {
    aligned
        .into_iter()
        .map(|q| {
            (q[2] > 0.5 && q[0].is_finite() && q[1].is_finite())
                .then_some(Vector2::new(f64::from(q[0]), f64::from(q[1])))
        })
        .collect()
}

fn reference_pyramid_cached(
    previous: &mut Option<Vec<GrayImage>>,
    reference: &GrayImage,
    profile: &mut profile::Profile,
) -> (Vec<GrayImage>, bool) {
    // Content and dimensions both matter: callers can reuse or mutate image storage.
    let timer = profile.timer();
    let cached = previous
        .take()
        .filter(|levels| levels.first() == Some(reference));
    let hit = cached.is_some();
    let stage = if hit {
        "reference_pyramid_hit"
    } else {
        "reference_pyramid_miss"
    };
    let result = cached.unwrap_or_else(|| pyramid::build(reference));
    profile.record(timer, stage, 0, 0);
    (result, hit)
}

#[cfg(test)]
mod cache_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod sequence_tests;

#[cfg(test)]
mod residency_tests;
