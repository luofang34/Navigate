//! Amortized live semantic occlusion masking, attitude sky warping, and propeller gating.

use crate::PipelineError;
use burn::tensor::Device;
use image::{GrayImage, RgbImage, imageops::colorops::grayscale};
use nalgebra::Matrix3;
pub use navigate_segment::amortized::MaskFrameTelemetry;
use navigate_segment::{
    MaskPolicy, PropellerDetector, Segformer,
    amortized::{
        SegEngine, compose_amortized_u8, compose_fallback_u8, compose_unmasked_u8,
        horizon_from_keep_layers,
    },
};
use std::{path::Path, time::Instant};

mod refresh;

use refresh::{KeyLayers, Refresher};

/// Amortized live frame masker combining `Segformer`, attitude sky warping, and [`PropellerDetector`].
pub struct AmortizedMasker {
    seg: Option<Refresher>,
    refresh_interval: usize,
    propeller: PropellerDetector,
    key: KeyLayers,
}

impl AmortizedMasker {
    /// Load an [`AmortizedMasker`] with an optional `Segformer` checkpoint directory.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when `policy_str` or `seg_model_dir` fails to load.
    pub fn new(
        seg_model_dir: Option<&Path>,
        policy_str: &str,
        margin_px: usize,
        refresh_interval: usize,
        device: &Device,
    ) -> Result<Self, PipelineError> {
        let policy = MaskPolicy::parse(policy_str)?;
        let seg = if let Some(dir) = seg_model_dir {
            let (model, cfg, preproc) = Segformer::load_pretrained(dir, None, device)?;
            Some(Refresher::Inline(Box::new(SegEngine::from_loaded(
                model,
                &cfg,
                preproc,
                policy,
                margin_px,
                device.clone(),
            ))))
        } else {
            None
        };
        Ok(Self {
            seg,
            refresh_interval: refresh_interval.max(1),
            propeller: PropellerDetector::default(),
            key: KeyLayers::default(),
        })
    }

    /// Run `Segformer` refreshes on a named background thread.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when the worker thread cannot start.
    pub fn with_background_refresh(mut self) -> Result<Self, PipelineError> {
        self.seg = self
            .seg
            .take()
            .map(Refresher::into_background)
            .transpose()?;
        Ok(self)
    }

    /// Compute the binary valid-terrain mask (`255` = keep, `0` = occluded) for one frame.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when `Segformer` inference or mask post-processing fails.
    pub fn mask_frame(
        &mut self,
        rgb: &RgbImage,
        fallback_mask: Option<&GrayImage>,
        cur_rot: Option<&Matrix3<f64>>,
        k: &Matrix3<f64>,
    ) -> Result<(GrayImage, MaskFrameTelemetry), PipelineError> {
        let luma = grayscale(rgb);
        self.mask_frame_with_luma(rgb, &luma, fallback_mask, cur_rot, k)
    }

    /// Compute the binary valid-terrain mask using a precomputed grayscale image `luma`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when `Segformer` inference or mask post-processing fails.
    pub fn mask_frame_with_luma(
        &mut self,
        rgb: &RgbImage,
        luma: &GrayImage,
        fallback_mask: Option<&GrayImage>,
        cur_rot: Option<&Matrix3<f64>>,
        k: &Matrix3<f64>,
    ) -> Result<(GrayImage, MaskFrameTelemetry), PipelineError> {
        let t0 = Instant::now();
        let (w, h) = rgb.dimensions();
        let total = (w as usize).saturating_mul(h as usize);
        if total == 0 || luma.dimensions() != (w, h) {
            return Ok((
                GrayImage::new(w, h),
                MaskFrameTelemetry {
                    mask_ms: 0.0,
                    segformer_refreshed: false,
                    segformer_submitted: false,
                    segformer_ms: None,
                    valid_fraction: 0.0,
                    propeller_pixels: 0,
                },
            ));
        }

        let prop_occluded = self.propeller.mask_luma(luma);
        let prop_pixels = prop_occluded.iter().filter(|&&b| b).count();
        let mut step = refresh::RefreshStep::default();

        if let Some(seg) = &mut self.seg {
            step = seg.step(&mut self.key, rgb, cur_rot, self.refresh_interval)?;
        }
        let (raw_mask, valid_count) = if self.key.usable([w, h]) {
            compose_amortized_u8(
                [w, h],
                &self.key.airframe_keep,
                &self.key.sky_keep,
                &prop_occluded,
                self.key.rot.as_ref(),
                cur_rot,
                k,
            )
        } else if let Some(fb) = fallback_mask {
            compose_fallback_u8(total, fb.as_raw(), &prop_occluded)
        } else {
            compose_unmasked_u8(total, &prop_occluded)
        };

        let out = GrayImage::from_raw(w, h, raw_mask).unwrap_or_else(|| GrayImage::new(w, h));
        Ok((
            out,
            MaskFrameTelemetry {
                mask_ms: t0.elapsed().as_secs_f64() * 1000.0,
                segformer_refreshed: step.refreshed,
                segformer_submitted: step.submitted,
                segformer_ms: step.segformer_ms,
                valid_fraction: (valid_count as f64) / (total as f64),
                propeller_pixels: prop_pixels,
            },
        ))
    }

    /// Record the solved world-to-camera rotation of the most recent `Segformer` refresh frame.
    pub fn set_key_rotation(&mut self, rot: Matrix3<f64>) {
        self.key.rot = Some(rot);
    }

    /// Record the solved world-to-camera rotation of the most recent frame.
    pub fn note_solved_rotation(&mut self, rot: Matrix3<f64>) {
        if let Some(seg) = &mut self.seg {
            seg.note_solved_rotation(&mut self.key, rot);
        }
    }

    /// Check whether a `Segformer` model is configured.
    #[must_use]
    pub fn has_segformer(&self) -> bool {
        self.seg.is_some()
    }

    /// Check whether usable key layers exist for `dims`.
    #[must_use]
    pub fn has_usable_key(&self, dims: [u32; 2]) -> bool {
        self.key.usable(dims)
    }

    /// Run a fast initial `Segformer` pass on frame 1 before cold anchor jobs.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when background worker communication fails.
    pub fn fast_initial_pass(
        &mut self,
        rgb: &RgbImage,
        cur_rot: Option<&Matrix3<f64>>,
    ) -> Result<bool, PipelineError> {
        let Some(seg) = &mut self.seg else {
            return Ok(false);
        };
        seg.initial_fast_pass(&mut self.key, rgb, cur_rot)
    }

    /// Estimate `(down_deg, roll_deg)` from the cached `Segformer` sky mask if sky is visible.
    #[must_use]
    pub fn horizon_attitude(&self, k: &Matrix3<f64>) -> Option<(f64, f64)> {
        horizon_from_keep_layers(
            self.key.dims,
            &self.key.sky_keep,
            &self.key.airframe_keep,
            k,
        )
    }
}
