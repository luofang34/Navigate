//! Amortized live semantic occlusion masking, mount gating, and luminance night bypass.

use crate::{error::FlightVisualError, pixels::median_luma_u8};
use burn::tensor::Device;
use image::{
    GrayImage, Luma, RgbImage,
    imageops::{FilterType, colorops::grayscale, resize},
};
use nalgebra::Matrix3;
use navigate_camera::SpectralBand;
pub use navigate_match::segment::amortized::MaskFrameTelemetry;
use navigate_match::segment::{
    MaskPolicy, PropellerDetector, Segformer,
    amortized::{
        SegEngine, compose_amortized_u8, compose_fallback_u8, compose_unmasked_u8,
        horizon_from_keep_layers,
    },
};
use std::{path::Path, time::Instant};

mod refresh;
#[cfg(test)]
mod tests;

use refresh::{KeyLayers, Refresher};

/// Minimum median luma (`0..=255`) required to run daytime `Segformer` airframe masking.
pub const NIGHT_BYPASS_LUMA_P50: u8 = 35;

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
    /// Returns [`FlightVisualError`] when `policy_str` or `seg_model_dir` fails to load.
    pub fn new(
        seg_model_dir: Option<&Path>,
        policy_str: &str,
        margin_px: usize,
        refresh_interval: usize,
        device: &Device,
    ) -> Result<Self, FlightVisualError> {
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

    /// Run `Segformer` refreshes on a named single-slot supersede background thread.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when the worker thread cannot start.
    pub fn with_background_refresh(mut self) -> Result<Self, FlightVisualError> {
        self.enable_background_refresh()?;
        Ok(self)
    }

    /// Move `Segformer` refreshes to a named single-slot supersede background worker thread.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when the worker thread cannot start.
    pub fn enable_background_refresh(&mut self) -> Result<(), FlightVisualError> {
        self.seg = self
            .seg
            .take()
            .map(Refresher::into_background)
            .transpose()?;
        Ok(())
    }

    /// Clear cached frame masks and propeller history before a new session.
    pub fn reset_session(&mut self) {
        self.propeller = PropellerDetector::default();
        self.key = KeyLayers::default();
    }

    /// Return the cumulative count of superseded background `Segformer` refresh jobs.
    #[must_use]
    pub fn superseded_count(&self) -> u32 {
        self.seg.as_ref().map_or(0, Refresher::superseded_count)
    }

    /// Compute the live flight valid-terrain mask gated on mount occlusion and scene luminance.
    ///
    /// When `has_airframe_occlusion` is false, this method skips `Segformer` and propeller masking
    /// completely. When median luma `luma_p50 < 35`, this method bypasses `Segformer` so dark
    /// night terrain is never misclassified as airframe or sky.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when `Segformer` inference or worker communication fails.
    #[expect(
        clippy::too_many_arguments,
        reason = "live flight masking accepts mount flag, optional RGB, luma, static mask, attitude, intrinsics, and refresh gate"
    )]
    pub fn mask_frame_flight(
        &mut self,
        has_airframe_occlusion: bool,
        rgb: Option<&RgbImage>,
        luma: &GrayImage,
        fallback_mask: Option<&GrayImage>,
        cur_rot: Option<&Matrix3<f64>>,
        k: &Matrix3<f64>,
        allow_refresh: bool,
    ) -> Result<(GrayImage, MaskFrameTelemetry, u8), FlightVisualError> {
        self.mask_frame_flight_fast(
            has_airframe_occlusion,
            rgb,
            luma,
            fallback_mask,
            cur_rot,
            k,
            allow_refresh,
            false,
            SpectralBand::Visible,
        )
    }

    /// Compute the binary valid-terrain mask and skip propeller morphology when `skip_propeller` is true.
    ///
    /// A frame outside the visible band skips `Segformer` as a night frame does,
    /// because the model learned visible colours only.
    #[expect(
        clippy::too_many_arguments,
        reason = "live flight masking accepts mount flag, optional RGB, luma, static mask, attitude, intrinsics, refresh gate, propeller bypass, and spectral band"
    )]
    pub(crate) fn mask_frame_flight_fast(
        &mut self,
        has_airframe_occlusion: bool,
        rgb: Option<&RgbImage>,
        luma: &GrayImage,
        fallback_mask: Option<&GrayImage>,
        cur_rot: Option<&Matrix3<f64>>,
        k: &Matrix3<f64>,
        allow_refresh: bool,
        skip_propeller: bool,
        band: SpectralBand,
    ) -> Result<(GrayImage, MaskFrameTelemetry, u8), FlightVisualError> {
        let t0 = Instant::now();
        let (w, h) = luma.dimensions();
        let total = (w as usize).saturating_mul(h as usize);
        let luma_p50 = median_luma_u8(luma);
        if total == 0 {
            return Ok((GrayImage::new(w, h), empty_mask_telemetry(), luma_p50));
        }
        if !has_airframe_occlusion {
            let (out, valid_fraction) = passthrough_static_mask(w, h, total, fallback_mask);
            return Ok((
                out,
                MaskFrameTelemetry {
                    mask_ms: t0.elapsed().as_secs_f64() * 1000.0,
                    segformer_refreshed: false,
                    segformer_submitted: false,
                    segformer_ms: None,
                    valid_fraction,
                    propeller_pixels: 0,
                },
                luma_p50,
            ));
        }

        let (prop_occluded, prop_pixels) = if skip_propeller && self.key.usable([w, h]) {
            (vec![false; total], 0)
        } else {
            let occ = self.propeller.mask_luma(luma);
            let cnt = occ.iter().filter(|&&b| b).count();
            (occ, cnt)
        };
        let skip_semantic = luma_p50 < NIGHT_BYPASS_LUMA_P50 || !runs_semantic_mask(band);
        let mut step = refresh::RefreshStep::default();

        if !skip_semantic && let Some(seg) = &mut self.seg {
            step = seg.step(
                &mut self.key,
                [w, h],
                rgb,
                cur_rot,
                self.refresh_interval,
                allow_refresh,
            )?;
        }

        let (raw_mask, valid_count) = if !skip_semantic && self.key.usable([w, h]) {
            let (composed, cnt) = compose_amortized_u8(
                [w, h],
                &self.key.airframe_keep,
                &self.key.sky_keep,
                &prop_occluded,
                self.key.rot.as_ref(),
                cur_rot,
                k,
            );
            let air_cnt = self.key.airframe_keep_count;
            let overmasked = cnt * 100 < total * 4 || (luma_p50 < 75 && air_cnt * 100 < total * 45);
            if overmasked {
                fallback_or_unmasked(w, h, total, fallback_mask, &prop_occluded)
            } else {
                (composed, cnt)
            }
        } else {
            fallback_or_unmasked(w, h, total, fallback_mask, &prop_occluded)
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
            luma_p50,
        ))
    }

    /// Compute the binary valid-terrain mask (`255` = keep, `0` = occluded) for one `RGB` frame.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when `Segformer` inference or post-processing fails.
    pub fn mask_frame(
        &mut self,
        rgb: &RgbImage,
        fallback_mask: Option<&GrayImage>,
        cur_rot: Option<&Matrix3<f64>>,
        k: &Matrix3<f64>,
    ) -> Result<(GrayImage, MaskFrameTelemetry), FlightVisualError> {
        let luma = grayscale(rgb);
        self.mask_frame_with_luma(rgb, &luma, fallback_mask, cur_rot, k)
    }

    /// Compute the binary valid-terrain mask using a precomputed grayscale image `luma`.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when `Segformer` inference or post-processing fails.
    pub fn mask_frame_with_luma(
        &mut self,
        rgb: &RgbImage,
        luma: &GrayImage,
        fallback_mask: Option<&GrayImage>,
        cur_rot: Option<&Matrix3<f64>>,
        k: &Matrix3<f64>,
    ) -> Result<(GrayImage, MaskFrameTelemetry), FlightVisualError> {
        self.mask_frame_with_luma_controlled(rgb, luma, fallback_mask, cur_rot, k, true)
    }

    /// Compute the binary valid-terrain mask while optionally pausing new `Segformer` submissions.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when `Segformer` inference or post-processing fails.
    pub fn mask_frame_with_luma_controlled(
        &mut self,
        rgb: &RgbImage,
        luma: &GrayImage,
        fallback_mask: Option<&GrayImage>,
        cur_rot: Option<&Matrix3<f64>>,
        k: &Matrix3<f64>,
        allow_refresh: bool,
    ) -> Result<(GrayImage, MaskFrameTelemetry), FlightVisualError> {
        if rgb.dimensions() != luma.dimensions() {
            let (w, h) = rgb.dimensions();
            return Ok((GrayImage::new(w, h), empty_mask_telemetry()));
        }
        let (out, tel, _) = self.mask_frame_flight(
            true,
            Some(rgb),
            luma,
            fallback_mask,
            cur_rot,
            k,
            allow_refresh,
        )?;
        Ok((out, tel))
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

    /// Return true when a `Segformer` model is configured.
    #[must_use]
    pub fn has_segformer(&self) -> bool {
        self.seg.is_some()
    }

    /// Return true when usable key layers exist for `dims`.
    #[must_use]
    pub fn has_usable_key(&self, dims: [u32; 2]) -> bool {
        self.key.usable(dims)
    }

    /// Return true when a live warm-tracked frame can reuse the warped key mask without submitting `Segformer`.
    #[must_use]
    pub fn should_throttle_live_warm_refresh(
        &self,
        dims: [u32; 2],
        cur_rot: Option<&Matrix3<f64>>,
    ) -> bool {
        self.should_throttle_live_warm_refresh_for_runtime(dims, cur_rot, false)
    }

    /// Return true when a live warm-tracked frame on the active runtime can defer `Segformer`.
    #[must_use]
    pub(crate) fn should_throttle_live_warm_refresh_for_runtime(
        &self,
        dims: [u32; 2],
        cur_rot: Option<&Matrix3<f64>>,
        is_coreml: bool,
    ) -> bool {
        let mult = if is_coreml { 8 } else { 4 };
        if !self.key.usable(dims) || self.key.frames_since_refresh() >= self.refresh_interval * mult
        {
            return false;
        }
        let total = (dims[0] as usize).saturating_mul(dims[1] as usize);
        let min_keep_pct = if is_coreml { 20 } else { 30 };
        if total == 0 || self.key.airframe_keep_count * 100 < total * min_keep_pct {
            return false;
        }
        if is_coreml {
            return true;
        }
        match (self.key.rot.as_ref(), cur_rot) {
            (Some(kr), Some(cr)) => (((cr * kr.transpose()).trace() - 1.0) * 0.5) >= 0.92,
            _ => false,
        }
    }

    /// Return true when `mask_frame_flight` may evaluate or submit `Segformer` on this frame.
    #[must_use]
    pub fn needs_rgb_for_step(
        &self,
        has_airframe_occlusion: bool,
        dims: [u32; 2],
        allow_refresh: bool,
    ) -> bool {
        if !has_airframe_occlusion
            || self.seg.is_none()
            || !allow_refresh
            || self.seg.as_ref().is_some_and(Refresher::has_pending)
        {
            return false;
        }
        !self.key.usable(dims) || self.key.frames_since_refresh() + 1 >= self.refresh_interval
    }

    /// Cap the `Segformer` preprocessor input side length in pixels for fast inline inference.
    pub fn clamp_input_side(&mut self, max_side: u32) {
        if let Some(seg) = &mut self.seg {
            seg.clamp_input_side(max_side);
        }
    }

    /// Run a fast initial `Segformer` pass before cold anchor jobs.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when inference or worker communication fails.
    pub fn fast_initial_pass(
        &mut self,
        rgb: &RgbImage,
        cur_rot: Option<&Matrix3<f64>>,
    ) -> Result<Option<f64>, FlightVisualError> {
        let Some(seg) = &mut self.seg else {
            return Ok(None);
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

/// Whether the visible-trained `Segformer` applies to a frame in `band`.
#[must_use]
pub fn runs_semantic_mask(band: SpectralBand) -> bool {
    band == SpectralBand::Visible
}

fn empty_mask_telemetry() -> MaskFrameTelemetry {
    MaskFrameTelemetry {
        mask_ms: 0.0,
        segformer_refreshed: false,
        segformer_submitted: false,
        segformer_ms: None,
        valid_fraction: 0.0,
        propeller_pixels: 0,
    }
}

fn passthrough_static_mask(
    w: u32,
    h: u32,
    total: usize,
    fallback_mask: Option<&GrayImage>,
) -> (GrayImage, f64) {
    if let Some(fb) = fallback_mask {
        let scaled = if fb.dimensions() == (w, h) {
            fb.clone()
        } else {
            resize(fb, w, h, FilterType::Nearest)
        };
        let valid = scaled.as_raw().iter().filter(|&&b| b >= 128).count();
        return (scaled, (valid as f64) / (total as f64));
    }
    (GrayImage::from_pixel(w, h, Luma([255])), 1.0)
}

fn fallback_or_unmasked(
    w: u32,
    h: u32,
    total: usize,
    fallback_mask: Option<&GrayImage>,
    prop_occluded: &[bool],
) -> (Vec<u8>, usize) {
    if let Some(fb) = fallback_mask {
        if fb.dimensions() == (w, h) && fb.as_raw().len() == total {
            return compose_fallback_u8(total, fb.as_raw(), prop_occluded);
        }
        let scaled = resize(fb, w, h, FilterType::Nearest);
        if scaled.as_raw().len() == total {
            return compose_fallback_u8(total, scaled.as_raw(), prop_occluded);
        }
    }
    compose_unmasked_u8(total, prop_occluded)
}
