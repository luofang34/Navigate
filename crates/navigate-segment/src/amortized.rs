//! Sans-IO amortized live semantic occlusion masking, sky warping, and propeller gating.

pub mod compose;
#[cfg(test)]
mod tests;

pub use compose::{compose_amortized_u8, compose_fallback_u8, compose_unmasked_u8};

use crate::{
    AIRFRAME, ColorMode, GROUND, MaskPolicy, PROPELLER, PreprocessorConfig, PropellerDetector, SKY,
    Segformer, SegformerConfig, SegmentError, horizon::horizon_attitude, morphology::dilate_rect,
    surface_mask,
};
use burn::tensor::Device;
use image::{GrayImage, RgbImage, imageops::colorops::grayscale};
use nalgebra::Matrix3;
use web_time::Instant;

/// Per-frame masking telemetry from [`AmortizedMasker::mask_frame`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskFrameTelemetry {
    /// Elapsed wall-clock time for mask generation in milliseconds.
    pub mask_ms: f64,
    /// `true` when new `Segformer` key layers took effect on this frame.
    pub segformer_refreshed: bool,
    /// `true` when this frame went to a background `Segformer` worker.
    pub segformer_submitted: bool,
    /// `Segformer` evaluation time in milliseconds for refreshed key layers.
    pub segformer_ms: Option<f64>,
    /// Fraction of pixels retained as valid matchable terrain (`[0.0, 1.0]`).
    pub valid_fraction: f64,
    /// Number of pixels excluded by [`PropellerDetector`] on this frame.
    pub propeller_pixels: usize,
}

/// Key layers computed from one `Segformer` evaluation.
#[derive(Clone, Debug)]
pub struct RefreshLayers {
    /// Image dimensions `[width, height]`.
    pub dims: [u32; 2],
    /// Airframe and propeller keep mask (`255` = keep, `0` = occluded).
    pub airframe_keep: Vec<u8>,
    /// Sky keep mask (`255` = ground, `0` = sky).
    pub sky_keep: Vec<u8>,
    /// Inference wall-clock duration in milliseconds.
    pub eval_ms: f64,
}

/// Sans-IO `Segformer` evaluation engine for amortized key-layer refreshes.
pub struct SegEngine {
    /// Loaded `Segformer` network.
    pub model: Segformer,
    /// Preprocessor configuration.
    pub preproc: PreprocessorConfig,
    /// Class indices treated as airframe obstructions.
    pub airplane_indices: Vec<usize>,
    /// Class index treated as sky.
    pub sky_label: u8,
    /// `true` when the checkpoint uses the 4-class Navigate label space.
    pub is_four_class: bool,
    /// Burn execution device.
    pub device: Device,
    /// Mask composition policy.
    pub policy: MaskPolicy,
    /// Morphological dilation margin in pixels.
    pub margin_px: usize,
}

impl SegEngine {
    /// Construct a [`SegEngine`] from an in-memory `Segformer` model and configuration.
    #[must_use]
    pub fn from_loaded(
        model: Segformer,
        cfg: &SegformerConfig,
        preproc: PreprocessorConfig,
        policy: MaskPolicy,
        margin_px: usize,
        device: Device,
    ) -> Self {
        let sky_label = cfg.class_index("sky").unwrap_or(2) as u8;
        let airplane_indices = cfg.class_index("airplane").into_iter().collect();
        let is_four_class = cfg.num_labels() == 4;
        Self {
            model,
            preproc,
            airplane_indices,
            sky_label,
            is_four_class,
            device,
            policy,
            margin_px,
        }
    }

    /// Cap the preprocessor input resolution to `max_side` pixels per axis.
    pub fn clamp_input_side(&mut self, max_side: u32) {
        let (w, h) = self.preproc.target_size();
        self.preproc.size = serde_json::json!({
            "width": w.min(max_side),
            "height": h.min(max_side),
        });
    }

    /// Run one `Segformer` key-layer evaluation on `rgb`.
    ///
    /// # Errors
    /// Returns [`SegmentError`] if model evaluation or mask extraction fails.
    pub fn refresh(&self, rgb: &RgbImage) -> Result<RefreshLayers, SegmentError> {
        let t0 = Instant::now();
        let (airframe_keep, sky_keep) = self.eval(rgb)?;
        Ok(RefreshLayers {
            dims: [rgb.width(), rgb.height()],
            airframe_keep,
            sky_keep,
            eval_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn eval(&self, rgb: &RgbImage) -> Result<(Vec<u8>, Vec<u8>), SegmentError> {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let pred = self.model.predict_image(
            rgb,
            &self.preproc,
            ColorMode::Color,
            &self.airplane_indices,
            &self.device,
        )?;
        if self.is_four_class {
            Ok(four_class_keep_layers(
                &pred.labels,
                self.sky_label,
                w,
                h,
                self.margin_px,
            ))
        } else {
            let air_probs = pred.selected_probs.first().map(Vec::as_slice);
            let (combined_keep, _) = surface_mask(
                rgb,
                &pred.labels,
                &pred.confidence,
                air_probs,
                self.policy,
                self.margin_px,
            )?;
            let raw_sky: Vec<bool> = pred
                .labels
                .iter()
                .zip(&pred.confidence)
                .map(|(&l, &c)| l == self.sky_label && c >= 0.35)
                .collect();
            let sky_occ = dilate_rect(&raw_sky, w, h, self.margin_px);
            let sky_keep: Vec<u8> = sky_occ.iter().map(|&b| if b { 0 } else { 255 }).collect();
            let air_keep: Vec<u8> = combined_keep
                .iter()
                .zip(&sky_occ)
                .map(|(&keep_px, &is_sky)| if keep_px || is_sky { 255 } else { 0 })
                .collect();
            Ok((air_keep, sky_keep))
        }
    }
}

/// Split 4-class Navigate semantic predictions into dilated `(airframe_keep, sky_keep)` byte masks.
#[must_use]
pub fn four_class_keep_layers(
    labels: &[u8],
    sky_label: u8,
    w: usize,
    h: usize,
    margin_px: usize,
) -> (Vec<u8>, Vec<u8>) {
    let raw_sky: Vec<bool> = labels.iter().map(|&l| l == sky_label).collect();
    let raw_air: Vec<bool> = labels
        .iter()
        .map(|&l| matches!(l, AIRFRAME | PROPELLER))
        .collect();
    let sky_occ = dilate_rect(&raw_sky, w, h, margin_px);
    let air_occ = dilate_rect(&raw_air, w, h, margin_px);
    (
        air_occ
            .into_iter()
            .map(|b| if b { 0 } else { 255 })
            .collect(),
        sky_occ
            .into_iter()
            .map(|b| if b { 0 } else { 255 })
            .collect(),
    )
}

/// Sans-IO inline amortized frame masker combining `Segformer`, sky warping, and [`PropellerDetector`].
pub struct AmortizedMasker {
    seg: Option<Box<SegEngine>>,
    refresh_interval: usize,
    propeller: PropellerDetector,
    dims: [u32; 2],
    airframe_keep: Vec<u8>,
    sky_keep: Vec<u8>,
    key_rot: Option<Matrix3<f64>>,
    frames_since_refresh: usize,
    refreshed_on_current: bool,
}

impl AmortizedMasker {
    /// Construct a sans-IO [`AmortizedMasker`] from an optional loaded `Segformer` engine.
    #[must_use]
    pub fn from_engine(seg: Option<SegEngine>, refresh_interval: usize) -> Self {
        Self {
            seg: seg.map(Box::new),
            refresh_interval: refresh_interval.max(1),
            propeller: PropellerDetector::default(),
            dims: [0, 0],
            airframe_keep: Vec::new(),
            sky_keep: Vec::new(),
            key_rot: None,
            frames_since_refresh: 0,
            refreshed_on_current: false,
        }
    }

    /// Return `true` when a `Segformer` engine is configured.
    #[must_use]
    pub fn has_segformer(&self) -> bool {
        self.seg.is_some()
    }

    /// Return `true` when cached key layers match `dims`.
    #[must_use]
    pub fn has_usable_key(&self, dims: [u32; 2]) -> bool {
        let total = (dims[0] as usize).saturating_mul(dims[1] as usize);
        self.dims == dims && self.airframe_keep.len() == total && self.sky_keep.len() == total
    }

    /// Record the solved world-to-camera rotation of the key refresh frame.
    pub fn set_key_rotation(&mut self, rot: Matrix3<f64>) {
        self.key_rot = Some(rot);
    }

    /// Record the solved world-to-camera rotation when key layers refreshed on the current frame.
    pub fn note_solved_rotation(&mut self, rot: Matrix3<f64>) {
        if self.refreshed_on_current {
            self.key_rot = Some(rot);
        }
    }

    /// Compute the binary valid-terrain mask (`255` = keep, `0` = occluded) for `rgb`.
    ///
    /// # Errors
    /// Returns [`SegmentError`] when `Segformer` evaluation fails.
    pub fn mask_frame(
        &mut self,
        rgb: &RgbImage,
        fallback_mask: Option<&GrayImage>,
        cur_rot: Option<&Matrix3<f64>>,
        k: &Matrix3<f64>,
    ) -> Result<(GrayImage, MaskFrameTelemetry), SegmentError> {
        let luma = grayscale(rgb);
        self.mask_frame_with_luma(rgb, &luma, fallback_mask, cur_rot, k)
    }

    /// Compute the binary valid-terrain mask using a precomputed grayscale image `luma`.
    ///
    /// # Errors
    /// Returns [`SegmentError`] when `Segformer` evaluation fails.
    pub fn mask_frame_with_luma(
        &mut self,
        rgb: &RgbImage,
        luma: &GrayImage,
        fallback_mask: Option<&GrayImage>,
        cur_rot: Option<&Matrix3<f64>>,
        k: &Matrix3<f64>,
    ) -> Result<(GrayImage, MaskFrameTelemetry), SegmentError> {
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
        self.refreshed_on_current = false;
        let mut segformer_ms = None;

        if let Some(engine) = &self.seg {
            if !self.has_usable_key([w, h]) || self.frames_since_refresh >= self.refresh_interval {
                let layers = engine.refresh(rgb)?;
                segformer_ms = Some(layers.eval_ms);
                self.dims = layers.dims;
                self.airframe_keep = layers.airframe_keep;
                self.sky_keep = layers.sky_keep;
                self.key_rot = cur_rot.copied();
                self.frames_since_refresh = 1;
                self.refreshed_on_current = true;
            } else {
                self.frames_since_refresh += 1;
            }
        }

        let (raw_mask, valid_count) = if self.has_usable_key([w, h]) {
            compose_amortized_u8(
                [w, h],
                &self.airframe_keep,
                &self.sky_keep,
                &prop_occluded,
                self.key_rot.as_ref(),
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
                segformer_refreshed: self.refreshed_on_current,
                segformer_submitted: false,
                segformer_ms,
                valid_fraction: (valid_count as f64) / (total as f64),
                propeller_pixels: prop_pixels,
            },
        ))
    }

    /// Estimate `(down_deg, roll_deg)` from the cached sky and airframe layers.
    #[must_use]
    pub fn horizon_attitude(&self, k: &Matrix3<f64>) -> Option<(f64, f64)> {
        horizon_from_keep_layers(self.dims, &self.sky_keep, &self.airframe_keep, k)
    }
}

/// Estimate `(down_deg, roll_deg)` from cached `sky_keep` and `airframe_keep` byte masks.
#[must_use]
pub fn horizon_from_keep_layers(
    [w, h]: [u32; 2],
    sky_keep: &[u8],
    airframe_keep: &[u8],
    k: &Matrix3<f64>,
) -> Option<(f64, f64)> {
    let total = (w as usize) * (h as usize);
    if sky_keep.len() != total || airframe_keep.len() != total || w < 8 || h < 8 {
        return None;
    }
    if sky_keep.iter().all(|&keep| keep != 0) {
        return None;
    }
    let map: Vec<u8> = sky_keep
        .iter()
        .zip(airframe_keep)
        .map(|(&sky, &air)| {
            if air == 0 {
                AIRFRAME
            } else if sky == 0 {
                SKY
            } else {
                GROUND
            }
        })
        .collect();
    horizon_attitude(&map, w, h, k)
}
