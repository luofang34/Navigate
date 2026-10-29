//! Temporal propeller-blade detector and video frame mask compositor (`video_masks`).

use crate::{
    SegmentError,
    config::{AIRFRAME, PROPELLER, SKY, SegformerConfig},
    morphology::{connected_components_8, dilate_rect, erode_rect, morph_close, morph_open},
};
use image::{DynamicImage, GrayImage, RgbImage, imageops::FilterType};
use std::collections::VecDeque;

const DETECTOR_WIDTH: u32 = 480;
const HISTORY_LEN: usize = 8;

/// Temporal transient-darkness detector for spinning propeller blades in video streams.
#[derive(Clone, Debug)]
pub struct PropellerDetector {
    history: VecDeque<Vec<u8>>,
    padding_pixels: usize,
    min_area_fraction: f64,
}

impl Default for PropellerDetector {
    fn default() -> Self {
        Self::new(3, 0.002)
    }
}

impl PropellerDetector {
    /// Construct a detector with `padding_pixels` dilation radius at width `480` and minimum
    /// component area fraction `min_area_fraction`.
    pub fn new(padding_pixels: usize, min_area_fraction: f64) -> Self {
        Self {
            history: VecDeque::with_capacity(HISTORY_LEN),
            padding_pixels,
            min_area_fraction,
        }
    }

    /// Update history with `rgb` and return a native-resolution propeller mask (`true` = blade).
    ///
    /// Requires at least three prior frames in `history` before claiming transient dark blades,
    /// so the first three frames always return an all-`false` mask.
    pub fn mask(&mut self, rgb: &RgbImage) -> Vec<bool> {
        let orig_w = rgb.width() as usize;
        let orig_h = rgb.height() as usize;
        let total = orig_w * orig_h;
        if total == 0 {
            return Vec::new();
        }
        let gray_full: GrayImage = DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
        let det_w = DETECTOR_WIDTH;
        let det_h = ((orig_h as f64 * f64::from(det_w) / orig_w as f64).round() as u32).max(1);
        let resized = if gray_full.dimensions() == (det_w, det_h) {
            gray_full
        } else {
            image::imageops::resize(&gray_full, det_w, det_h, FilterType::Triangle)
        };
        let cur = resized.into_raw();
        let w = det_w as usize;
        let h = det_h as usize;
        let n = w * h;

        if self.history.len() < 3 {
            self.push_frame(cur);
            return vec![false; total];
        }

        let mut dark = vec![false; n];
        for (i, &val) in cur.iter().enumerate() {
            let mut top1 = 0u8;
            let mut top2 = 0u8;
            for past in &self.history {
                let pv = past[i];
                if pv >= top1 {
                    top2 = top1;
                    top1 = pv;
                } else if pv > top2 {
                    top2 = pv;
                }
            }
            dark[i] = i32::from(val) < i32::from(top2) - 35 && val < 110;
        }
        self.push_frame(cur);

        let opened = morph_open(&dark, w, h, 1);
        let min_area = ((n as f64) * self.min_area_fraction) as usize;
        let (comp_labels, stats) = connected_components_8(&opened, w, h);
        let mut keep_comp = vec![false; stats.len() + 1];
        let mut any_kept = false;
        for stat in &stats {
            if stat.area >= min_area {
                keep_comp[stat.label as usize] = true;
                any_kept = true;
            }
        }
        if !any_kept {
            return vec![false; total];
        }
        let filtered: Vec<bool> = comp_labels
            .into_iter()
            .map(|l| l != 0 && keep_comp[l as usize])
            .collect();
        let padded = dilate_rect(&filtered, w, h, self.padding_pixels);
        if (w, h) == (orig_w, orig_h) {
            return padded;
        }
        let mut native = vec![false; total];
        for y in 0..orig_h {
            let sy = ((y * h) / orig_h).min(h - 1);
            for x in 0..orig_w {
                let sx = ((x * w) / orig_w).min(w - 1);
                native[y * orig_w + x] = padded[sy * w + sx];
            }
        }
        native
    }

    fn push_frame(&mut self, frame: Vec<u8>) {
        if self.history.len() == HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back(frame);
    }
}

/// Decomposed per-class occlusion masks and final matchable-terrain mask for one video frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameMaskResult {
    /// Final eroded terrain mask (`true` = usable ground/water pixel).
    pub valid: Vec<bool>,
    /// Airframe mask (`true` = airframe pixel).
    pub airframe: Vec<bool>,
    /// Propeller mask (`true` = propeller pixel).
    pub propeller: Vec<bool>,
    /// Sky mask (`true` = sky pixel).
    pub sky: Vec<bool>,
}

/// Decomposed `(airframe, propeller, sky)` binary masks.
pub type ClassMaskTriple = (Vec<bool>, Vec<bool>, Vec<bool>);

/// Extract `(airframe, propeller, sky)` masks from semantic predictions for either a 4-class
/// fine-tuned model (`sky`, `ground`, `airframe`, `propeller`) or a 150-class ADE20K model
/// combined with a [`PropellerDetector`].
///
/// # Errors
///
/// Returns [`SegmentError::Shape`] when `labels` or `confidence` lengths do not match `rgb`.
pub fn class_masks(
    rgb: &RgbImage,
    labels: &[u8],
    confidence: &[f32],
    config: &SegformerConfig,
    threshold: f32,
    detector: Option<&mut PropellerDetector>,
) -> Result<ClassMaskTriple, SegmentError> {
    let n = (rgb.width() as usize) * (rgb.height() as usize);
    if labels.len() != n || confidence.len() != n {
        return Err(SegmentError::Shape {
            name: "video class_masks".into(),
            expected: vec![n],
            actual: vec![labels.len()],
        });
    }

    let is_four_class =
        config.class_index("airframe").is_some() && config.class_index("propeller").is_some();
    if is_four_class {
        let mut airframe = vec![false; n];
        let mut propeller = vec![false; n];
        let mut sky = vec![false; n];
        for i in 0..n {
            if confidence[i] >= threshold {
                match labels[i] {
                    AIRFRAME => airframe[i] = true,
                    PROPELLER => propeller[i] = true,
                    SKY => sky[i] = true,
                    _ => {}
                }
            }
        }
        Ok((airframe, propeller, sky))
    } else {
        let air_idx = config.class_index("airplane").unwrap_or(90) as u8;
        let sky_idx = config.class_index("sky").unwrap_or(2) as u8;
        let mut airframe = vec![false; n];
        let mut sky = vec![false; n];
        for i in 0..n {
            if confidence[i] >= threshold {
                if labels[i] == air_idx {
                    airframe[i] = true;
                } else if labels[i] == sky_idx {
                    sky[i] = true;
                }
            }
        }
        let propeller = match detector {
            Some(det) => det.mask(rgb),
            None => vec![false; n],
        };
        Ok((airframe, propeller, sky))
    }
}

/// Compose a [`FrameMaskResult`] with morphological closing (`9 × 9`) and `margin_pixels` erosion.
///
/// # Errors
///
/// Returns [`SegmentError`] when input dimensions mismatch.
pub fn frame_masks(
    rgb: &RgbImage,
    labels: &[u8],
    confidence: &[f32],
    config: &SegformerConfig,
    threshold: f32,
    margin_pixels: usize,
    detector: Option<&mut PropellerDetector>,
) -> Result<FrameMaskResult, SegmentError> {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let (airframe, propeller, sky) =
        class_masks(rgb, labels, confidence, config, threshold, detector)?;
    let excluded: Vec<bool> = (0..w * h)
        .map(|i| airframe[i] || propeller[i] || sky[i])
        .collect();
    let closed = morph_close(&excluded, w, h, 4);
    let raw_valid: Vec<bool> = closed.into_iter().map(|e| !e).collect();
    let valid = erode_rect(&raw_valid, w, h, margin_pixels, true);
    Ok(FrameMaskResult {
        valid,
        airframe,
        propeller,
        sky,
    })
}
