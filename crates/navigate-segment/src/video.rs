//! Temporal propeller-blade detector and video frame mask compositor (`video_masks`).

use crate::{
    SegmentError,
    config::{AIRFRAME, PROPELLER, SKY, SegformerConfig},
    morphology::{connected_components_8, dilate_rect, erode_rect, morph_close, morph_open},
};
use image::{GrayImage, RgbImage, imageops::colorops::grayscale};
use std::collections::VecDeque;

const DETECTOR_WIDTH: u32 = 480;
const HISTORY_LEN: usize = 8;
const SCAN_CHUNK: usize = 64;

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
        if rgb.width() == 0 || rgb.height() == 0 {
            return Vec::new();
        }
        let gray_full = grayscale(rgb);
        self.mask_luma(&gray_full)
    }

    /// Update history from a grayscale image `gray` and return a native-resolution blade mask.
    ///
    /// Returns an all-`false` mask until `history` holds at least three prior frames.
    pub fn mask_luma(&mut self, gray: &GrayImage) -> Vec<bool> {
        let orig_w = gray.width() as usize;
        let orig_h = gray.height() as usize;
        let total = orig_w * orig_h;
        if total == 0 {
            return Vec::new();
        }
        let det_w = DETECTOR_WIDTH;
        let det_h = ((orig_h as f64 * f64::from(det_w) / orig_w as f64).round() as u32).max(1);
        let cur = resize_luma_u8(gray, det_w, det_h);
        let w = det_w as usize;
        let h = det_h as usize;
        let n = w * h;

        if self.history.len() < 3 {
            self.push_frame(cur);
            return vec![false; total];
        }

        let (dark, any_dark) = detect_transient_dark(&cur, &self.history);
        self.push_frame(cur);
        if !any_dark {
            return vec![false; total];
        }

        let opened = morph_open(&dark, w, h, 1);
        if !opened.iter().any(|&b| b) {
            return vec![false; total];
        }
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
        upsample_nearest_bool(&padded, w, h, orig_w, orig_h)
    }

    fn push_frame(&mut self, frame: Vec<u8>) {
        if self.history.len() == HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back(frame);
    }
}

fn detect_transient_dark(cur: &[u8], history: &VecDeque<Vec<u8>>) -> (Vec<bool>, bool) {
    let n = cur.len();
    let mut dark = vec![false; n];
    let mut any_dark = false;
    let mut base = 0usize;
    while base + SCAN_CHUNK <= n {
        let mut top1 = [0u8; SCAN_CHUNK];
        let mut top2 = [0u8; SCAN_CHUNK];
        for past in history {
            let ps = &past[base..base + SCAN_CHUNK];
            for k in 0..SCAN_CHUNK {
                let pv = ps[k];
                top2[k] = top2[k].max(top1[k].min(pv));
                top1[k] = top1[k].max(pv);
            }
        }
        let cs = &cur[base..base + SCAN_CHUNK];
        let ds = &mut dark[base..base + SCAN_CHUNK];
        let mut chunk_or = 0u8;
        for k in 0..SCAN_CHUNK {
            let is_dark = cs[k] < top2[k].saturating_sub(35) && cs[k] < 110;
            ds[k] = is_dark;
            chunk_or |= u8::from(is_dark);
        }
        any_dark |= chunk_or != 0;
        base += SCAN_CHUNK;
    }
    for i in base..n {
        let mut top1 = 0u8;
        let mut top2 = 0u8;
        for past in history {
            let pv = past[i];
            top2 = top2.max(top1.min(pv));
            top1 = top1.max(pv);
        }
        let is_dark = cur[i] < top2.saturating_sub(35) && cur[i] < 110;
        dark[i] = is_dark;
        any_dark |= is_dark;
    }
    (dark, any_dark)
}

fn resize_luma_u8(gray: &GrayImage, dst_w: u32, dst_h: u32) -> Vec<u8> {
    if gray.dimensions() == (dst_w, dst_h) {
        return gray.as_raw().clone();
    }
    let sw = gray.width() as usize;
    let sh = gray.height() as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let src = gray.as_raw();
    let build_axis = |src_len: usize, dst_len: usize| -> Vec<(usize, usize, u32)> {
        let scale = (src_len as f64) / (dst_len as f64);
        (0..dst_len)
            .map(|d| {
                let pos = ((d as f64) + 0.5) * scale - 0.5;
                if pos <= 0.0 {
                    (0, 0, 0)
                } else if pos >= (src_len - 1) as f64 {
                    (src_len - 1, src_len - 1, 0)
                } else {
                    let i0 = pos as usize;
                    let i1 = (i0 + 1).min(src_len - 1);
                    let frac = ((pos - (i0 as f64)) * 256.0).round() as u32;
                    (i0, i1, frac.min(256))
                }
            })
            .collect()
    };
    let x_tab = build_axis(sw, dw);
    let y_tab = build_axis(sh, dh);
    let mut out = vec![0u8; dw * dh];
    for (dy, &(y0, y1, wy)) in y_tab.iter().enumerate() {
        let r0 = &src[y0 * sw..(y0 + 1) * sw];
        let r1 = &src[y1 * sw..(y1 + 1) * sw];
        let w_y0 = 256 - wy;
        let out_row = &mut out[dy * dw..(dy + 1) * dw];
        for (dx, &(x0, x1, wx)) in x_tab.iter().enumerate() {
            let w_x0 = 256 - wx;
            let top = u32::from(r0[x0]) * w_x0 + u32::from(r0[x1]) * wx;
            let bot = u32::from(r1[x0]) * w_x0 + u32::from(r1[x1]) * wx;
            out_row[dx] = ((top * w_y0 + bot * wy + 32_768) >> 16) as u8;
        }
    }
    out
}

fn upsample_nearest_bool(
    padded: &[bool],
    w: usize,
    h: usize,
    orig_w: usize,
    orig_h: usize,
) -> Vec<bool> {
    let sx_lut: Vec<usize> = (0..orig_w).map(|x| ((x * w) / orig_w).min(w - 1)).collect();
    let mut native = vec![false; orig_w * orig_h];
    let mut prev_sy = usize::MAX;
    for y in 0..orig_h {
        let sy = ((y * h) / orig_h).min(h - 1);
        if y > 0 && sy == prev_sy {
            native.copy_within((y - 1) * orig_w..y * orig_w, y * orig_w);
            continue;
        }
        prev_sy = sy;
        let src_row = &padded[sy * w..(sy + 1) * w];
        let dst_row = &mut native[y * orig_w..(y + 1) * orig_w];
        for (dst, &sx) in dst_row.iter_mut().zip(&sx_lut) {
            *dst = src_row[sx];
        }
    }
    native
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
