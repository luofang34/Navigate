//! Amortized live semantic occlusion masking, attitude sky warping, and propeller gating.

use crate::PipelineError;
use burn::tensor::Device;
use image::{GrayImage, Luma, RgbImage};
use nalgebra::{Matrix3, Vector3};
use navigate_segment::{
    ColorMode, MaskPolicy, PreprocessorConfig, PropellerDetector, Segformer,
    morphology::dilate_rect, surface_mask,
};
use std::{path::Path, time::Instant};

/// Per-frame masking telemetry from [`AmortizedMasker::mask_frame`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskFrameTelemetry {
    /// Elapsed wall-clock time for mask generation in milliseconds.
    pub mask_ms: f64,
    /// `true` when `Segformer` executed on this frame rather than warping cached sky state.
    pub segformer_refreshed: bool,
    /// Fraction of pixels retained as valid matchable terrain (`[0.0, 1.0]`).
    pub valid_fraction: f64,
    /// Number of pixels excluded by [`PropellerDetector`] on this frame.
    pub propeller_pixels: usize,
}

struct SegEngine {
    model: Segformer,
    preproc: PreprocessorConfig,
    airplane_indices: Vec<usize>,
    sky_label: u8,
    is_four_class: bool,
}

/// Amortized live frame masker combining `Segformer`, attitude sky warping, and [`PropellerDetector`].
pub struct AmortizedMasker {
    seg: Option<SegEngine>,
    device: Device,
    policy: MaskPolicy,
    margin_px: usize,
    refresh_interval: usize,
    propeller: PropellerDetector,
    key_dims: [u32; 2],
    key_airframe_keep: Vec<bool>,
    key_sky_keep: Vec<bool>,
    key_rot: Option<Matrix3<f64>>,
    frames_since_refresh: usize,
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
            let sky_label = cfg.class_index("sky").unwrap_or(2) as u8;
            let airplane_indices = cfg.class_index("airplane").into_iter().collect();
            let is_four_class = cfg.num_labels() == 4;
            Some(SegEngine {
                model,
                preproc,
                airplane_indices,
                sky_label,
                is_four_class,
            })
        } else {
            None
        };
        Ok(Self {
            seg,
            device: device.clone(),
            policy,
            margin_px,
            refresh_interval: refresh_interval.max(1),
            propeller: PropellerDetector::default(),
            key_dims: [0, 0],
            key_airframe_keep: Vec::new(),
            key_sky_keep: Vec::new(),
            key_rot: None,
            frames_since_refresh: 0,
        })
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
        let t0 = Instant::now();
        let (w, h) = rgb.dimensions();
        let total = (w as usize).saturating_mul(h as usize);
        if total == 0 {
            return Ok((
                GrayImage::new(w, h),
                MaskFrameTelemetry {
                    mask_ms: 0.0,
                    segformer_refreshed: false,
                    valid_fraction: 0.0,
                    propeller_pixels: 0,
                },
            ));
        }

        let prop_occluded = self.propeller.mask(rgb);
        let prop_pixels = prop_occluded.iter().filter(|&&b| b).count();
        let mut refreshed = false;

        let mut keep = if let Some(seg) = &self.seg {
            let need_refresh = self.key_dims != [w, h]
                || self.key_airframe_keep.len() != total
                || self.frames_since_refresh >= self.refresh_interval;
            if need_refresh {
                refreshed = true;
                let (air_keep, sky_keep) = self.eval_segformer_layers(seg, rgb)?;
                self.key_dims = [w, h];
                self.key_airframe_keep = air_keep;
                self.key_sky_keep = sky_keep;
                self.key_rot = cur_rot.copied();
                self.frames_since_refresh = 1;
            } else {
                self.frames_since_refresh += 1;
            }
            compose_amortized_keep(
                [w, h],
                &self.key_airframe_keep,
                &self.key_sky_keep,
                self.key_rot.as_ref(),
                cur_rot,
                k,
            )
        } else if let Some(fb) = fallback_mask {
            fb.as_raw().iter().take(total).map(|&v| v >= 128).collect()
        } else {
            vec![true; total]
        };

        for (slot, &blade) in keep.iter_mut().zip(&prop_occluded) {
            if blade {
                *slot = false;
            }
        }
        let valid_count = keep.iter().filter(|&&v| v).count();
        let mut out = GrayImage::new(w, h);
        for (idx, &ok) in keep.iter().enumerate() {
            let x = (idx % (w as usize)) as u32;
            let y = (idx / (w as usize)) as u32;
            out.put_pixel(x, y, Luma([if ok { 255 } else { 0 }]));
        }
        Ok((
            out,
            MaskFrameTelemetry {
                mask_ms: t0.elapsed().as_secs_f64() * 1000.0,
                segformer_refreshed: refreshed,
                valid_fraction: (valid_count as f64) / (total as f64),
                propeller_pixels: prop_pixels,
            },
        ))
    }

    /// Record the solved world-to-camera rotation of the most recent `Segformer` refresh frame.
    pub fn set_key_rotation(&mut self, rot: Matrix3<f64>) {
        self.key_rot = Some(rot);
    }

    /// Estimate `(down_deg, roll_deg)` from the cached `Segformer` sky mask if sky is visible.
    #[must_use]
    pub fn horizon_attitude(&self, k: &Matrix3<f64>) -> Option<(f64, f64)> {
        let [w, h] = self.key_dims;
        if self.key_sky_keep.len() != (w as usize) * (h as usize) || w < 8 || h < 8 {
            return None;
        }
        if self.key_sky_keep.iter().all(|&keep| keep) {
            return None;
        }
        let map: Vec<u8> = self
            .key_sky_keep
            .iter()
            .map(|&keep| u8::from(keep))
            .collect();
        crate::surface::horizon_attitude(&map, w, h, k)
    }

    fn eval_segformer_layers(
        &self,
        seg: &SegEngine,
        rgb: &RgbImage,
    ) -> Result<(Vec<bool>, Vec<bool>), PipelineError> {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let pred = seg.model.predict_image(
            rgb,
            &seg.preproc,
            ColorMode::Color,
            &seg.airplane_indices,
            &self.device,
        )?;
        if seg.is_four_class {
            let raw_sky: Vec<bool> = pred.labels.iter().map(|&l| l == seg.sky_label).collect();
            let raw_air: Vec<bool> = pred
                .labels
                .iter()
                .map(|&l| l != 0 && l != seg.sky_label)
                .collect();
            let sky_occ = dilate_rect(&raw_sky, w, h, self.margin_px);
            let air_occ = dilate_rect(&raw_air, w, h, self.margin_px);
            Ok((
                air_occ.into_iter().map(|b| !b).collect(),
                sky_occ.into_iter().map(|b| !b).collect(),
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
                .map(|(&l, &c)| l == seg.sky_label && c >= 0.35)
                .collect();
            let sky_occ = dilate_rect(&raw_sky, w, h, self.margin_px);
            let sky_keep: Vec<bool> = sky_occ.iter().map(|&b| !b).collect();
            let air_keep: Vec<bool> = combined_keep
                .iter()
                .zip(&sky_occ)
                .map(|(&keep_px, &is_sky)| keep_px || is_sky)
                .collect();
            Ok((air_keep, sky_keep))
        }
    }
}

fn compose_amortized_keep(
    [w, h]: [u32; 2],
    airframe_keep: &[bool],
    sky_keep: &[bool],
    key_rot: Option<&Matrix3<f64>>,
    cur_rot: Option<&Matrix3<f64>>,
    k: &Matrix3<f64>,
) -> Vec<bool> {
    let width = w as usize;
    let height = h as usize;
    let Some((r_key, r_cur)) = key_rot.zip(cur_rot) else {
        return airframe_keep
            .iter()
            .zip(sky_keep)
            .map(|(&a, &s)| a && s)
            .collect();
    };
    let delta_key_from_cur = r_key * r_cur.transpose();
    let ident_err = (delta_key_from_cur - Matrix3::identity()).norm();
    if ident_err < 1e-4 {
        return airframe_keep
            .iter()
            .zip(sky_keep)
            .map(|(&a, &s)| a && s)
            .collect();
    }

    let fx = k[(0, 0)].max(1e-6);
    let fy = k[(1, 1)].max(1e-6);
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let mut out = vec![false; width * height];
    for y in 0..height {
        let yc = (y as f64 - cy) / fy;
        for x in 0..width {
            let idx = y * width + x;
            if !airframe_keep[idx] {
                continue;
            }
            let ray_cur = Vector3::new((x as f64 - cx) / fx, yc, 1.0);
            let ray_key = delta_key_from_cur * ray_cur;
            if ray_key.z <= 1e-3 {
                continue;
            }
            let uk = (fx * (ray_key.x / ray_key.z) + cx)
                .round()
                .clamp(0.0, (width - 1) as f64) as usize;
            let vk = (fy * (ray_key.y / ray_key.z) + cy)
                .round()
                .clamp(0.0, (height - 1) as f64) as usize;
            out[idx] = sky_keep[vk * width + uk];
        }
    }
    out
}
