//! Amortized live semantic occlusion masking, attitude sky warping, and propeller gating.

use crate::PipelineError;
use burn::tensor::Device;
use image::{GrayImage, RgbImage, imageops::colorops::grayscale};
use nalgebra::Matrix3;
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
    key_airframe_keep: Vec<u8>,
    key_sky_keep: Vec<u8>,
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
                    valid_fraction: 0.0,
                    propeller_pixels: 0,
                },
            ));
        }

        let prop_occluded = self.propeller.mask_luma(luma);
        let prop_pixels = prop_occluded.iter().filter(|&&b| b).count();
        let mut refreshed = false;

        let (raw_mask, valid_count) = if let Some(seg) = &self.seg {
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
            compose_amortized_u8(
                [w, h],
                &self.key_airframe_keep,
                &self.key_sky_keep,
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
        if self.key_sky_keep.iter().all(|&keep| keep != 0) {
            return None;
        }
        let map: Vec<u8> = self
            .key_sky_keep
            .iter()
            .map(|&keep| u8::from(keep != 0))
            .collect();
        crate::surface::horizon_attitude(&map, w, h, k)
    }

    fn eval_segformer_layers(
        &self,
        seg: &SegEngine,
        rgb: &RgbImage,
    ) -> Result<(Vec<u8>, Vec<u8>), PipelineError> {
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
                air_occ
                    .into_iter()
                    .map(|b| if b { 0 } else { 255 })
                    .collect(),
                sky_occ
                    .into_iter()
                    .map(|b| if b { 0 } else { 255 })
                    .collect(),
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

fn compose_fallback_u8(total: usize, fb_raw: &[u8], prop_occluded: &[bool]) -> (Vec<u8>, usize) {
    let mut out = vec![0u8; total];
    let mut valid = 0usize;
    for i in 0..total.min(fb_raw.len()).min(prop_occluded.len()) {
        if fb_raw[i] >= 128 && !prop_occluded[i] {
            out[i] = 255;
            valid += 1;
        }
    }
    (out, valid)
}

fn compose_unmasked_u8(total: usize, prop_occluded: &[bool]) -> (Vec<u8>, usize) {
    let mut out = vec![0u8; total];
    let mut valid = 0usize;
    for (dst, &blade) in out.iter_mut().zip(prop_occluded.iter().take(total)) {
        if !blade {
            *dst = 255;
            valid += 1;
        }
    }
    (out, valid)
}

fn compose_static_u8(
    airframe_keep: &[u8],
    sky_keep: &[u8],
    prop_occluded: &[bool],
) -> (Vec<u8>, usize) {
    let total = airframe_keep
        .len()
        .min(sky_keep.len())
        .min(prop_occluded.len());
    let mut out = vec![0u8; total];
    let mut valid = 0usize;
    for i in 0..total {
        let keep = if prop_occluded[i] {
            0
        } else {
            airframe_keep[i] & sky_keep[i]
        };
        out[i] = keep;
        valid += usize::from(keep != 0);
    }
    (out, valid)
}

fn compose_amortized_u8(
    [w, h]: [u32; 2],
    airframe_keep: &[u8],
    sky_keep: &[u8],
    prop_occluded: &[bool],
    key_rot: Option<&Matrix3<f64>>,
    cur_rot: Option<&Matrix3<f64>>,
    k: &Matrix3<f64>,
) -> (Vec<u8>, usize) {
    let width = w as usize;
    let height = h as usize;
    let total = width * height;
    let Some((r_key, r_cur)) = key_rot.zip(cur_rot) else {
        return compose_static_u8(airframe_keep, sky_keep, prop_occluded);
    };
    let delta_key_from_cur = r_key * r_cur.transpose();
    if (delta_key_from_cur - Matrix3::identity()).norm() < 1e-4 {
        return compose_static_u8(airframe_keep, sky_keep, prop_occluded);
    }

    let fx = k[(0, 0)].max(1e-6);
    let fy = k[(1, 1)].max(1e-6);
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let k_pix = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    let k_inv = Matrix3::new(
        1.0 / fx,
        0.0,
        -cx / fx,
        0.0,
        1.0 / fy,
        -cy / fy,
        0.0,
        0.0,
        1.0,
    );
    let hom = k_pix * delta_key_from_cur * k_inv;
    let mut out = vec![0u8; total];
    let workers = if total >= 120_000 {
        std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(1)
    } else {
        1
    };
    if workers <= 1 {
        let valid = warp_sky_rows(
            0,
            [width, height],
            &hom,
            airframe_keep,
            sky_keep,
            prop_occluded,
            &mut out,
        );
        return (out, valid);
    }

    let rows_per_band = height.div_ceil(workers).max(1);
    let valid_total = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut rem = out.as_mut_slice();
        let mut y0 = 0usize;
        while !rem.is_empty() && y0 < height {
            let band_rows = rows_per_band.min(height - y0);
            let (band_out, rest) = rem.split_at_mut(band_rows * width);
            rem = rest;
            let start_y = y0;
            y0 += band_rows;
            handles.push(scope.spawn(move || {
                warp_sky_rows(
                    start_y,
                    [width, height],
                    &hom,
                    airframe_keep,
                    sky_keep,
                    prop_occluded,
                    band_out,
                )
            }));
        }
        handles
            .into_iter()
            .map(|h_join| h_join.join().unwrap_or(0))
            .sum()
    });
    (out, valid_total)
}

fn warp_sky_rows(
    start_y: usize,
    [width, height]: [usize; 2],
    hom: &Matrix3<f64>,
    airframe_keep: &[u8],
    sky_keep: &[u8],
    prop_occluded: &[bool],
    band_out: &mut [u8],
) -> usize {
    let h00 = hom[(0, 0)];
    let h10 = hom[(1, 0)];
    let h20 = hom[(2, 0)];
    let max_u = (width.saturating_sub(1)) as f64;
    let max_v = (height.saturating_sub(1)) as f64;
    let band_rows = band_out.len() / width;
    let mut valid = 0usize;

    for local_y in 0..band_rows {
        let y = start_y + local_y;
        let yf = y as f64;
        let row_px = hom[(0, 1)] * yf + hom[(0, 2)];
        let row_py = hom[(1, 1)] * yf + hom[(1, 2)];
        let row_pz = hom[(2, 1)] * yf + hom[(2, 2)];
        let row_off = y * width;
        let air_row = &airframe_keep[row_off..row_off + width];
        let prop_row = &prop_occluded[row_off..row_off + width];
        let out_row = &mut band_out[local_y * width..(local_y + 1) * width];

        for x in 0..width {
            if air_row[x] == 0 || prop_row[x] {
                continue;
            }
            let xf = x as f64;
            let pz = row_pz + h20 * xf;
            if pz <= 1e-3 {
                continue;
            }
            let inv_z = 1.0 / pz;
            let uk = ((row_px + h00 * xf) * inv_z).round().clamp(0.0, max_u) as usize;
            let vk = ((row_py + h10 * xf) * inv_z).round().clamp(0.0, max_v) as usize;
            let keep = sky_keep[vk * width + uk];
            out_row[x] = keep;
            valid += usize::from(keep != 0);
        }
    }
    valid
}

#[cfg(test)]
mod tests {
    use super::compose_amortized_u8;
    use nalgebra::{Matrix3, Rotation3, Vector3};

    #[test]
    fn homography_parallel_warp_matches_ray_projection_at_800x600() {
        let (w, h) = (800u32, 600u32);
        let (width, height) = (w as usize, h as usize);
        let total = width * height;
        let air: Vec<u8> = (0..total)
            .map(|i| if (i % width) < 48 { 0 } else { 255 })
            .collect();
        let sky: Vec<u8> = (0..total)
            .map(|i| if (i / width) < 180 { 0 } else { 255 })
            .collect();
        let prop: Vec<bool> = (0..total).map(|i| (i % 97) == 0).collect();
        let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
        let r_key = Matrix3::identity();
        let r_cur = *Rotation3::from_euler_angles(0.03, -0.04, 0.05).matrix();

        let (fast, valid_fast) =
            compose_amortized_u8([w, h], &air, &sky, &prop, Some(&r_key), Some(&r_cur), &k);
        let delta = r_key * r_cur.transpose();
        let mut expected = vec![0u8; total];
        let mut valid_exp = 0usize;
        for y in 0..height {
            let yc = (y as f64 - 299.5) / 600.0;
            for x in 0..width {
                let idx = y * width + x;
                if air[idx] == 0 || prop[idx] {
                    continue;
                }
                let ray = delta * Vector3::new((x as f64 - 399.5) / 600.0, yc, 1.0);
                if ray.z <= 1e-3 {
                    continue;
                }
                let uk = (600.0 * (ray.x / ray.z) + 399.5).round().clamp(0.0, 799.0) as usize;
                let vk = (600.0 * (ray.y / ray.z) + 299.5).round().clamp(0.0, 599.0) as usize;
                let keep = sky[vk * width + uk];
                expected[idx] = keep;
                valid_exp += usize::from(keep != 0);
            }
        }
        assert_eq!(valid_fast, valid_exp);
        assert_eq!(fast, expected);
    }
}
