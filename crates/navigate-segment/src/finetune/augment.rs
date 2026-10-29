//! Training augmentations, inverse-sqrt class weights, and OneCycleLR schedule (`segment_finetune`).

use crate::config::{ColorMode, IGNORE_INDEX, PreprocessorConfig};
use image::RgbImage;

/// Deterministic 64-bit PRNG (`SplitMix64`) for reproducible training crops and jitter without
/// platform-specific RNG drift.
#[derive(Clone, Copy, Debug)]
pub struct Rng64(u64);

impl Rng64 {
    /// Seed a new deterministic generator.
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
    }

    /// Draw a `u64` uniform word.
    pub fn next_u64(&mut self) -> u64 {
        let mut z = self.0;
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Draw a uniform `f32` in `[0, 1)`.
    pub fn unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Draw a uniform `f32` in `[low, high)`.
    pub fn range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit_f32()
    }

    /// Shuffle `slice` in place using Fisher-Yates.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

/// Compute inverse-sqrt class weights clipped to `20.0` and normalized to unit mean.
pub fn class_weights(label_maps: &[&[u8]], num_classes: usize) -> Vec<f32> {
    let mut counts = vec![0u64; num_classes];
    for labels in label_maps {
        for &lbl in *labels {
            let idx = usize::from(lbl);
            if idx < num_classes {
                counts[idx] += 1;
            }
        }
    }
    let total: f64 = counts.iter().map(|&c| c as f64).sum::<f64>().max(1.0);
    let mut weights = Vec::with_capacity(num_classes);
    for &c in &counts {
        if c == 0 {
            weights.push(1.0_f32);
        } else {
            let freq = (c as f64) / total;
            let raw = (1.0 / freq.sqrt()).min(20.0) as f32;
            weights.push(raw);
        }
    }
    let mean = weights.iter().sum::<f32>() / (num_classes.max(1) as f32);
    if mean > 0.0 {
        for w in &mut weights {
            *w /= mean;
        }
    }
    weights
}

/// Evaluate OneCycleLR learning rate at `step` (`0..total_steps`) with linear warmup over
/// `pct_start` and cosine annealing down to `max_lr * 1e-4`.
pub fn one_cycle_lr(step: usize, total_steps: usize, max_lr: f64, pct_start: f64) -> f64 {
    if total_steps <= 1 {
        return max_lr;
    }
    let progress = (step as f64) / ((total_steps - 1) as f64);
    let warmup = pct_start.clamp(0.01, 0.5);
    let start_lr = max_lr / 25.0;
    let end_lr = max_lr * 1e-4;
    if progress < warmup {
        let t = progress / warmup;
        start_lr + t * (max_lr - start_lr)
    } else {
        let t = (progress - warmup) / (1.0 - warmup);
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
        end_lr + cosine * (max_lr - end_lr)
    }
}

/// Augment `(rgb, labels)` into a normalized `[3, size, size]` tensor and `[size, size]` labels.
///
/// Applies random scale (`0.6..=1.0`) and aspect (`0.75..=1.8`) crop, 50% horizontal flip, HSV
/// photometric jitter (`hue ±10°`, `sat 0.5..=1.5`, `val 0.6..=1.4`), and `color_mode` luminance
/// conversion.
pub fn augment_pair(
    rgb: &RgbImage,
    labels: &[u8],
    size: usize,
    color_mode: ColorMode,
    preprocessor: &PreprocessorConfig,
    train: bool,
    rng: &mut Rng64,
) -> (Vec<f32>, Vec<u8>) {
    let src_w = rgb.width() as usize;
    let src_h = rgb.height() as usize;
    let (crop_x, crop_y, crop_w, crop_h, flip) = if train {
        let scale = f64::from(rng.range_f32(0.6, 1.0));
        let aspect = f64::from(rng.range_f32(0.75, 1.8));
        let area = (src_w * src_h) as f64 * scale;
        let cw = ((area * aspect).sqrt().round() as usize).clamp(1, src_w);
        let ch = ((area / aspect).sqrt().round() as usize).clamp(1, src_h);
        let x0 = if src_w > cw {
            (rng.next_u64() as usize) % (src_w - cw + 1)
        } else {
            0
        };
        let y0 = if src_h > ch {
            (rng.next_u64() as usize) % (src_h - ch + 1)
        } else {
            0
        };
        (x0, y0, cw, ch, rng.unit_f32() < 0.5)
    } else {
        (0, 0, src_w, src_h, false)
    };

    let (hue_deg, sat_gain, val_gain) = if train {
        (
            rng.range_f32(-10.0, 10.0),
            rng.range_f32(0.5, 1.5),
            rng.range_f32(0.6, 1.4),
        )
    } else {
        (0.0, 1.0, 1.0)
    };
    let force_gray = match color_mode {
        ColorMode::Color => false,
        ColorMode::Gray => true,
        ColorMode::RandGray => train && rng.unit_f32() < 0.3,
    };

    let plane = size * size;
    let mut tensor = vec![0.0_f32; 3 * plane];
    let mut target = vec![IGNORE_INDEX; plane];

    for oy in 0..size {
        let fy = (oy as f64 + 0.5) * (crop_h as f64 / size as f64) - 0.5;
        let y0 = (crop_y as f64 + fy.floor().clamp(0.0, (crop_h - 1) as f64)) as usize;
        let y1 = (crop_y as f64 + (fy.floor() + 1.0).clamp(0.0, (crop_h - 1) as f64)) as usize;
        let ty = (fy - fy.floor()).clamp(0.0, 1.0) as f32;
        let ny = (crop_y + ((oy * crop_h + crop_h / 2) / size).min(crop_h - 1)).min(src_h - 1);

        for ox in 0..size {
            let col = if flip { size - 1 - ox } else { ox };
            let fx = (col as f64 + 0.5) * (crop_w as f64 / size as f64) - 0.5;
            let x0 = (crop_x as f64 + fx.floor().clamp(0.0, (crop_w - 1) as f64)) as usize;
            let x1 = (crop_x as f64 + (fx.floor() + 1.0).clamp(0.0, (crop_w - 1) as f64)) as usize;
            let tx = (fx - fx.floor()).clamp(0.0, 1.0) as f32;
            let nx = (crop_x + ((col * crop_w + crop_w / 2) / size).min(crop_w - 1)).min(src_w - 1);

            let p00 = rgb
                .get_pixel(x0.min(src_w - 1) as u32, y0.min(src_h - 1) as u32)
                .0;
            let p10 = rgb
                .get_pixel(x1.min(src_w - 1) as u32, y0.min(src_h - 1) as u32)
                .0;
            let p01 = rgb
                .get_pixel(x0.min(src_w - 1) as u32, y1.min(src_h - 1) as u32)
                .0;
            let p11 = rgb
                .get_pixel(x1.min(src_w - 1) as u32, y1.min(src_h - 1) as u32)
                .0;

            let mut ch = [0.0_f32; 3];
            for k in 0..3 {
                let top = f32::from(p00[k]) * (1.0 - tx) + f32::from(p10[k]) * tx;
                let bot = f32::from(p01[k]) * (1.0 - tx) + f32::from(p11[k]) * tx;
                ch[k] = ((top * (1.0 - ty) + bot * ty) / 255.0).clamp(0.0, 1.0);
            }
            if train {
                ch = jitter_hsv(ch, hue_deg, sat_gain, val_gain);
            }
            if force_gray {
                let luma = 0.299 * ch[0] + 0.587 * ch[1] + 0.114 * ch[2];
                ch = [luma, luma, luma];
            }

            let dst = oy * size + ox;
            tensor[dst] = (ch[0] - preprocessor.image_mean[0]) / preprocessor.image_std[0];
            tensor[plane + dst] = (ch[1] - preprocessor.image_mean[1]) / preprocessor.image_std[1];
            tensor[2 * plane + dst] =
                (ch[2] - preprocessor.image_mean[2]) / preprocessor.image_std[2];
            target[dst] = labels.get(ny * src_w + nx).copied().unwrap_or(IGNORE_INDEX);
        }
    }
    (tensor, target)
}

fn jitter_hsv(rgb: [f32; 3], hue_deg: f32, sat_gain: f32, val_gain: f32) -> [f32; 3] {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let delta = max - min;
    let mut h = if delta <= 1e-6 {
        0.0
    } else if (max - rgb[0]).abs() <= 1e-6 {
        60.0 * (((rgb[1] - rgb[2]) / delta).rem_euclid(6.0))
    } else if (max - rgb[1]).abs() <= 1e-6 {
        60.0 * (((rgb[2] - rgb[0]) / delta) + 2.0)
    } else {
        60.0 * (((rgb[0] - rgb[1]) / delta) + 4.0)
    };
    let mut s = if max <= 1e-6 { 0.0 } else { delta / max };
    let mut v = max;

    h = (h + hue_deg).rem_euclid(360.0);
    s = (s * sat_gain).clamp(0.0, 1.0);
    v = (v * val_gain).clamp(0.0, 1.0);

    let c = v * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    [
        (r1 + m).clamp(0.0, 1.0),
        (g1 + m).clamp(0.0, 1.0),
        (b1 + m).clamp(0.0, 1.0),
    ]
}
