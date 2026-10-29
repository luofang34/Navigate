//! 4-class training label composition, speck filtering, split assignment, and mask propagation
//! (`airframe_labels`).

use crate::{
    SegmentError,
    config::{AIRFRAME, GROUND, PROPELLER, SKY},
    morphology::connected_components_8,
};
use image::{DynamicImage, GrayImage, Rgb, RgbImage, imageops::FilterType};
use sha2::{Digest, Sha256};

/// Prompted foreground class in [`parse_object`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectClass {
    /// Sky region (`SKY = 0`).
    Sky,
    /// Fixed airframe structure (`AIRFRAME = 2`).
    Airframe,
    /// Spinning propeller disc/blade (`PROPELLER = 3`).
    Propeller,
}

impl ObjectClass {
    /// Class index written into 4-class training label PNGs.
    pub fn label_id(self) -> u8 {
        match self {
            Self::Sky => SKY,
            Self::Airframe => AIRFRAME,
            Self::Propeller => PROPELLER,
        }
    }
}

/// Parse an object specification such as `"airframe"`, `"airframe_strut"`, `"propeller_2"`, or `"sky"`.
///
/// # Errors
///
/// Returns [`SegmentError::Invalid`] when `name` does not begin with a supported class prefix.
pub fn parse_object(name: &str) -> Result<ObjectClass, SegmentError> {
    let head = name.split(['_', '-']).next().unwrap_or(name);
    match head {
        "sky" => Ok(ObjectClass::Sky),
        "airframe" => Ok(ObjectClass::Airframe),
        "propeller" => Ok(ObjectClass::Propeller),
        _ => Err(SegmentError::Invalid(format!(
            "unsupported object label '{name}'; expected sky, airframe, or propeller"
        ))),
    }
}

/// Remove 8-connected components smaller than `width * height * min_area_fraction` (`0.001` default).
pub fn drop_specks(
    mask: &[bool],
    width: usize,
    height: usize,
    min_area_fraction: f64,
) -> Vec<bool> {
    let total = width * height;
    if total == 0 {
        return Vec::new();
    }
    let min_area = ((total as f64) * min_area_fraction).ceil().max(1.0) as usize;
    let (labels, stats) = connected_components_8(mask, width, height);
    let mut keep = vec![false; stats.len() + 1];
    for stat in &stats {
        if stat.area >= min_area {
            keep[stat.label as usize] = true;
        }
    }
    labels
        .into_iter()
        .map(|l| l != 0 && keep[l as usize])
        .collect()
}

/// Compose 4-class training labels (`GROUND = 1` default; `SKY = 0`, `AIRFRAME = 2`,
/// `PROPELLER = 3` overwrite in priority order so propeller wins over airframe and sky).
///
/// # Errors
///
/// Returns [`SegmentError::Shape`] when input masks do not all have length `len`.
pub fn compose_labels(
    sky: &[bool],
    airframe: &[bool],
    propeller: &[bool],
) -> Result<Vec<u8>, SegmentError> {
    let n = sky.len();
    if airframe.len() != n || propeller.len() != n {
        return Err(SegmentError::Shape {
            name: "compose_labels".into(),
            expected: vec![n],
            actual: vec![airframe.len(), propeller.len()],
        });
    }
    let mut labels = vec![GROUND; n];
    for i in 0..n {
        if sky[i] {
            labels[i] = SKY;
        }
        if airframe[i] {
            labels[i] = AIRFRAME;
        }
        if propeller[i] {
            labels[i] = PROPELLER;
        }
    }
    Ok(labels)
}

/// Deterministically assign a flight or video clip key to `"train"` or `"val"` via SHA-256.
///
/// `val_percent` is clamped to `1..=50` (typically `20` for an 80/20 flight-disjoint split).
pub fn split_name(group_key: &str, val_percent: u8) -> &'static str {
    let pct = u64::from(val_percent.clamp(1, 50));
    let mut hasher = Sha256::new();
    hasher.update(group_key.as_bytes());
    let digest = hasher.finalize();
    let bucket = u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ]) % 100;
    if bucket < pct { "val" } else { "train" }
}

/// Seed-prompted color-distance region segmentation for labeling static frames without Python SAM2.
pub fn segment_from_prompts(
    rgb: &RgbImage,
    positive_points: &[[usize; 2]],
    negative_points: &[[usize; 2]],
) -> Vec<bool> {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let n = w * h;
    if n == 0 || positive_points.is_empty() {
        return vec![false; n];
    }
    let sample = |pt: [usize; 2]| -> [f32; 3] {
        let p = rgb
            .get_pixel(pt[0].min(w - 1) as u32, pt[1].min(h - 1) as u32)
            .0;
        [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])]
    };
    let pos_colors: Vec<[f32; 3]> = positive_points.iter().copied().map(sample).collect();
    let neg_colors: Vec<[f32; 3]> = negative_points.iter().copied().map(sample).collect();

    let mut raw = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x as u32, y as u32).0;
            let c = [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])];
            let d_pos = pos_colors
                .iter()
                .map(|q| sq_dist(c, *q))
                .fold(f32::INFINITY, f32::min);
            let d_neg = neg_colors
                .iter()
                .map(|q| sq_dist(c, *q))
                .fold(42.0_f32 * 42.0_f32, f32::min);
            raw[y * w + x] = d_pos < d_neg && d_pos <= 55.0 * 55.0;
        }
    }

    let (comp_labels, stats) = connected_components_8(&raw, w, h);
    let mut seeded = vec![false; stats.len() + 1];
    for &[px, py] in positive_points {
        let lbl = comp_labels[py.min(h - 1) * w + px.min(w - 1)] as usize;
        if lbl != 0 {
            seeded[lbl] = true;
        }
    }
    let connected: Vec<bool> = comp_labels
        .into_iter()
        .map(|l| l != 0 && seeded[l as usize])
        .collect();
    drop_specks(&connected, w, h, 0.001)
}

/// Propagate a binary mask across consecutive video frames using coarse-to-fine block matching.
pub fn propagate_masks(frames: &[RgbImage], initial_mask: &[bool]) -> Vec<Vec<bool>> {
    if frames.is_empty() {
        return Vec::new();
    }
    let w = frames[0].width() as usize;
    let h = frames[0].height() as usize;
    let mut results = Vec::with_capacity(frames.len());
    results.push(initial_mask.to_vec());
    if frames.len() == 1 {
        return results;
    }

    let mut prev_gray: GrayImage = DynamicImage::ImageRgb8(frames[0].clone()).to_luma8();
    let mut prev_mask = initial_mask.to_vec();

    for frame in &frames[1..] {
        let cur_gray: GrayImage = DynamicImage::ImageRgb8(frame.clone()).to_luma8();
        let next_mask = warp_mask_block_sad(&prev_gray, &cur_gray, &prev_mask, w, h);
        let cleaned = drop_specks(&next_mask, w, h, 0.001);
        results.push(cleaned.clone());
        prev_gray = cur_gray;
        prev_mask = cleaned;
    }
    results
}

fn warp_mask_block_sad(
    prev: &GrayImage,
    cur: &GrayImage,
    prev_mask: &[bool],
    w: usize,
    h: usize,
) -> Vec<bool> {
    let cell = 16usize;
    let radius = 12isize;
    let gw = w.div_ceil(cell);
    let gh = h.div_ceil(cell);
    let p_raw = prev.as_raw();
    let c_raw = cur.as_raw();

    let mut flow = vec![[0isize; 2]; gw * gh];
    for gy in 0..gh {
        let cy = (gy * cell + cell / 2).min(h - 1) as isize;
        for gx in 0..gw {
            let cx = (gx * cell + cell / 2).min(w - 1) as isize;
            let mut best_cost = u32::MAX;
            let mut best = [0isize; 2];
            for dy in (-radius..=radius).step_by(2) {
                for dx in (-radius..=radius).step_by(2) {
                    let mut cost = 0u32;
                    for py in (-4isize..=4).step_by(2) {
                        for px in (-4isize..=4).step_by(2) {
                            let x0 = (cx + px - dx).clamp(0, w as isize - 1) as usize;
                            let y0 = (cy + py - dy).clamp(0, h as isize - 1) as usize;
                            let x1 = (cx + px).clamp(0, w as isize - 1) as usize;
                            let y1 = (cy + py).clamp(0, h as isize - 1) as usize;
                            cost += u32::from(p_raw[y0 * w + x0].abs_diff(c_raw[y1 * w + x1]));
                        }
                    }
                    if cost < best_cost {
                        best_cost = cost;
                        best = [dx, dy];
                    }
                }
            }
            flow[gy * gw + gx] = best;
        }
    }

    let mut out = vec![false; w * h];
    for y in 0..h {
        let gy = (y / cell).min(gh - 1);
        for x in 0..w {
            let gx = (x / cell).min(gw - 1);
            let [dx, dy] = flow[gy * gw + gx];
            let sx = (x as isize - dx).clamp(0, w as isize - 1) as usize;
            let sy = (y as isize - dy).clamp(0, h as isize - 1) as usize;
            out[y * w + x] = prev_mask[sy * w + sx];
        }
    }
    out
}

/// Render a diagnostic contact sheet overlaying 4-class labels onto `samples`.
pub fn contact_sheet(
    samples: &[(RgbImage, Vec<u8>)],
    thumb_width: u32,
    columns: usize,
) -> RgbImage {
    if samples.is_empty() || columns == 0 {
        return RgbImage::new(1, 1);
    }
    let cols = columns.min(samples.len());
    let rows = samples.len().div_ceil(cols);
    let aspect = samples[0].0.height().max(1) as f64 / samples[0].0.width().max(1) as f64;
    let tw = thumb_width.max(32);
    let th = ((f64::from(tw) * aspect).round() as u32).max(24);
    let mut sheet = RgbImage::new(tw * cols as u32, th * rows as u32);

    for (idx, (rgb, labels)) in samples.iter().enumerate() {
        let w = rgb.width() as usize;
        let h = rgb.height() as usize;
        let mut tinted = rgb.clone();
        for y in 0..h {
            for x in 0..w {
                let lbl = labels.get(y * w + x).copied().unwrap_or(GROUND);
                let overlay = match lbl {
                    SKY => Some([70u8, 150, 255]),
                    AIRFRAME => Some([255u8, 70, 70]),
                    PROPELLER => Some([255u8, 210, 40]),
                    _ => None,
                };
                if let Some(ov) = overlay {
                    let p = tinted.get_pixel(x as u32, y as u32).0;
                    tinted.put_pixel(
                        x as u32,
                        y as u32,
                        Rgb([
                            ((u16::from(p[0]) + u16::from(ov[0])) / 2) as u8,
                            ((u16::from(p[1]) + u16::from(ov[1])) / 2) as u8,
                            ((u16::from(p[2]) + u16::from(ov[2])) / 2) as u8,
                        ]),
                    );
                }
            }
        }
        let thumb = image::imageops::resize(&tinted, tw, th, FilterType::Triangle);
        let ox = (idx % cols) as u32 * tw;
        let oy = (idx / cols) as u32 * th;
        for y in 0..th {
            for x in 0..tw {
                sheet.put_pixel(ox + x, oy + y, *thumb.get_pixel(x, y));
            }
        }
    }
    sheet
}

fn sq_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}
