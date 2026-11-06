//! Multi-yaw nadir landmark synthesis, exact quadrant rotation, and orientation gating.

use crate::{error::FlightVisualError, runtime::LoadedXFeat, surface::Surface};
use image::{GrayImage, Luma};
use nalgebra::Matrix3;
use navigate_match::SparseFeatures;
use std::sync::OnceLock;

/// 6 canonical nadir yaw band heading angles in degrees.
pub const NADIR_YAW_BANDS_DEG: [f64; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];

/// 12 dense nadir yaw band heading angles in degrees (`30` degree step).
pub const DENSE_NADIR_YAW_BANDS_DEG: [f64; 12] = [
    0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
];

/// Minimum direction cosine threshold for nadir yaw alignment (`cos(36 deg) = 0.809 >= 0.80`).
pub const MIN_COS_ORIENTATION: f32 = 0.80;

/// Canonical nadir pitch angle in degrees.
pub const NADIR_DOWN_DEG: f64 = 89.0;

/// Return true when a rotation matrix represents a near-nadir camera pose.
#[must_use]
pub fn is_nadir_pose(rot: &Matrix3<f64>) -> bool {
    -rot[(2, 2)] > 0.95
}

/// Return the minimum cosine orientation threshold for a given camera rotation.
#[must_use]
pub fn min_cos_orientation_for_pose(rot: &Matrix3<f64>) -> f32 {
    if is_nadir_pose(rot) {
        MIN_COS_ORIENTATION
    } else {
        0.90
    }
}

/// Return the 6 canonical nadir yaw band headings in degrees.
#[must_use]
pub const fn synthesize_nadir_yaw_bands() -> [f64; 6] {
    NADIR_YAW_BANDS_DEG
}

/// Match a query heading in degrees against the nearest synthesized nadir yaw band.
#[must_use]
pub fn match_nadir_yaw_band(query_heading_deg: f64) -> Option<(usize, f64)> {
    let q_norm = query_heading_deg.rem_euclid(360.0);
    let q_rad = q_norm.to_radians();
    let mut best_idx = 0;
    let mut best_cos = -1.0;

    for (i, &band_deg) in NADIR_YAW_BANDS_DEG.iter().enumerate() {
        let b_rad = band_deg.to_radians();
        let cos_val = (q_rad - b_rad).cos();
        if cos_val > best_cos {
            best_cos = cos_val;
            best_idx = i;
        }
    }

    (best_cos >= f64::from(MIN_COS_ORIENTATION)).then_some((best_idx, best_cos))
}

/// Rotate a grayscale image around its center, using exact integer indexing at `0/90/180/270 deg`.
#[must_use]
pub fn rotate_image_gray(src: &GrayImage, angle_deg: f64) -> GrayImage {
    let (width, height) = src.dimensions();
    if width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }
    let norm = angle_deg.rem_euclid(360.0);
    if norm <= 1e-9 || (360.0 - norm) <= 1e-9 {
        return src.clone();
    }
    if (norm - 180.0).abs() <= 1e-9 {
        return rotate_180_exact(src, width as usize, height as usize);
    }
    if (width + height).is_multiple_of(2) {
        if (norm - 90.0).abs() <= 1e-9 {
            return rotate_90_exact(src, width as usize, height as usize);
        }
        if (norm - 270.0).abs() <= 1e-9 {
            return rotate_270_exact(src, width as usize, height as usize);
        }
    }
    rotate_image_gray_bilinear(src, angle_deg)
}

fn rotate_180_exact(src: &GrayImage, w: usize, h: usize) -> GrayImage {
    let mut dst = GrayImage::new(w as u32, h as u32);
    let src_buf = src.as_raw();
    let dst_buf = dst.as_mut();
    for y in 0..h {
        let src_row = &src_buf[(h - 1 - y) * w..(h - y) * w];
        let dst_row = &mut dst_buf[y * w..(y + 1) * w];
        for x in 0..w {
            dst_row[x] = src_row[w - 1 - x];
        }
    }
    dst
}

fn rotate_90_exact(src: &GrayImage, w: usize, h: usize) -> GrayImage {
    let mut dst = GrayImage::new(w as u32, h as u32);
    let src_buf = src.as_raw();
    let dst_buf = dst.as_mut();
    let sum_half = ((w + h) / 2) as isize - 1;
    let diff_half = ((w as isize) - (h as isize)) / 2;
    let w_i = w as isize;
    let h_i = h as isize;
    for y in 0..h {
        let sx = sum_half - (y as isize);
        if sx < 0 || sx >= w_i {
            continue;
        }
        let sx_u = sx as usize;
        let dst_row = &mut dst_buf[y * w..(y + 1) * w];
        for (x, out_px) in dst_row.iter_mut().enumerate() {
            let sy = (x as isize) - diff_half;
            if sy >= 0 && sy < h_i {
                *out_px = src_buf[(sy as usize) * w + sx_u];
            }
        }
    }
    dst
}

fn rotate_270_exact(src: &GrayImage, w: usize, h: usize) -> GrayImage {
    let mut dst = GrayImage::new(w as u32, h as u32);
    let src_buf = src.as_raw();
    let dst_buf = dst.as_mut();
    let sum_half = ((w + h) / 2) as isize - 1;
    let diff_half = ((w as isize) - (h as isize)) / 2;
    let w_i = w as isize;
    let h_i = h as isize;
    for y in 0..h {
        let sx = (y as isize) + diff_half;
        if sx < 0 || sx >= w_i {
            continue;
        }
        let sx_u = sx as usize;
        let dst_row = &mut dst_buf[y * w..(y + 1) * w];
        for (x, out_px) in dst_row.iter_mut().enumerate() {
            let sy = sum_half - (x as isize);
            if sy >= 0 && sy < h_i {
                *out_px = src_buf[(sy as usize) * w + sx_u];
            }
        }
    }
    dst
}

pub(crate) fn rotate_image_gray_bilinear(src: &GrayImage, angle_deg: f64) -> GrayImage {
    let (width, height) = src.dimensions();
    let mut dst = GrayImage::new(width, height);
    if width == 0 || height == 0 {
        return dst;
    }
    let rad = angle_deg.to_radians();
    let (sin_t, cos_t) = rad.sin_cos();
    let cx = f64::from(width) * 0.5 - 0.5;
    let cy = f64::from(height) * 0.5 - 0.5;
    let max_x = f64::from(width - 1);
    let max_y = f64::from(height - 1);
    let w_usize = width as usize;
    let src_buf = src.as_raw();
    let dst_buf = dst.as_mut();
    const EPS: f64 = 1e-6;

    for y in 0..height {
        let dy = f64::from(y) - cy;
        let row_off = (y as usize) * w_usize;
        for x in 0..width {
            let dx = f64::from(x) - cx;
            let sx = cx + dx * cos_t - dy * sin_t;
            let sy = cy + dx * sin_t + dy * cos_t;
            if sx >= -EPS && sx <= max_x + EPS && sy >= -EPS && sy <= max_y + EPS {
                let sx_c = sx.clamp(0.0, max_x);
                let sy_c = sy.clamp(0.0, max_y);
                let x0 = (sx_c.floor() as usize).min(w_usize - 1);
                let y0 = (sy_c.floor() as usize).min((height - 1) as usize);
                let x1 = (x0 + 1).min(w_usize - 1);
                let y1 = (y0 + 1).min((height - 1) as usize);
                let fx = sx_c - (x0 as f64);
                let fy = sy_c - (y0 as f64);
                let p00 = f64::from(src_buf[y0 * w_usize + x0]);
                let p10 = f64::from(src_buf[y0 * w_usize + x1]);
                let p01 = f64::from(src_buf[y1 * w_usize + x0]);
                let p11 = f64::from(src_buf[y1 * w_usize + x1]);
                let val = (1.0 - fx) * (1.0 - fy) * p00
                    + fx * (1.0 - fy) * p10
                    + (1.0 - fx) * fy * p01
                    + fx * fy * p11;
                dst_buf[row_off + (x as usize)] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// Map a keypoint coordinate in a rotated image back to source image coordinates.
#[must_use]
pub fn unrotate_pixel(pixel: [f64; 2], angle_deg: f64, width: u32, height: u32) -> [f64; 2] {
    let rad = angle_deg.to_radians();
    let (sin_t, cos_t) = rad.sin_cos();
    let cx = f64::from(width) * 0.5 - 0.5;
    let cy = f64::from(height) * 0.5 - 0.5;
    let dx = pixel[0] - cx;
    let dy = pixel[1] - cy;
    [cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t]
}

/// Build a valid-pixel mask (`255` valid, `0` masked) for [`rotate_image_gray`], eroded from borders.
#[must_use]
pub fn rotated_valid_mask(
    width: u32,
    height: u32,
    angle_deg: f64,
    border_erode_px: f64,
) -> GrayImage {
    let mut mask = GrayImage::new(width, height);
    if width == 0 || height == 0 {
        return mask;
    }
    let erode = border_erode_px.max(0.0);
    let min_x = erode;
    let max_x = f64::from(width - 1) - erode;
    let min_y = erode;
    let max_y = f64::from(height - 1) - erode;
    if min_x > max_x || min_y > max_y {
        return mask;
    }
    for y in 0..height {
        for x in 0..width {
            let [sx, sy] = unrotate_pixel([f64::from(x), f64::from(y)], angle_deg, width, height);
            if sx >= min_x && sx <= max_x && sy >= min_y && sy <= max_y {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
    }
    mask
}

/// Synthesized rotated nadir band view.
#[derive(Clone, Debug)]
pub struct SynthesizedNadirBand {
    /// Heading angle of this yaw band in degrees.
    pub heading_deg: f64,
    /// Camera x-axis unit vector in `ENU` coordinates for this band.
    pub cam_x_enu: [f32; 3],
    /// Extracted feature keypoints and descriptors in rotated image space.
    pub features: SparseFeatures,
    /// Keypoint coordinates mapped back to base image space.
    pub unrotated_pixels: Vec<[f64; 2]>,
}

pub(crate) fn cached_800x600_masks(dense_30deg: bool) -> &'static [GrayImage] {
    static DENSE_MASKS: OnceLock<Vec<GrayImage>> = OnceLock::new();
    static COARSE_MASKS: OnceLock<Vec<GrayImage>> = OnceLock::new();
    if dense_30deg {
        DENSE_MASKS.get_or_init(|| {
            DENSE_NADIR_YAW_BANDS_DEG[1..]
                .iter()
                .map(|&angle| rotated_valid_mask(800, 600, angle, 12.0))
                .collect()
        })
    } else {
        COARSE_MASKS.get_or_init(|| {
            NADIR_YAW_BANDS_DEG[1..]
                .iter()
                .map(|&angle| rotated_valid_mask(800, 600, angle, 12.0))
                .collect()
        })
    }
}

/// Synthesize rotated nadir yaw bands (`30` or `60` degree spacing) from a base nadir image.
///
/// # Errors
///
/// Returns [`FlightVisualError`] if feature extraction fails.
pub fn synthesize_rotated_bands(
    base_gray: &GrayImage,
    surface: &Surface,
    clat: f64,
    clon: f64,
    xfeat: &LoadedXFeat,
    kp_limit: usize,
    dense_30deg: bool,
) -> Result<Vec<SynthesizedNadirBand>, FlightVisualError> {
    let angles_deg: &[f64] = if dense_30deg {
        &DENSE_NADIR_YAW_BANDS_DEG[1..]
    } else {
        &NADIR_YAW_BANDS_DEG[1..]
    };
    let (width, height) = base_gray.dimensions();
    let rotated_views: Vec<GrayImage> = angles_deg
        .iter()
        .map(|&angle| rotate_image_gray(base_gray, angle))
        .collect();
    let owned_masks: Vec<GrayImage>;
    let mask_slice: &[GrayImage] = if (width, height) == (800, 600) {
        cached_800x600_masks(dense_30deg)
    } else {
        owned_masks = angles_deg
            .iter()
            .map(|&angle| rotated_valid_mask(width, height, angle, 12.0))
            .collect();
        &owned_masks
    };
    let pairs: Vec<(&GrayImage, Option<&GrayImage>)> = rotated_views
        .iter()
        .zip(mask_slice.iter())
        .map(|(img, m)| (img, Some(m)))
        .collect();
    let feats = xfeat.extract_batch_masked(&pairs, kp_limit)?;
    let mut out = Vec::with_capacity(angles_deg.len());

    for (&angle, feat) in angles_deg.iter().zip(feats) {
        let unrotated: Vec<[f64; 2]> = feat
            .pixels
            .iter()
            .map(|&p| unrotate_pixel(p, angle, width, height))
            .collect();
        let rot_angle = surface.camera_rotation(clat, clon, angle, NADIR_DOWN_DEG, 0.0);
        let cam_x = [
            rot_angle[(0, 0)] as f32,
            rot_angle[(0, 1)] as f32,
            rot_angle[(0, 2)] as f32,
        ];
        out.push(SynthesizedNadirBand {
            heading_deg: angle,
            cam_x_enu: cam_x,
            features: feat,
            unrotated_pixels: unrotated,
        });
    }
    Ok(out)
}
