//! Multi-yaw nadir landmark synthesis and orientation gating.

use crate::{PipelineError, surface::Surface, xfeat_runtime::LoadedXFeat};
use image::{GrayImage, Luma};
use nalgebra::Matrix3;
use navigate_match::SparseFeatures;

/// 6 canonical nadir yaw band heading angles in degrees.
pub const NADIR_YAW_BANDS_DEG: [f64; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];

/// Minimum direction cosine threshold for nadir yaw alignment (cos(36 deg) = 0.809 >= 0.80).
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
pub fn synthesize_nadir_yaw_bands() -> [f64; 6] {
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

    if best_cos >= f64::from(MIN_COS_ORIENTATION) {
        Some((best_idx, best_cos))
    } else {
        None
    }
}

/// Rotate a grayscale image around its center using bilinear interpolation.
#[must_use]
pub fn rotate_image_gray(src: &GrayImage, angle_deg: f64) -> GrayImage {
    let (width, height) = src.dimensions();
    let mut dst = GrayImage::new(width, height);
    if width == 0 || height == 0 {
        return dst;
    }
    let rad = angle_deg.to_radians();
    let cos_t = rad.cos();
    let sin_t = rad.sin();
    let cx = f64::from(width) * 0.5 - 0.5;
    let cy = f64::from(height) * 0.5 - 0.5;

    for y in 0..height {
        let dy = f64::from(y) - cy;
        for x in 0..width {
            let dx = f64::from(x) - cx;
            let sx = cx + dx * cos_t - dy * sin_t;
            let sy = cy + dx * sin_t + dy * cos_t;
            let max_x = f64::from(width - 1);
            let max_y = f64::from(height - 1);
            const EPS: f64 = 1e-6;
            if sx >= -EPS && sx <= max_x + EPS && sy >= -EPS && sy <= max_y + EPS {
                let sx_c = sx.clamp(0.0, max_x);
                let sy_c = sy.clamp(0.0, max_y);
                let x0 = (sx_c.floor() as u32).min(width - 1);
                let y0 = (sy_c.floor() as u32).min(height - 1);
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);
                let fx = sx_c - f64::from(x0);
                let fy = sy_c - f64::from(y0);
                let p00 = f64::from(src.get_pixel(x0, y0).0[0]);
                let p10 = f64::from(src.get_pixel(x1, y0).0[0]);
                let p01 = f64::from(src.get_pixel(x0, y1).0[0]);
                let p11 = f64::from(src.get_pixel(x1, y1).0[0]);
                let val = (1.0 - fx) * (1.0 - fy) * p00
                    + fx * (1.0 - fy) * p10
                    + (1.0 - fx) * fy * p01
                    + fx * fy * p11;
                dst.put_pixel(x, y, Luma([val.round().clamp(0.0, 255.0) as u8]));
            }
        }
    }
    dst
}

/// Map a keypoint coordinate in a rotated image back to source image coordinates.
#[must_use]
pub fn unrotate_pixel(pixel: [f64; 2], angle_deg: f64, width: u32, height: u32) -> [f64; 2] {
    let rad = angle_deg.to_radians();
    let cos_t = rad.cos();
    let sin_t = rad.sin();
    let cx = f64::from(width) * 0.5 - 0.5;
    let cy = f64::from(height) * 0.5 - 0.5;
    let dx = pixel[0] - cx;
    let dy = pixel[1] - cy;
    let sx = cx + dx * cos_t - dy * sin_t;
    let sy = cy + dx * sin_t + dy * cos_t;
    [sx, sy]
}

/// Synthesized rotated nadir band view.
#[derive(Clone, Debug)]
pub struct SynthesizedNadirBand {
    /// Heading angle of this yaw band in degrees.
    pub heading_deg: f64,
    /// Camera x-axis unit vector in ENU coordinates for this band.
    pub cam_x_enu: [f32; 3],
    /// Extracted feature keypoints and descriptors in rotated image space.
    pub features: SparseFeatures,
    /// Keypoint coordinates mapped back to base image space.
    pub unrotated_pixels: Vec<[f64; 2]>,
}

/// Synthesize 5 rotated nadir yaw bands from a base nadir image.
///
/// # Errors
///
/// Returns [`PipelineError`] if feature extraction fails.
pub(crate) fn synthesize_rotated_bands(
    base_gray: &GrayImage,
    surface: &Surface,
    clat: f64,
    clon: f64,
    xfeat: &LoadedXFeat,
    kp_limit: usize,
) -> Result<Vec<SynthesizedNadirBand>, PipelineError> {
    let angles_deg = [60.0_f64, 120.0, 180.0, 240.0, 300.0];
    let (width, height) = base_gray.dimensions();
    let mut out = Vec::with_capacity(angles_deg.len());

    for &angle in &angles_deg {
        let rotated = rotate_image_gray(base_gray, angle);
        let feat = xfeat.extract(&rotated, kp_limit)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_yaw_bands_has_six_intervals() {
        let bands = synthesize_nadir_yaw_bands();
        assert_eq!(bands.len(), 6);
        assert_eq!(bands, [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]);
    }

    #[test]
    fn match_yaw_band_selects_closest_heading() {
        let (idx, cos_val) = match_nadir_yaw_band(246.6).expect("matches 240 deg band");
        assert_eq!(idx, 4);
        assert!(cos_val >= 0.80);
    }

    #[test]
    fn unrotate_pixel_inverts_rotate_transform() {
        let (w, h) = (800_u32, 600_u32);
        let original = [400.0, 300.0];
        let unrot = unrotate_pixel(original, 60.0, w, h);
        let (cx, cy) = (f64::from(w) * 0.5 - 0.5, f64::from(h) * 0.5 - 0.5);
        assert!((unrot[0] - cx).abs() < 1.0);
        assert!((unrot[1] - cy).abs() < 1.0);
    }

    #[test]
    fn rotate_image_gray_preserves_full_perimeter_at_zero_degrees() {
        let (w, h) = (20_u32, 20_u32);
        let mut src = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                src.put_pixel(x, y, Luma([(x * 11 + y * 7 + 10) as u8]));
            }
        }
        let rot0 = rotate_image_gray(&src, 0.0);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    rot0.get_pixel(x, y),
                    src.get_pixel(x, y),
                    "zero-degree rotation must match source at perimeter pixel ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn rotate_image_gray_perimeter_at_180_and_360_has_no_gutter() {
        let (w, h) = (40_u32, 30_u32);
        let src = GrayImage::from_pixel(w, h, Luma([200]));
        let rot180 = rotate_image_gray(&src, 180.0);
        let rot360 = rotate_image_gray(&src, 360.0);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    rot180.get_pixel(x, y).0[0],
                    200,
                    "180 deg has no gutter at ({x}, {y})"
                );
                assert_eq!(
                    rot360.get_pixel(x, y).0[0],
                    200,
                    "360 deg has no gutter at ({x}, {y})"
                );
            }
        }
    }
}
