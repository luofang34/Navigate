//! Multi-yaw nadir synthesis, horizon RANSAC, focal length sweep, and milliradian scaling.

use image::{GrayImage, Luma};
use nalgebra::{Matrix3, SMatrix};
use navigate_visual::{EngineKind, HostTensor, ModelInference, NamedTensor};

/// Standard grazing pitch angle seeds in degrees.
pub const GRAZING_PITCH_SEEDS_DEG: [f64; 6] = [-10.0, 8.0, 22.0, 38.0, 58.0, 88.0];

/// 6 canonical nadir yaw band heading angles in degrees.
pub const NADIR_YAW_BANDS_DEG: [f64; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];

/// Minimum direction cosine threshold for nadir yaw alignment (cos(36 deg) = 0.809 >= 0.80).
pub const MIN_COS_ORIENTATION: f64 = 0.80;

/// Returns the 6 synthesized nadir yaw band headings in degrees.
#[must_use]
pub fn synthesize_nadir_yaw_bands() -> [f64; 6] {
    NADIR_YAW_BANDS_DEG
}

/// Computes direction cosine between two heading angles in radians.
#[must_use]
pub fn direction_cosine(h1_rad: f64, h2_rad: f64) -> f64 {
    (h1_rad - h2_rad).cos()
}

/// Matches a query heading in degrees against the nearest synthesized yaw band.
#[must_use]
pub fn match_nadir_yaw_band(query_heading_deg: f64) -> Option<(usize, f64)> {
    let q_norm = query_heading_deg.rem_euclid(360.0);
    let q_rad = q_norm.to_radians();
    let mut best_idx = 0;
    let mut best_cos = -1.0;

    for (i, &band_deg) in NADIR_YAW_BANDS_DEG.iter().enumerate() {
        let b_rad = band_deg.to_radians();
        let cos_val = direction_cosine(q_rad, b_rad);
        if cos_val > best_cos {
            best_cos = cos_val;
            best_idx = i;
        }
    }

    if best_cos >= MIN_COS_ORIENTATION {
        Some((best_idx, best_cos))
    } else {
        None
    }
}

/// Returns true when a camera rotation matrix represents a near-nadir orientation.
#[must_use]
pub fn is_nadir_pose(rot: &Matrix3<f64>) -> bool {
    -rot[(2, 2)] > 0.95
}

/// Returns the minimum orientation cosine threshold for a given camera rotation.
#[must_use]
pub fn min_cos_orientation_for_pose(rot: &Matrix3<f64>) -> f64 {
    if is_nadir_pose(rot) { 0.80 } else { 0.90 }
}

/// Rotates a grayscale image around its center using bilinear interpolation.
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
    let max_x = f64::from(width - 1);
    let max_y = f64::from(height - 1);
    const EPS: f64 = 1e-6;

    for y in 0..height {
        let dy = f64::from(y) - cy;
        for x in 0..width {
            let dx = f64::from(x) - cx;
            let sx = cx + dx * cos_t - dy * sin_t;
            let sy = cy + dx * sin_t + dy * cos_t;
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

/// Inverts coordinate transformation from rotated image back to source image coordinates.
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

/// Generates pitch and roll seeds adhering to low-AGL nadir prioritization rules.
#[must_use]
pub fn generate_corridor_seeds(
    agl_m: f64,
    attitude: Option<(f64, f64)>,
    headings: &[f64],
) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let prioritize_nadir = attitude.is_none() && agl_m <= 250.0;
    let pitches = match attitude {
        Some(att) => vec![att],
        None => {
            let mut list = Vec::new();
            for &p in &GRAZING_PITCH_SEEDS_DEG {
                list.push((p, 0.0));
            }
            for &r in &[25.0, -25.0] {
                for &p in &GRAZING_PITCH_SEEDS_DEG {
                    list.push((p, r));
                }
            }
            list
        }
    };

    if prioritize_nadir {
        for _ in headings {
            for &down_deg in &[88.0, 75.0] {
                out.push((down_deg, 0.0));
            }
        }
        for &(down_deg, roll_deg) in &pitches {
            if (down_deg - 88.0).abs() < 1e-3 && roll_deg.abs() < 1e-3 {
                continue;
            }
            for _ in headings {
                out.push((down_deg, roll_deg));
            }
        }
    } else {
        for &(down_deg, roll_deg) in &pitches {
            for _ in headings {
                out.push((down_deg, roll_deg));
            }
        }
    }
    out
}

/// Mock inference session for testing dynamic input tensor name discovery.
pub struct DynamicMockSession {
    /// Expected primary input tensor name.
    pub expected_name: String,
    /// Count of executed forward inferences.
    pub forward_count: usize,
}

impl DynamicMockSession {
    /// Creates a mock session expecting the given primary input tensor name.
    #[must_use]
    pub fn new(expected_name: &str) -> Self {
        Self {
            expected_name: expected_name.to_owned(),
            forward_count: 0,
        }
    }
}

impl ModelInference for DynamicMockSession {
    type Tensor = HostTensor;
    type Error = String;

    fn engine_kind(&self) -> EngineKind {
        EngineKind::Ort
    }

    fn forward(
        &mut self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, String> {
        let primary = inputs
            .first()
            .ok_or_else(|| "missing primary input tensor".to_string())?;
        if primary.name != self.expected_name {
            return Err(format!(
                "input name mismatch: expected '{}', got '{}'",
                self.expected_name, primary.name
            ));
        }
        self.forward_count += 1;
        const CELLS: usize = 72 * 100;
        let descriptors = HostTensor::new(vec![1, 64, 72, 100], vec![0.05_f32; 64 * CELLS])
            .map_err(|e| format!("{e:?}"))?;
        let keypoint_logits = HostTensor::new(vec![1, 65, 72, 100], vec![0.02_f32; 65 * CELLS])
            .map_err(|e| format!("{e:?}"))?;
        let reliability = HostTensor::new(vec![1, 1, 72, 100], vec![0.9_f32; CELLS])
            .map_err(|e| format!("{e:?}"))?;

        Ok(vec![
            NamedTensor::new("descriptors", descriptors),
            NamedTensor::new("keypoint_logits", keypoint_logits),
            NamedTensor::new("reliability", reliability),
        ])
    }
}

/// Computes the condition number (ratio of max to min eigenvalue) of a 6x6 symmetric matrix.
#[must_use]
pub fn condition_number_6x6(m: &SMatrix<f64, 6, 6>) -> f64 {
    let eig = m.symmetric_eigen().eigenvalues;
    let min_e = eig.min();
    let max_e = eig.max();
    if min_e <= 1e-15 {
        f64::INFINITY
    } else {
        max_e / min_e
    }
}

/// Robust RANSAC horizon line estimator.
#[derive(Debug, Clone)]
pub struct RansacHorizonEstimator {
    /// Inlier distance threshold in pixels.
    pub inlier_threshold_px: f64,
    /// Maximum RANSAC iterations.
    pub max_iterations: usize,
}

impl Default for RansacHorizonEstimator {
    fn default() -> Self {
        Self {
            inlier_threshold_px: 3.0,
            max_iterations: 100,
        }
    }
}

impl RansacHorizonEstimator {
    /// Fits a horizon line (y = m * x + c) rejecting outliers and extracts pitch and roll.
    #[must_use]
    pub fn fit_horizon(
        &self,
        sky_ground_transitions: &[[f64; 2]],
        image_size: [u32; 2],
        focal_px: f64,
    ) -> Option<(f64, f64)> {
        if sky_ground_transitions.len() < 4 {
            return None;
        }
        let mid_y = image_size[1] as f64 * 0.5;
        let mut best_inliers = 0;
        let mut best_line = (0.0, mid_y);

        for i in 0..sky_ground_transitions.len() {
            let j = (i + 1) % sky_ground_transitions.len();
            let p1 = sky_ground_transitions[i];
            let p2 = sky_ground_transitions[j];
            let dx = p2[0] - p1[0];
            let dy = p2[1] - p1[1];
            if dx.abs() < 1e-4 {
                continue;
            }
            let m = dy / dx;
            let c = p1[1] - m * p1[0];

            let mut inliers = 0;
            for pt in sky_ground_transitions {
                let dist = (m * pt[0] - pt[1] + c).abs() / (m * m + 1.0).sqrt();
                if dist <= self.inlier_threshold_px {
                    inliers += 1;
                }
            }
            if inliers > best_inliers {
                best_inliers = inliers;
                best_line = (m, c);
            }
        }

        if best_inliers < 4 {
            return None;
        }

        let roll_deg = (-best_line.0).atan().to_degrees();
        let mid_x = image_size[0] as f64 * 0.5;
        let horizon_y = best_line.0 * mid_x + best_line.1;
        // Negative pitch indicates nose-up (horizon below center)
        let pitch_deg = ((mid_y - horizon_y) / focal_px).atan().to_degrees();

        Some((pitch_deg, roll_deg))
    }
}

/// Fallback video focal length estimator with test multiplier sweeps.
#[derive(Debug, Clone)]
pub struct FocalLengthEstimator {
    /// Default focal length 35mm-equivalent in mm (nominal 26.0 mm).
    pub default_focal_mm: f64,
    /// Multiplier sweep candidates [1.0, 0.55, 1.45, 2.0].
    pub sweep_multipliers: [f64; 4],
    /// Locked refined focal length in pixels.
    pub locked_focal_px: Option<f64>,
}

impl Default for FocalLengthEstimator {
    fn default() -> Self {
        Self {
            default_focal_mm: 26.0,
            sweep_multipliers: [1.0, 0.55, 1.45, 2.0],
            locked_focal_px: None,
        }
    }
}

impl FocalLengthEstimator {
    /// Computes baseline pixel focal length from equivalent mm and sensor width.
    #[must_use]
    pub fn baseline_focal_px(&self, image_width_px: u32) -> f64 {
        // Standard 36mm sensor diagonal baseline
        (self.default_focal_mm / 36.0) * (image_width_px as f64)
    }

    /// Selects best focal multiplier based on candidate reprojection RMSE scores.
    pub fn select_multiplier(&mut self, scores: &[f64; 4], baseline_px: f64) -> f64 {
        let mut best_idx = 0;
        let mut min_score = f64::INFINITY;
        for (i, &score) in scores.iter().enumerate() {
            if score < min_score {
                min_score = score;
                best_idx = i;
            }
        }
        let chosen_focal = baseline_px * self.sweep_multipliers[best_idx];
        self.locked_focal_px = Some(chosen_focal);
        chosen_focal
    }
}

/// Milliradian covariance scaler maintaining 0.001 pre- and post-scaling.
pub struct MilliradianCovarianceScaler;

impl MilliradianCovarianceScaler {
    /// Scales 6x6 normal matrix H, inverts it, and scales covariance back.
    ///
    /// # Errors
    ///
    /// Returns error if normal matrix inversion fails.
    pub fn invert_with_scaling(
        normal_matrix_h: &SMatrix<f64, 6, 6>,
    ) -> Result<SMatrix<f64, 6, 6>, String> {
        // Scale vector: [1.0, 1.0, 1.0, 0.001, 0.001, 0.001]
        let scale_diag =
            nalgebra::SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 0.001, 0.001, 0.001]);
        let scale_mat = SMatrix::<f64, 6, 6>::from_diagonal(&scale_diag);

        let scaled_h = scale_mat * normal_matrix_h * scale_mat;
        let inv_scaled_h = scaled_h
            .try_inverse()
            .ok_or_else(|| "Degenerate normal matrix inversion failed".to_owned())?;

        let covariance = scale_mat * inv_scaled_h * scale_mat;
        Ok(covariance)
    }
}
