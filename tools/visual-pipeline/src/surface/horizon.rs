//! Robust horizon line fitting, airframe and propeller exclusion, and attitude estimation.

use nalgebra::Matrix3;

/// Navigate four-class label index for sky.
pub const SKY_LABEL: u8 = 0;
/// Navigate four-class label index for ground terrain.
pub const GROUND_LABEL: u8 = 1;
/// Navigate four-class label index for airframe parts.
pub const AIRFRAME_LABEL: u8 = 2;
/// Navigate four-class label index for rotating propellers.
pub const PROPELLER_LABEL: u8 = 3;

/// ADE20K semantic label index for sky.
pub const ADE20K_SKY_LABEL: u8 = 2;
/// ADE20K semantic label index for airplane bodies.
pub const ADE20K_AIRPLANE_LABEL: u8 = 90;

/// Robust RANSAC and Theil-Sen horizon line estimator.
#[derive(Clone, Copy, Debug)]
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
    /// Fits a horizon line (`y = m * x + c`) rejecting outliers, and extracts pitch and roll angles.
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
        let mid_x = f64::from(image_size[0]) * 0.5;
        let mid_y = f64::from(image_size[1]) * 0.5;

        let (slope, intercept) = ransac_fit_line(
            sky_ground_transitions,
            self.inlier_threshold_px,
            self.max_iterations,
        )?;

        let roll_deg = (-slope).atan().to_degrees();
        let horizon_y = slope * mid_x + intercept;
        let pitch_deg = ((mid_y - horizon_y) / focal_px.max(1.0))
            .atan()
            .to_degrees();

        if pitch_deg.is_nan() || roll_deg.is_nan() {
            return None;
        }

        Some((pitch_deg, roll_deg))
    }
}

/// Fits a 2D line using RANSAC with deterministic pair sampling and Theil-Sen refinement.
#[must_use]
pub fn ransac_fit_line(
    points: &[[f64; 2]],
    inlier_thresh: f64,
    max_iterations: usize,
) -> Option<(f64, f64)> {
    let n = points.len();
    if n < 4 {
        return None;
    }

    let mut best_inliers = 0;
    let mut best_line = (0.0, 0.0);

    let strides = [1, 2, 3, 5, 7, 11, 13, 17, 23];
    let mut iters = 0;

    'outer: for &stride in &strides {
        for i in 0..n {
            let j = (i + stride) % n;
            if i == j {
                continue;
            }
            let p1 = points[i];
            let p2 = points[j];
            let dx = p2[0] - p1[0];
            let dy = p2[1] - p1[1];
            if dx.abs() < 2.0 {
                continue;
            }
            let m = dy / dx;
            let c = p1[1] - m * p1[0];

            let denom = (m * m + 1.0).sqrt();
            let mut count = 0;
            for pt in points {
                let dist = (m * pt[0] - pt[1] + c).abs() / denom;
                if dist <= inlier_thresh {
                    count += 1;
                }
            }

            if count > best_inliers {
                best_inliers = count;
                best_line = (m, c);
            }

            iters += 1;
            if iters >= max_iterations {
                break 'outer;
            }
        }
    }

    if best_inliers < 4 {
        return None;
    }

    // Refine line parameters using Theil-Sen median slope on inliers
    let (m, c) = best_line;
    let denom = (m * m + 1.0).sqrt();
    let inliers: Vec<[f64; 2]> = points
        .iter()
        .copied()
        .filter(|pt| (m * pt[0] - pt[1] + c).abs() / denom <= inlier_thresh)
        .collect();

    theil_sen_fit_line(&inliers).or(Some(best_line))
}

/// Fits a 2D line using Theil-Sen median slope regression.
#[must_use]
pub fn theil_sen_fit_line(points: &[[f64; 2]]) -> Option<(f64, f64)> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[j][0] - points[i][0];
            let dy = points[j][1] - points[i][1];
            if dx.abs() >= 1.0 {
                slopes.push(dy / dx);
            }
        }
    }
    if slopes.is_empty() {
        return None;
    }
    slopes.sort_by(|a, b| a.total_cmp(b));
    let median_slope = slopes[slopes.len() / 2];

    let mut intercepts: Vec<f64> = points.iter().map(|p| p[1] - median_slope * p[0]).collect();
    intercepts.sort_by(|a, b| a.total_cmp(b));
    let median_intercept = intercepts[intercepts.len() / 2];

    Some((median_slope, median_intercept))
}

/// Estimates `(down_deg, roll_deg)` from the sky boundary in a semantic label map.
///
/// Excludes airframe and propeller occlusions and supports negative pitch angles (nose-up).
#[must_use]
pub fn horizon_attitude(
    labels: &[u8],
    width: u32,
    height: u32,
    k: &Matrix3<f64>,
) -> Option<(f64, f64)> {
    if labels.len() != (width * height) as usize || width < 8 || height < 8 {
        return None;
    }
    let max_lbl = labels.iter().copied().max().unwrap_or(0);
    let is_four_class = max_lbl <= 3;
    let sky_id = if is_four_class {
        SKY_LABEL
    } else {
        ADE20K_SKY_LABEL
    };

    let mut pts = Vec::new();
    let step_x = 4_u32;
    for x in (0..width).step_by(step_x as usize) {
        let mut last_sky_y = None;
        for y in 0..(height - 1) {
            let a = labels[(y * width + x) as usize];
            let b = labels[((y + 1) * width + x) as usize];

            if is_four_class {
                if a == sky_id && b == GROUND_LABEL {
                    last_sky_y = Some(f64::from(y) + 0.5);
                    break;
                }
                if a == sky_id && (b == AIRFRAME_LABEL || b == PROPELLER_LABEL) {
                    // Sky transitions into airframe or propeller, not ground
                    break;
                }
            } else {
                if a == sky_id && b != sky_id && b != ADE20K_AIRPLANE_LABEL {
                    last_sky_y = Some(f64::from(y) + 0.5);
                    break;
                }
                if a == sky_id && b == ADE20K_AIRPLANE_LABEL {
                    break;
                }
            }
        }
        if let Some(hy) = last_sky_y {
            pts.push([f64::from(x), hy]);
        }
    }

    if pts.len() < 4 {
        return None;
    }

    let estimator = RansacHorizonEstimator::default();
    let (slope, intercept) = ransac_fit_line(
        &pts,
        estimator.inlier_threshold_px,
        estimator.max_iterations,
    )?;

    let roll_deg = (-slope).atan().to_degrees();
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let fy = k[(1, 1)].max(1.0);
    let y_at_cx = slope * cx + intercept;
    let down_deg = ((cy - y_at_cx) / fy).atan().to_degrees();

    if down_deg.is_nan() || roll_deg.is_nan() {
        return None;
    }

    Some((down_deg, roll_deg))
}
