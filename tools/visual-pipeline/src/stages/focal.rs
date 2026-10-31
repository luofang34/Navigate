//! Fallback video focal length estimation with test multiplier sweeps.

/// Default 35mm-equivalent focal length in millimetres (26.0 mm).
pub const DEFAULT_FOCAL_MM: f64 = 26.0;

/// Focal multiplier sweep values for unverified cold anchors.
pub const FOCAL_SWEEP_MULTIPLIERS: [f64; 4] = [1.0, 0.55, 1.45, 2.0];

/// Estimator for camera focal length with sweep multipliers and lock state.
#[derive(Debug, Clone, PartialEq)]
pub struct FocalLengthEstimator {
    /// Nominal 35mm-equivalent focal length in millimetres.
    pub default_focal_mm: f64,
    /// Multipliers evaluated during cold anchor acquisition.
    pub sweep_multipliers: [f64; 4],
    /// Locked refined focal length in pixels.
    pub locked_focal_px: Option<f64>,
}

impl Default for FocalLengthEstimator {
    fn default() -> Self {
        Self {
            default_focal_mm: DEFAULT_FOCAL_MM,
            sweep_multipliers: FOCAL_SWEEP_MULTIPLIERS,
            locked_focal_px: None,
        }
    }
}

impl FocalLengthEstimator {
    /// Computes baseline pixel focal length using standard 36mm sensor width.
    #[must_use]
    pub fn baseline_focal_px(&self, image_width_px: u32) -> f64 {
        (self.default_focal_mm / 36.0) * f64::from(image_width_px)
    }

    /// Selects the best multiplier based on candidate reprojection RMSE scores.
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
