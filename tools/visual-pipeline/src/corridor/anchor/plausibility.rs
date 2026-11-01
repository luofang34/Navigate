//! CLI bridge and `Surface` terrain adapter for `navigate_locate::corridor::plausibility`.

pub use navigate_locate::corridor::plausibility::{
    CorridorTerrain, PhysicalPlausibilityConfig, PlausibilityError,
    compute_fisher_geometry_covariance, compute_fisher_min_eigenvalue, compute_holdout_rmse_px,
    compute_inlier_positive_depth_fraction, is_plausible_candidate, is_sane_seed_candidate,
    local_camera_rotation, verify_candidate_plausibility, verify_physical_plausibility,
    verify_prior_bounding,
};

use crate::surface::Surface;
use nalgebra::Matrix3;

impl CorridorTerrain for Surface {
    fn geodetic(&self, local_enu: [f64; 3]) -> [f64; 3] {
        self.geodetic(local_enu)
    }

    fn local(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
        self.local(lat_deg, lon_deg, alt_m)
    }

    fn sample_dem(&self, lat_deg: f64, lon_deg: f64) -> Option<f64> {
        self.sample_dem(lat_deg, lon_deg)
    }

    fn camera_rotation(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        heading_deg: f64,
        down_deg: f64,
        roll_deg: f64,
    ) -> Matrix3<f64> {
        self.camera_rotation(lat_deg, lon_deg, heading_deg, down_deg, roll_deg)
    }
}
