//! Tuning of the terrain means.

use crate::MapRefusal;

/// How the terrain means searches, renders, and admits. The defaults suit a
/// camera 1 km to 3 km above the ground.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TerrainConfig {
    /// Horizontal one-sigma multiples of the navigator that the position
    /// search spans.
    pub search_sigma_multiple: f64,
    /// Smallest position search radius, metres.
    pub min_search_radius_m: f64,
    /// Largest position search radius, metres.
    pub max_search_radius_m: f64,
    /// Attitude search bound, radians.
    pub attitude_radius_rad: f64,
    /// Longest time from the middle of the exposure to the delivery of a
    /// frame, seconds. An older frame is not matched.
    pub max_capture_lag_s: f64,
    /// Bound on the acceleration of the vehicle between the exposure and
    /// the delivery, m/s². It widens the position search with the lag.
    pub lag_acceleration_m_s2: f64,
    /// Bound on the body rate between the exposure and the delivery,
    /// rad/s. It widens the attitude search with the lag.
    pub lag_rate_rad_s: f64,
    /// Smallest share of a reference view that must show the map. A view
    /// with less is outside the coverage of the package.
    pub min_view_coverage: f64,
    /// Farthest terrain distance of a rendered reference, metres.
    pub far_m: f64,
    /// Ray-march steps of a rendered reference.
    pub steps: u32,
    /// Bisection refinements of a bracketed terrain hit.
    pub refinements: u32,
    /// Grid stride of the adaptive renderer, pixels.
    pub stride: u32,
    /// Luma step that the renderer refines and does not interpolate.
    pub luma_edge: u8,
    /// Relative depth step that the renderer refines.
    pub depth_edge: f32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            search_sigma_multiple: 3.0,
            min_search_radius_m: 30.0,
            max_search_radius_m: 300.0,
            attitude_radius_rad: 0.05,
            max_capture_lag_s: 0.5,
            lag_acceleration_m_s2: 20.0,
            lag_rate_rad_s: 0.5,
            min_view_coverage: 0.5,
            far_m: 6_000.0,
            steps: 32,
            refinements: 10,
            stride: 8,
            luma_edge: 8,
            depth_edge: 0.02,
        }
    }
}

impl TerrainConfig {
    /// Checks each value.
    pub(crate) fn validate(&self) -> Result<(), MapRefusal> {
        let positive = [
            self.search_sigma_multiple,
            self.min_search_radius_m,
            self.max_search_radius_m,
            self.attitude_radius_rad,
            self.far_m,
            f64::from(self.depth_edge),
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0);
        let bounds = [
            self.max_capture_lag_s,
            self.lag_acceleration_m_s2,
            self.lag_rate_rad_s,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0);
        let ordered = self.min_search_radius_m <= self.max_search_radius_m
            && (0.0..=1.0).contains(&self.min_view_coverage);
        let counts = self.steps > 0 && self.stride > 0;
        if positive && bounds && ordered && counts {
            Ok(())
        } else {
            Err(MapRefusal::Request {
                field: "terrain configuration",
            })
        }
    }

    /// The position and attitude search bounds for a navigator horizontal
    /// one-sigma `sigma_h_m` and a frame delivered `lag_s` after its
    /// exposure.
    pub(crate) fn search(&self, sigma_h_m: f64, lag_s: f64) -> (f64, f64) {
        let position = (self.search_sigma_multiple * sigma_h_m)
            .clamp(self.min_search_radius_m, self.max_search_radius_m)
            + 0.5 * self.lag_acceleration_m_s2 * lag_s * lag_s;
        let attitude =
            (self.attitude_radius_rad + self.lag_rate_rad_s * lag_s).min(core::f64::consts::PI);
        (position, attitude)
    }
}
