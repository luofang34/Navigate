//! Tuning of the terrain means.

use navigate_camera::Intrinsics;

use crate::MapRefusal;

/// What set the working resolution of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkingLimit {
    /// The frame has the resolution of the camera.
    Native,
    /// The ground size of a pixel is that of the map imagery.
    MapTexel,
    /// `max_match_pixels` bounds the frame: its pixels are larger on the
    /// ground than those of the map, so the match is coarser than the
    /// map allows.
    PixelBound,
    /// `min_match_pixels` holds the frame above the resolution of a
    /// coarse map.
    PixelFloor,
    /// The frame has the resolution of the stored keyframe.
    Keyframe,
}

/// The camera of the reference view and of the match for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkingCamera {
    /// The declared camera, resampled: the same field of view.
    pub intrinsics: Intrinsics,
    /// Ground size of one working pixel at the range of the view, metres.
    pub ground_pixel_m: f64,
    /// What set the resolution.
    pub limit: WorkingLimit,
}

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
    /// Grid stride of the adaptive renderer, pixels: one terrain ray for
    /// each block of this side, and interpolation between the rays.
    pub stride: u32,
    /// Luma step between rays that the renderer refines with a ray for
    /// each pixel. The largest value refines none: on textured ground a
    /// small value casts a ray for nearly each pixel, at six times the
    /// cost of the frame and with no gain in the fix.
    pub luma_edge: u8,
    /// Relative depth step that the renderer refines.
    pub depth_edge: f32,
    /// Largest count of pixels of the reference view and of the match: the
    /// bound of the cost of one frame.
    pub max_match_pixels: u32,
    /// Smallest count of pixels that a resampled frame keeps. The matcher
    /// needs corners in each part of the image, so a frame is not made
    /// smaller than this for a coarse map.
    pub min_match_pixels: u32,
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
            stride: 4,
            luma_edge: u8::MAX,
            depth_edge: 0.02,
            max_match_pixels: 320 * 240,
            min_match_pixels: 160 * 120,
        }
    }
}

impl TerrainConfig {
    /// The working camera for one frame: the camera of the reference view
    /// and of the match. `range_m` is the distance to the ground along
    /// the optical axis, and `map_texel_m` is the ground size of one pixel
    /// of the imagery in use.
    ///
    /// A match finer than the map gains nothing, so a frame whose pixels
    /// are smaller on the ground than the map pixels is resampled until
    /// they are equal. `max_match_pixels` then bounds the cost, and
    /// `min_match_pixels` keeps enough of the image for the matcher. The
    /// field of view does not change. `None` for values that give no
    /// usable camera.
    #[must_use]
    pub fn working_camera(
        &self,
        declared: &Intrinsics,
        range_m: f64,
        map_texel_m: f64,
    ) -> Option<WorkingCamera> {
        declared.validate().ok()?;
        let usable = [range_m, map_texel_m]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0);
        if !usable {
            return None;
        }
        let pixels = f64::from(declared.width) * f64::from(declared.height);
        let texel = (range_m / declared.fx / map_texel_m).min(1.0);
        let bound = (f64::from(self.max_match_pixels) / pixels).sqrt().min(1.0);
        let floor = (f64::from(self.min_match_pixels) / pixels).sqrt().min(1.0);
        let (side, limit) = if texel.min(bound) < floor {
            (floor, WorkingLimit::PixelFloor)
        } else if texel <= bound {
            (texel, WorkingLimit::MapTexel)
        } else {
            (bound, WorkingLimit::PixelBound)
        };
        if side >= 1.0 {
            return Some(WorkingCamera {
                intrinsics: *declared,
                ground_pixel_m: range_m / declared.fx,
                limit: WorkingLimit::Native,
            });
        }
        let size = [
            (f64::from(declared.width) * side).round() as u32,
            (f64::from(declared.height) * side).round() as u32,
        ];
        let intrinsics = declared.resized(size).ok()?;
        Some(WorkingCamera {
            intrinsics,
            ground_pixel_m: range_m / intrinsics.fx,
            limit,
        })
    }

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
        let counts = self.steps > 0
            && self.stride > 0
            && self.min_match_pixels >= 64 * 48
            && self.min_match_pixels <= self.max_match_pixels;
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
