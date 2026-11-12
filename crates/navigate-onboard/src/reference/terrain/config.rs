//! Tuning of the terrain means.

use navigate_camera::Intrinsics;
use navigate_visual::acquisition::AcquisitionConfig;

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
    /// Largest position search radius of a reference view, metres. A
    /// wider belief is first acquired.
    pub max_search_radius_m: f64,
    /// Largest position search radius of an acquisition, metres: the cost
    /// of an acquisition grows with its square. A value not above the
    /// largest search radius of a view turns acquisition off: a wider
    /// belief is then searched only that far about itself.
    pub max_acquisition_radius_m: f64,
    /// Side of a ground cell of an acquisition, metres.
    pub acquisition_cell_m: f64,
    /// What an acquisition requires of the place it finds.
    pub acquisition: AcquisitionConfig,
    /// Largest heading error of the belief that an acquisition searches,
    /// radians: it lays the frame at headings this far about the believed
    /// one, fewer for a belief whose attitude one-sigma is small.
    pub max_acquisition_rotation_rad: f64,
    /// Step of the headings at which an acquisition lays the frame,
    /// radians.
    pub acquisition_rotation_step_rad: f64,
    /// Cell pairs that an acquisition compares for one frame. A search
    /// that needs more goes on over the next frames, so that one frame
    /// takes a bounded time; the default is some 60 ms on WebAssembly.
    pub acquisition_pairs_per_frame: u64,
    /// Longest time between two acquisitions that confirm one another,
    /// seconds.
    pub acquisition_confirm_s: f64,
    /// Smallest attitude search bound, radians: the bound is the search
    /// multiple of the attitude one-sigma of the belief when that is
    /// larger.
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
    /// Smallest structure of a frame that is matched: the contrast between
    /// blocks of 8 by 8 pixels as a multiple of the block contrast of white
    /// noise with the pixel contrast of the frame. A frame of noise alone
    /// has one.
    pub min_structure: f64,
    /// Smallest block contrast of a frame of an oblique camera that is
    /// matched: the mean absolute difference of the means of adjacent
    /// blocks of 8 by 8 pixels, in counts of the 8-bit frame as the host
    /// delivers it, before the matcher normalises its contrast. A sensor
    /// of another bit depth or gain rescales it to its counts. It holds
    /// for a camera whose optical axis is `faint_floor_full_rad` or more
    /// from the local down. Under it, the matcher finds ground that its
    /// normalisation raised above the noise, and the fix is off by more
    /// than its covariance states: with the camera mismatch of
    /// `tests::geometry`, frames of 1 to 3.5 counts over 45° views were
    /// over-confident and frames of 3.6 and more were not.
    pub min_block_contrast: f64,
    /// Angle of the optical axis from the local down up to which a frame
    /// has no block contrast floor, radians. Default: 30°. At the nadir,
    /// at 15° and at 30°, the fixes of frames of 1.3 to 4 counts state
    /// their error, with the sensor of `tests::geometry` and with a
    /// sensor of two counts of noise; under about one count the matcher
    /// finds no fix. From this angle the floor rises in proportion to the
    /// angle, to `min_block_contrast` at `faint_floor_full_rad`.
    pub faint_floor_from_rad: f64,
    /// Angle of the optical axis from the local down from which the
    /// floor is `min_block_contrast`, radians. Default: 45°, the view of
    /// the frames that set that floor.
    pub faint_floor_full_rad: f64,
    /// Largest count of pixels of the reference view and of the match: the
    /// bound of the cost of one frame.
    pub max_match_pixels: u32,
    /// Smallest count of pixels that a resampled frame keeps. The matcher
    /// needs corners in each part of the image, so a frame is not made
    /// smaller than this for a coarse map.
    pub min_match_pixels: u32,
    /// Length over which the error of the elevation model is correlated,
    /// metres. A package states the one-sigma of its elevation and not
    /// this length; the tilt of that error under a footprint moves a fix
    /// horizontally, and the length sets how large the tilt can be. The
    /// default suits a 1-arc-second radar model such as SRTM over farmland
    /// and hills.
    pub dem_correlation_length_m: f64,
    /// How long a fix that the filter took counts toward the integrity
    /// assessment of the solution, seconds. The filter gets one fix for
    /// each map cell, so the fixes come as far apart as the vehicle needs
    /// for a cell: a cell of 450 m (zoom 16 at middle latitudes) takes
    /// 30 s at 15 m/s, and less for a faster vehicle. The default holds
    /// that time; between two fixes the solution then still names the
    /// terrain camera as a contributing sensor. A vehicle that needs
    /// longer for a cell, or a map with larger cells, needs more.
    pub fix_hold_s: f64,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            search_sigma_multiple: 3.0,
            min_search_radius_m: 30.0,
            max_search_radius_m: 300.0,
            max_acquisition_radius_m: 3_000.0,
            acquisition_cell_m: 32.0,
            acquisition: AcquisitionConfig::default(),
            max_acquisition_rotation_rad: 6.0_f64.to_radians(),
            acquisition_rotation_step_rad: 2.0_f64.to_radians(),
            acquisition_pairs_per_frame: 20_000_000,
            acquisition_confirm_s: 10.0,
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
            min_structure: 2.0,
            min_block_contrast: 4.0,
            faint_floor_from_rad: 30.0_f64.to_radians(),
            faint_floor_full_rad: 45.0_f64.to_radians(),
            max_match_pixels: 320 * 240,
            min_match_pixels: 160 * 120,
            dem_correlation_length_m: 300.0,
            fix_hold_s: 30.0,
        }
    }
}

impl TerrainConfig {
    /// The block contrast floor of a frame whose optical axis is
    /// `off_nadir_rad` from the local down, counts.
    pub(crate) fn faint_floor(&self, off_nadir_rad: f64) -> f64 {
        let span = self.faint_floor_full_rad - self.faint_floor_from_rad;
        let part = ((off_nadir_rad - self.faint_floor_from_rad) / span).clamp(0.0, 1.0);
        self.min_block_contrast * part
    }

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
            self.max_acquisition_radius_m,
            self.acquisition_cell_m,
            self.acquisition_confirm_s,
            self.acquisition_rotation_step_rad,
            self.acquisition.exclusion_m,
            self.attitude_radius_rad,
            self.far_m,
            self.min_structure,
            self.min_block_contrast,
            f64::from(self.depth_edge),
            self.dem_correlation_length_m,
            self.fix_hold_s,
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
            && self.max_acquisition_rotation_rad.is_finite()
            && self.max_acquisition_rotation_rad >= 0.0
            && self.acquisition_pairs_per_frame > 0
            && self.acquisition.peaks > 0
            && self.acquisition.min_score.is_finite()
            && self.acquisition.min_margin.is_finite()
            && (0.0..=1.0).contains(&self.min_view_coverage)
            && self.faint_floor_from_rad >= 0.0
            && self.faint_floor_from_rad < self.faint_floor_full_rad
            && self.faint_floor_full_rad.is_finite();
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
    /// one-sigma `sigma_h_m` and attitude one-sigma `sigma_attitude_rad`,
    /// and a frame delivered `lag_s` after its exposure.
    pub(crate) fn search(
        &self,
        sigma_h_m: f64,
        sigma_attitude_rad: Option<f64>,
        lag_s: f64,
    ) -> (f64, f64) {
        let position = (self.search_sigma_multiple * sigma_h_m)
            .clamp(self.min_search_radius_m, self.max_search_radius_m)
            + 0.5 * self.lag_acceleration_m_s2 * lag_s * lag_s;
        let believed = self.search_sigma_multiple * sigma_attitude_rad.unwrap_or(0.0);
        let attitude = (self.attitude_radius_rad.max(believed) + self.lag_rate_rad_s * lag_s)
            .min(core::f64::consts::PI);
        (position, attitude)
    }
}
