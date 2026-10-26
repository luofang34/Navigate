//! Messages between a browser page and the Navigate locate worker.
//!
//! A Web Worker is an isolated browser thread.
//! The page sends one [`LocateRequest`] with the photo.
//! The worker sends [`LocateProgress`] messages and then one [`LocateResponse`].
//! Enable the `serde` feature to serialize these messages as JSON across the Web Worker boundary.
//!
//! All heights use the terrain-model datum (mean sea level for Terrarium tiles).
//! All angles use radians.
//! The [`search`] module holds the messages of a region search.

pub mod search;

/// Protocol version of the locate worker messages.
///
/// Increment this version when a change breaks compatibility with an older reader.
pub const LOCATE_MESSAGES_VERSION: u32 = 1;

/// Camera hints that the module reads from photo metadata.
///
/// The page shows these hints so the operator can verify or correct them.
/// Metadata hints are not verified sensor measurements.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PhotoInfo {
    /// Stored image width in pixels.
    pub width_px: u32,
    /// Stored image height in pixels.
    pub height_px: u32,
    /// Focal length from EXIF metadata in pixels of the stored image.
    pub focal_px: Option<f64>,
    /// GPS latitude in radians.
    pub gps_latitude_rad: Option<f64>,
    /// GPS longitude in radians.
    pub gps_longitude_rad: Option<f64>,
    /// GPS altitude in meters.
    pub gps_altitude_m: Option<f64>,
    /// Gimbal compass direction in radians clockwise from north.
    pub gimbal_heading_rad: Option<f64>,
    /// Gimbal pitch angle below the horizon in radians.
    pub gimbal_pitch_rad: Option<f64>,
    /// Height above the take-off point in meters.
    pub relative_altitude_m: Option<f64>,
}

/// Search parameters for the camera pose of one photo.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LocateRequest {
    /// [`LOCATE_MESSAGES_VERSION`] of the sender.
    pub version: u32,
    /// Ground latitude at the photo center in radians.
    pub target_latitude_rad: f64,
    /// Ground longitude at the photo center in radians.
    pub target_longitude_rad: f64,
    /// View compass direction in radians clockwise from north.
    pub heading_rad: f64,
    /// Photo focal length in pixels of the input image.
    pub focal_px: f64,
    /// Maximum working width in pixels before the module downscales the photo.
    pub max_width_px: u32,
    /// Half-width of the map search square around the target in meters.
    pub area_radius_m: f64,
}

/// Active stage of the locate pipeline for a progress display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum LocateStage {
    /// The worker downloads model weights.
    Weights,
    /// The worker fetches and decodes map tiles.
    Tiles,
    /// The worker renders candidate views.
    Rendering,
    /// The worker matches candidate views against the photo.
    Matching,
    /// The worker verifies the pose against a view rendered at that pose.
    Refining,
}

/// Progress count for one [`LocateStage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LocateProgress {
    /// Active pipeline stage.
    pub stage: LocateStage,
    /// Completed item count, or zero when the stage does not count items.
    pub done: u32,
    /// Total item count, or zero when the stage does not count items.
    pub total: u32,
}

/// Verified camera pose and quality summary for one photo.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LocatedCamera {
    /// Camera latitude in radians.
    pub latitude_rad: f64,
    /// Camera longitude in radians.
    pub longitude_rad: f64,
    /// Camera height above the terrain-model datum in meters.
    pub height_m: f64,
    /// Optical-axis compass direction in radians clockwise from north.
    pub heading_rad: f64,
    /// Optical-axis pitch angle below the horizon in radians.
    pub pitch_rad: f64,
    /// Rotation angle around the optical axis in radians.
    pub roll_rad: f64,
    /// Camera-to-world unit quaternion `[x, y, z, w]` in East-North-Up coordinates.
    pub orientation_xyzw: [f64; 4],
    /// Working image width in pixels.
    pub width_px: u32,
    /// Working image height in pixels.
    pub height_px: u32,
    /// Working focal length in pixels.
    pub focal_px: f64,
    /// Geometric inlier count of the accepted pose fit.
    pub inliers: u32,
    /// Root-mean-square reprojection error in pixels.
    pub reprojection_rms_px: f64,
    /// True when a view rendered at the pose confirmed the fit.
    pub confirmed: bool,
    /// Count of other candidate poses that also passed the geometry checks.
    #[cfg_attr(feature = "serde", serde(default))]
    pub competing_candidates: u32,
    /// SHA-256 digest of the map tile manifest.
    pub map_sha256: String,
}

/// Terminal response message for one [`LocateRequest`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "outcome", rename_all = "snake_case"))]
pub enum LocateResponse {
    /// A candidate pose passed the geometry checks.
    Found {
        /// Located camera solution.
        camera: LocatedCamera,
    },
    /// No candidate pose passed the geometry checks.
    NotFound {
        /// Number of evaluated candidates.
        candidates: u32,
        /// Highest correspondence count among evaluated candidates before geometry checks.
        best_matches: u32,
    },
    /// The worker could not execute the request.
    Failed {
        /// Diagnostic message for the operator.
        reason: String,
    },
}

#[cfg(all(test, feature = "serde"))]
mod tests;
