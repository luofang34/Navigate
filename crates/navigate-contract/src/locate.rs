//! Messages between a page and the Navigate locate module that runs in a Web Worker.
//!
//! The page sends one [`LocateRequest`] with the photo, receives
//! [`LocateProgress`] messages, and then one [`LocateResponse`]. With the `serde`
//! feature the types encode as JSON, which crosses the Worker boundary.
//!
//! Heights are above the datum of the terrain model, not above the WGS84
//! ellipsoid: the Terrarium elevation tiles are close to mean sea level. Angles
//! are radians, as everywhere in this crate.

/// Version of these messages. A change that a reader of an earlier version would
/// read wrongly is a new version.
pub const LOCATE_MESSAGES_VERSION: u32 = 1;

/// Camera hints that the module read from a photo, for the page to show and to
/// let the user correct. A hint is not evidence: apps can edit or remove metadata.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PhotoInfo {
    /// Stored image width, pixels.
    pub width_px: u32,
    /// Stored image height, pixels.
    pub height_px: u32,
    /// Focal length from EXIF, pixels of the stored image.
    pub focal_px: Option<f64>,
    /// GPS latitude, radians.
    pub gps_latitude_rad: Option<f64>,
    /// GPS longitude, radians.
    pub gps_longitude_rad: Option<f64>,
    /// GPS altitude, metres.
    pub gps_altitude_m: Option<f64>,
    /// Gimbal compass direction, radians clockwise from north.
    pub gimbal_heading_rad: Option<f64>,
    /// Gimbal angle below the horizon, radians.
    pub gimbal_pitch_rad: Option<f64>,
    /// Height above the take-off point, metres.
    pub relative_altitude_m: Option<f64>,
}

/// Where to look for the camera of one photo.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LocateRequest {
    /// [`LOCATE_MESSAGES_VERSION`] of the sender.
    pub version: u32,
    /// Latitude of the ground at the photo centre, radians.
    pub target_latitude_rad: f64,
    /// Longitude of the ground at the photo centre, radians.
    pub target_longitude_rad: f64,
    /// Compass direction of the view, radians clockwise from north.
    pub heading_rad: f64,
    /// Focal length of the photo as sent, in its pixels.
    pub focal_px: f64,
    /// The module reduces wider photos to this width and scales the focal length.
    pub max_width_px: u32,
    /// Half side of the map area around the target, metres.
    pub area_radius_m: f64,
}

/// One pipeline stage, for a progress display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum LocateStage {
    /// Downloading the module weights.
    Weights,
    /// Map tiles fetched and decoded.
    Tiles,
    /// Candidate views rendered.
    Rendering,
    /// Candidate views matched with the photo.
    Matching,
    /// The best pose checked against a view rendered at that pose.
    Refining,
}

/// Progress of one stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LocateProgress {
    /// Stage.
    pub stage: LocateStage,
    /// Items finished; zero when the stage does not count items.
    pub done: u32,
    /// Items in the stage; zero when the stage does not count items.
    pub total: u32,
}

/// The located camera.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LocatedCamera {
    /// Latitude, radians.
    pub latitude_rad: f64,
    /// Longitude, radians.
    pub longitude_rad: f64,
    /// Height above the terrain-model datum, metres.
    pub height_m: f64,
    /// Compass direction of the optical axis, radians clockwise from north.
    pub heading_rad: f64,
    /// Angle of the optical axis below the horizon, radians.
    pub pitch_rad: f64,
    /// Rotation about the optical axis, radians.
    pub roll_rad: f64,
    /// Image width that the pose refers to, pixels.
    pub width_px: u32,
    /// Image height that the pose refers to, pixels.
    pub height_px: u32,
    /// Focal length at that size, pixels.
    pub focal_px: f64,
    /// Geometric inliers of the accepted fit.
    pub inliers: u32,
    /// Root mean square reprojection error, pixels.
    pub reprojection_rms_px: f64,
    /// True when a view rendered at the pose confirmed it.
    pub confirmed: bool,
    /// SHA-256 of the map tiles that the module used.
    pub map_sha256: String,
}

/// Final message of one request.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "outcome", rename_all = "snake_case"))]
pub enum LocateResponse {
    /// A pose passed the geometry checks.
    Found {
        /// The camera.
        camera: LocatedCamera,
    },
    /// No candidate passed the geometry checks.
    NotFound {
        /// Candidates evaluated.
        candidates: u32,
        /// Most correspondences of one candidate before the checks.
        best_matches: u32,
    },
    /// The module could not run the request.
    Failed {
        /// Reason for a person.
        reason: String,
    },
}

#[cfg(all(test, feature = "serde"))]
mod tests;
