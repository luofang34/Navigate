//! Messages between a page and the Navigate search module that runs in a Web Worker.
//!
//! The page sends one [`SearchRequest`] with the photo, receives [`SearchProgress`]
//! messages, and then one [`SearchResponse`]. The search region and the priors
//! replace the single target of a locate request: the module plans its own
//! hypotheses, ranks a place index and locates the camera at the best places.
//! All angles use radians. All heights are in meters above the ground.
use super::LocatedCamera;

/// Version of these messages. A change that a reader of an earlier version would
/// read wrongly is a new version.
pub const SEARCH_MESSAGES_VERSION: u32 = 1;

/// Where the camera can be.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum SearchRegion {
    /// All points within `radius_m` of the center.
    Circle {
        /// Latitude of the center, radians.
        latitude_rad: f64,
        /// Longitude of the center, radians.
        longitude_rad: f64,
        /// Radius, meters.
        radius_m: f64,
    },
    /// All points inside a closed outline.
    Polygon {
        /// Latitude and longitude of each corner, radians.
        points_rad: Vec<[f64; 2]>,
    },
}

/// A value and its standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Prior {
    /// Most likely value.
    pub value: f64,
    /// One standard deviation.
    pub sigma: f64,
}

/// What the page knows about the camera. Every prior is optional.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SearchPriors {
    /// Compass heading of the view, radians clockwise from north.
    pub heading_rad: Option<Prior>,
    /// Camera height above the ground, meters, for example from a barometer.
    pub height_m: Option<Prior>,
    /// Angle of the optical axis below the horizon, radians.
    pub pitch_rad: Option<Prior>,
}

/// One search.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SearchRequest {
    /// [`SEARCH_MESSAGES_VERSION`] of the sender.
    pub version: u32,
    /// Where the camera can be.
    pub region: SearchRegion,
    /// Focal length of the photo as sent, in its pixels.
    pub focal_px: f64,
    /// The module reduces wider photos to this width and scales the focal length.
    pub max_width_px: u32,
    /// What the page knows about the camera.
    pub priors: SearchPriors,
    /// Base URL of the place index; `manifest.json` and the shard paths are relative to it.
    pub index_url: String,
}

/// One search stage, for a progress display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SearchStage {
    /// Downloading the module weights.
    Weights,
    /// Photo projections described.
    Describing,
    /// Index shards read.
    Index,
    /// Ranked places checked against map windows.
    Checking,
    /// Map tiles of the locate step fetched.
    Tiles,
    /// Candidate views of the locate step rendered.
    Rendering,
    /// Candidate views matched with the photo.
    Matching,
    /// The best pose checked against a view rendered at that pose.
    Refining,
}

/// Progress of one stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SearchProgress {
    /// Stage.
    pub stage: SearchStage,
    /// Items finished; zero when the stage does not count items.
    pub done: u32,
    /// Items in the stage; zero when the stage does not count items.
    pub total: u32,
}

/// Final message of one search.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "outcome", rename_all = "snake_case"))]
pub enum SearchResponse {
    /// A place passed the check and the locate step accepted a pose there.
    Found {
        /// The camera.
        camera: LocatedCamera,
        /// Places checked.
        checked: u32,
    },
    /// No place passed, or the locate step rejected the best places.
    NotFound {
        /// Places ranked.
        ranked: u32,
        /// Places checked.
        checked: u32,
        /// Most inliers of one check.
        best_inliers: u32,
    },
    /// The index does not cover the region. A region small enough for the dense
    /// search can use a locate request instead.
    Uncovered {
        /// True when the region is small enough for a locate request.
        dense_search: bool,
        /// Reason for a person.
        reason: String,
    },
    /// The module could not run the request.
    Failed {
        /// Reason for a person.
        reason: String,
    },
}

#[cfg(all(test, feature = "serde"))]
mod tests;
