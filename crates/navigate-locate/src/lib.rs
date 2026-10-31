//! Camera pose localization and corridor tracking from map imagery and terrain.
//!
//! A [`TileSource`] supplies imagery and elevation tiles. The single-photo
//! pipeline renders candidate views around a target, matches each view with an
//! [`AsyncImageMatcher`], fits a pose through [`PairLocalizer`] or
//! [`FeatureLocalizer`], and verifies the best pose against a rendered view.
//! The [`corridor`] module tracks continuous flight sequences with 3D map
//! landmarks, platform dynamics gates, and `VisualSession` fix admission. Both
//! pipelines run natively and in a browser Web Worker without filesystem I/O.
//!
//! [`AsyncImageMatcher`]: navigate_visual::AsyncImageMatcher
mod area;
mod candidates;
pub mod corridor;
mod error;
#[cfg(feature = "http")]
mod http;
mod locate;
pub mod messages;
mod metadata;
mod scene;
mod search;
mod workflow;

pub use area::{TileGrid, tile_xy};
pub use candidates::{Candidate, LookAt, angles, look_at, orientation};
pub use corridor::{
    AnchorWorkerState, CorridorTracker, DeadReckonEstimate, DeadReckonPath, DualPathDeadReckoner,
    KinematicState, OdometryStep, TrackingSearchStrategy, VehicleDynamicsProfile,
    VisualCorridorSession, WarmTrackAttempt,
};
pub use error::LocateError;
#[cfg(feature = "http")]
pub use http::CachedHttpSource;
pub use locate::{LocateRequest, Located, Outcome, Progress, locate};
pub use metadata::PhotoMetadata;
pub use scene::{SceneRequest, TileKind, TileSource, build as build_scene, height_at};
pub use search::{
    Horizon, IndexSource, PlaceEncoder, SearchOutcome, SearchProgress, SearchSources, View,
    horizon, search, search_canvas,
};
pub use workflow::{
    CandidateReport, FeatureLocalizer, FrameReport, PairLocalizer, PreparedReference,
    ResourceLimits, StageTimes, WorkflowError,
};
