//! Camera pose of one photo from map imagery and terrain.
//!
//! The pipeline fetches imagery and elevation tiles through a [`TileSource`],
//! renders candidate views around a ground target with the Navigate reference
//! renderer, matches each view with the photo through an [`AsyncImageMatcher`]
//! (for example `navigate_match::LoftrMatcher`), fits the pose with the shared
//! verifier, and checks the best pose against a view rendered at that pose. The
//! same code runs natively and in a browser Web Worker.
//!
//! [`AsyncImageMatcher`]: navigate_visual::AsyncImageMatcher
mod area;
mod candidates;
mod error;
#[cfg(feature = "http")]
mod http;
mod locate;
pub mod messages;
mod metadata;
mod scene;
mod search;

pub use area::{TileGrid, tile_xy};
pub use candidates::{Candidate, LookAt, angles, look_at, orientation};
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
