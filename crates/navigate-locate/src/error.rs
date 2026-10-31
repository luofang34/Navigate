//! Locate errors with the stage and the item that failed.
use thiserror::Error;

/// A failure that stops a localization. A pose that the geometry rejects is not
/// an error; [`crate::Outcome::NotFound`] reports it.
#[derive(Debug, Error)]
pub enum LocateError {
    /// A request value is outside its permitted range.
    #[error("invalid request: {0}")]
    Request(&'static str),
    /// The tile source failed for one tile.
    #[error("{kind} tile {z}/{x}/{y} could not be fetched: {message}")]
    Fetch {
        /// Tile kind.
        kind: &'static str,
        /// Tile zoom.
        z: u32,
        /// Tile column.
        x: u32,
        /// Tile row.
        y: u32,
        /// Source message.
        message: String,
    },
    /// A tile is not a 256 × 256 image.
    #[error("{kind} tile {z}/{x}/{y} is not a valid 256 x 256 image")]
    Tile {
        /// Tile kind.
        kind: &'static str,
        /// Tile zoom.
        z: u32,
        /// Tile column.
        x: u32,
        /// Tile row.
        y: u32,
    },
    /// The area needs more memory than the request permits.
    #[error("the map area needs {needed} bytes; the limit is {limit}")]
    Memory {
        /// Bytes that the scene needs.
        needed: usize,
        /// Permitted bytes.
        limit: usize,
    },
    /// The terrain has no height at the target.
    #[error("the terrain has no height at the target")]
    NoTerrain,
    /// The reference renderer failed.
    #[error("rendering failed")]
    Render(#[from] navigate_visual_render::RenderError),
    /// The geometry or the matcher failed.
    #[error("localization failed")]
    Workflow(#[from] crate::WorkflowError),
    /// A geometry value was invalid.
    #[error("invalid geometry")]
    Visual(#[from] navigate_visual::VisualError),
    /// A place index file could not be read.
    #[error("place index file: {0}")]
    Index(String),
    /// A place index file or search request is not valid.
    #[error("place index")]
    Place(#[from] navigate_place::PlaceError),
    /// The place encoder failed.
    #[error("place encoder: {0}")]
    Encoder(String),
}
