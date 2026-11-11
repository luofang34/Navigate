//! Error types for the live flight visual localization engine.

use std::path::PathBuf;
use thiserror::Error;

/// Error returned by [`crate::FlightVisualEngine`] and its runtime builders.
#[derive(Debug, Error)]
pub enum FlightVisualError {
    /// The surface lacks a valid local tangent frame.
    #[error("surface lacks a valid local tangent frame")]
    MissingLocalFrame,

    /// An incoming frame timestamp violated monotonic ordering.
    #[error("non-monotonic frame timestamp: {0}")]
    NonMonotonicTimestamp(String),

    /// A background worker stopped or faulted.
    #[error("worker `{worker}` faulted: {message}")]
    WorkerFault {
        /// Worker thread name.
        worker: String,
        /// Diagnostic failure message.
        message: String,
    },

    /// A navigation data package or product refused a read or a write.
    #[error("navigation data: {0}")]
    Data(#[from] navigate_data::DataError),

    /// Input parameters or map contracts failed validation.
    #[error("{0}")]
    Invalid(String),

    /// Filesystem read or write failed during startup or bank build.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Target file or directory path.
        path: PathBuf,
        /// Underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// JSON parse or serialization failed during catalog loading.
    #[error("JSON error ({context}): {source}")]
    Json {
        /// Context string describing the JSON document.
        context: String,
        /// Underlying parser error.
        #[source]
        source: serde_json::Error,
    },

    /// Tile image decoding failed during catalog loading.
    #[error("image error at {path}: {source}")]
    Image {
        /// Path to the tile image.
        path: PathBuf,
        /// Underlying image decoder error.
        #[source]
        source: image::ImageError,
    },

    /// Sparse feature extraction or matching failed.
    #[error("match error: {0}")]
    Match(#[from] navigate_match::MatchError),

    /// Semantic segmentation or mask composition failed.
    #[error("segment error: {0}")]
    Segment(#[from] navigate_match::segment::SegmentError),

    /// Camera geometry or visual pose validation failed.
    #[error("visual geometry error: {0}")]
    Visual(#[from] navigate_visual::VisualError),

    /// GPU terrain rendering failed.
    #[error("GPU terrain render error: {0}")]
    Render(#[from] navigate_visual_render::RenderError),
}
