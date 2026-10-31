//! Error types for `navigate-visual-pipeline`.

use navigate_segment::SegmentError;
use std::path::PathBuf;
use thiserror::Error;

/// Errors returned by visual pipeline stages and CLI commands.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Filesystem read or write failed.
    #[error("I/O failed for {}: {source}", path.display())]
    Io {
        /// File or directory path.
        path: PathBuf,
        /// Underlying OS error.
        #[source]
        source: std::io::Error,
    },
    /// JSON parsing or serialization failed.
    #[error("JSON failure in {context}: {source}")]
    Json {
        /// Description of the JSON document.
        context: String,
        /// Underlying parser error.
        #[source]
        source: serde_json::Error,
    },
    /// Image decoding or encoding failed.
    #[error("image failure for {}: {source}", path.display())]
    Image {
        /// Image path.
        path: PathBuf,
        /// Underlying decoder/encoder error.
        #[source]
        source: image::ImageError,
    },
    /// Semantic segmentation or `.npy`/`.npz` archive error.
    #[error("segmentation error: {0}")]
    Segment(#[from] SegmentError),
    /// Learned matcher error.
    #[error("matcher error: {0}")]
    Match(#[from] navigate_match::MatchError),
    /// Visual geometry error.
    #[error("visual geometry error: {0}")]
    Visual(#[from] navigate_visual::VisualError),
    /// GPU reference renderer error.
    #[error("terrain renderer error: {0}")]
    Render(#[from] navigate_visual_render::RenderError),
    /// External media tool (`ffmpeg`, `ffprobe`, `curl`, `magick`) failed.
    #[error("external tool '{tool}' failed: {message}")]
    Tool {
        /// Command executable name.
        tool: String,
        /// Captured error text or exit code.
        message: String,
    },
    /// Domain, provenance, or integrity check failed.
    #[error("pipeline validation failed: {0}")]
    Invalid(String),
}
