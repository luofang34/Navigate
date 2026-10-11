//! Contextual failures while decoding and reconstructing recorded evidence.

use std::path::PathBuf;

/// Failure to read or process a recorded observation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A recording asset could not be read.
    #[error("cannot read recording asset {path}")]
    Io {
        /// Asset path.
        path: PathBuf,
        /// Underlying IO failure.
        #[source]
        source: std::io::Error,
    },
    /// Image decoding failed.
    #[error("cannot decode observation image {path}")]
    Image {
        /// Image path.
        path: PathBuf,
        /// Decoder failure.
        #[source]
        source: image::ImageError,
    },
    /// The input does not describe usable evidence.
    #[error("invalid recording: {0}")]
    Invalid(String),
    /// The visual geometry rejected malformed input.
    #[error("visual geometry failed")]
    Visual(#[from] navigate_visual::VisualError),
    /// A session identity or evidence contract was violated.
    #[error("visual session failed")]
    Session(#[from] navigate_visual_session::SessionError),
}
