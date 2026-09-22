//! Errors with source context for the visual command line.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum BenchError {
    #[error("invalid video presentation time {value:?} at frame {frame}")]
    Timestamp {
        frame: usize,
        value: String,
        #[source]
        source: std::num::ParseFloatError,
    },
    #[error("invalid command arguments")]
    Arguments(#[from] clap::Error),
    #[error("invalid record: {reason}")]
    Record { reason: String },
    #[error("cannot start {program}")]
    Spawn {
        program: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("{program} failed with {status}: {stderr}")]
    Process {
        program: &'static str,
        status: std::process::ExitStatus,
        stderr: String,
    },
    #[error("frame stream is empty")]
    EmptyStream,
    #[error("camera calibration changed within the frame stream")]
    ChangedCamera,
    #[error("cannot access {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON in {path}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("cannot decode or save image {path}")]
    Image {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("invalid map package: {reason}")]
    Package { reason: String },
    #[error("MapLibre rendering failed")]
    Render {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("GPU readback failed")]
    Readback {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("MapLibre has no color texture")]
    MissingTexture,
    #[error("visual observation failed")]
    Visual(#[from] navigate_visual::VisualError),
    #[error("synthetic acceptance failed for {failed} cases")]
    Evaluation { failed: usize },
}
