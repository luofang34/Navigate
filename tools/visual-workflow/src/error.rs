//! Errors retain input paths and underlying failures.
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{operation} {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("decode image {path}: {source}")]
    Image {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("invalid workflow input: {0}")]
    Input(String),
    #[error(transparent)]
    Cli(#[from] clap::Error),
    #[error(transparent)]
    Inference(#[from] navigate_visual_onnx::InferenceError),
    #[error(transparent)]
    Visual(#[from] navigate_visual::VisualError),
    #[error(transparent)]
    Workflow(#[from] navigate_visual_workflow::WorkflowError),
    #[error(transparent)]
    Recording(#[from] navigate_recording::Error),
}
