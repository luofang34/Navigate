//! Errors retain input paths and underlying failures.
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("decode feature packet {path}: {source}")]
    CatalogPacket {
        path: PathBuf,
        #[source]
        source: navigate_visual_onnx::InferenceError,
    },
    #[error("validate feature image {path}: {source}")]
    CatalogImage {
        path: PathBuf,
        #[source]
        source: navigate_visual::VisualError,
    },
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
    #[error("refine scene {path}: {source}")]
    Scene {
        path: PathBuf,
        #[source]
        source: navigate_visual::LocalSceneError,
    },
    #[error(transparent)]
    Cli(#[from] clap::Error),
    #[error(transparent)]
    Inference(#[from] navigate_visual_onnx::InferenceError),
    #[error(transparent)]
    Session(#[from] navigate_visual_session::SessionError),
    #[error(transparent)]
    Visual(#[from] navigate_visual::VisualError),
    #[error(transparent)]
    Workflow(#[from] navigate_visual_workflow::WorkflowError),
    #[error(transparent)]
    Recording(#[from] navigate_recording::Error),
    #[error(transparent)]
    Render(#[from] navigate_visual_render::RenderError),
}
