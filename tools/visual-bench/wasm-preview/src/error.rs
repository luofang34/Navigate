use thiserror::Error;
#[derive(Debug, Error)]
pub(crate) enum PreviewError {
    #[error("image track merge: {0}")]
    TrackMerge(#[from] navigate_visual::reconstruction::TrackMergeError),
    #[error("scene camera fit: {0}")]
    SceneResection(#[from] navigate_visual::reconstruction::SceneResectionError),
    #[error("scene registration: {0}")]
    SceneRegistration(#[from] navigate_visual::reconstruction::SceneRegistrationError),
    #[error("scene alignment: {0}")]
    SceneAlignment(#[from] navigate_visual::reconstruction::SceneAlignmentError),
    #[error("scene reconstruction: {0}")]
    Reconstruction(#[from] navigate_visual::reconstruction::ReconstructionError),
    #[error("local scene: {0}")]
    LocalScene(#[from] navigate_visual::LocalSceneError),
    #[error("invalid scene point ID {id}: {source}")]
    ScenePointId {
        id: String,
        #[source]
        source: std::num::ParseIntError,
    },
    #[error("visual estimate: {0}")]
    Visual(#[from] navigate_visual::VisualError),
    #[error("invalid preview input: {reason}")]
    Input { reason: String },
    #[error("preview JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("preview data: {0}")]
    Data(#[from] navigate_data::DataError),
    #[error("preview package: {0}")]
    Imagery(#[from] navigate_imagery::ImageryError),
    #[error("decode tile {uri}: {source}")]
    Image {
        uri: String,
        #[source]
        source: image::ImageError,
    },
    #[error("{operation}: {source}")]
    Renderer {
        operation: &'static str,
        #[source]
        source: Box<dyn std::error::Error>,
    },
}
pub(crate) fn render_error(
    operation: &'static str,
    source: impl std::error::Error + 'static,
) -> PreviewError {
    PreviewError::Renderer {
        operation,
        source: Box::new(source),
    }
}
impl From<PreviewError> for wasm_bindgen::JsValue {
    fn from(value: PreviewError) -> Self {
        let mut message = value.to_string();
        let mut source = std::error::Error::source(&value);
        while let Some(error) = source {
            message.push_str(&format!(": {error}"));
            source = error.source();
        }
        Self::from_str(&message)
    }
}
