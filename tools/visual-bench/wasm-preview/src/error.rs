use thiserror::Error;
#[derive(Debug, Error)]
pub(crate) enum PreviewError {
    #[error("visual estimate: {0}")]
    Visual(#[from] navigate_visual::VisualError),
    #[error("invalid preview input: {reason}")]
    Input { reason: String },
    #[error("preview JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("preview data: {0}")]
    Data(#[from] navigate_data::DataError),
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
