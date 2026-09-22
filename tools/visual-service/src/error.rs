#[derive(Debug, thiserror::Error)]
pub(crate) enum ServiceError {
    #[error("service I/O failed")]
    Io(#[from] std::io::Error),
    #[error("package operation failed")]
    Imagery(#[from] navigate_imagery::ImageryError),
    #[error("background initialization failed")]
    Join(#[from] tokio::task::JoinError),
    #[error("invalid public origin")]
    Origin {
        #[source]
        source: url::ParseError,
    },
    #[error("invalid service configuration: {0}")]
    Configuration(&'static str),
}
