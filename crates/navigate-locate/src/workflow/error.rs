//! Workflow failures preserve the stage and the source error.
use navigate_visual::VisualError;
use thiserror::Error;

/// A visual workflow could not execute within its resource contract.
#[derive(Debug, Error)]
pub enum WorkflowError {
    /// A configured limit is invalid.
    #[error("invalid resource limit: {field}")]
    InvalidLimit {
        /// Limit name.
        field: &'static str,
    },
    /// A frame or reference would exceed a configured limit.
    #[error("{resource} needs {requested}, limit is {limit}")]
    ResourceLimit {
        /// Resource name.
        resource: &'static str,
        /// Required amount.
        requested: usize,
        /// Maximum amount.
        limit: usize,
    },
    /// A feature stage could not run.
    #[error("{stage}: {source}")]
    Backend {
        /// Operation name.
        stage: &'static str,
        /// Backend failure.
        #[source]
        source: VisualError,
    },
    /// The geometric policy or reference is invalid.
    #[error("geometry: {0}")]
    Geometry(#[from] VisualError),
}
