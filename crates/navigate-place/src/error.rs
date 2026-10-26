//! Errors of the place index and the search plan.

/// A place index or search request that this crate cannot use.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// A shard file has a wrong header or length.
    #[error("index shard {path}: {reason}")]
    Shard {
        /// Shard path in the manifest.
        path: String,
        /// What is wrong.
        reason: String,
    },
    /// The manifest JSON is not valid.
    #[error("index manifest: {0}")]
    Manifest(#[from] serde_json::Error),
    /// The search request is not valid.
    #[error("search request: {0}")]
    Request(String),
    /// A descriptor has a width that differs from the index or head width.
    #[error("descriptor width {found}, expected {expected}")]
    Width {
        /// Expected width.
        expected: usize,
        /// Received width.
        found: usize,
    },
}
