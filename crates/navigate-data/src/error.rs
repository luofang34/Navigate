//! Errors of the data contract.

use alloc::string::String;

/// A refusal of the data contract.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DataError {
    /// An input value is not usable.
    #[error("invalid {field}: {detail}")]
    Invalid {
        /// The input.
        field: &'static str,
        /// Why it is not usable.
        detail: String,
    },
    /// The store does not hold a file that the package needs.
    #[error("missing {path}")]
    Missing {
        /// Logical path of the file.
        path: String,
    },
    /// The store failed.
    #[error("store failed at {path}: {message}")]
    Store {
        /// Logical path of the file.
        path: String,
        /// The message of the store.
        message: String,
    },
    /// The manifest is not a package manifest of this version, or it is
    /// not in canonical form.
    #[error("manifest {path}: {detail}")]
    Manifest {
        /// Logical path of the manifest.
        path: String,
        /// What is wrong.
        detail: String,
    },
    /// A file does not agree with its digest or its size.
    #[error("integrity failure at {path}: {detail}")]
    Integrity {
        /// Logical path of the file.
        path: String,
        /// What does not agree.
        detail: String,
    },
    /// The fetch port failed.
    #[error("fetch of {key} failed: {message}")]
    Fetch {
        /// The tile key.
        key: String,
        /// The message of the port.
        message: String,
    },
}
