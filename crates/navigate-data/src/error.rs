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
    /// A document of another format version. There is one current schema
    /// and no reader for other versions.
    #[error("{document} version {found} is not version {supported}")]
    Version {
        /// The document kind, for example `package manifest`.
        document: &'static str,
        /// The version in the document.
        found: u32,
        /// The version that this crate reads.
        supported: u32,
    },
    /// A value that this crate does not know, in a place where an unknown
    /// value changes the meaning of the data: a body, grid scheme, layer
    /// kind, encoding, datum, geoid model, or source.
    #[error("unsupported {field}")]
    Unsupported {
        /// The field.
        field: &'static str,
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
    /// A document does not decode, or is not in canonical form.
    #[error("document {path}: {detail}")]
    Document {
        /// Logical path of the document.
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
