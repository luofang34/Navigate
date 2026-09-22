//! Byte-range access independent of the host storage API.
use crate::DataUri;
use std::{future::Future, pin::Pin};
use thiserror::Error;
/// Storage failure with the resource and original backend error.
#[derive(Debug, Error)]
pub enum DataError {
    /// The resource name is not a valid logical URI.
    #[error("invalid data URI {uri}")]
    InvalidUri {
        /// Resource name supplied by the caller.
        uri: String,
    },
    /// The requested byte interval leaves the resource.
    #[error("range {offset}+{length} exceeds {size} bytes in {uri}")]
    Range {
        /// Resource name.
        uri: String,
        /// First byte requested.
        offset: u64,
        /// Requested byte count.
        length: usize,
        /// Resource size.
        size: u64,
    },
    /// The host storage backend failed.
    #[error("{operation} failed for {uri}: {source}")]
    Backend {
        /// Resource name.
        uri: String,
        /// Failed storage operation.
        operation: &'static str,
        /// Original backend error.
        #[source]
        source: Box<dyn std::error::Error>,
    },
}
/// Pending byte-range read.
pub type ReadFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<u8>, DataError>> + 'a>>;
/// Pending resource open.
pub type OpenFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Box<dyn RandomAccess>, DataError>> + 'a>>;
/// A versioned resource with stable length and byte addressing.
pub trait RandomAccess {
    /// Size of the open resource in bytes.
    fn len(&self) -> u64;
    /// Whether the resource has no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Read an exact interval. A short read is an error.
    fn read_at(&self, offset: u64, length: usize) -> ReadFuture<'_>;
}
/// Resolve logical resource names through the selected host adapter.
pub trait DataStore {
    /// Open a resource without exposing its platform storage path.
    fn open<'a>(&'a self, uri: &'a DataUri) -> OpenFuture<'a>;
}
/// Check an exact byte range before the host performs I/O.
///
/// # Errors
/// Rejects overflowing ranges and reads beyond the resource size.
pub fn check_range(uri: &DataUri, size: u64, offset: u64, length: usize) -> Result<(), DataError> {
    if offset
        .checked_add(length as u64)
        .is_none_or(|end| end > size)
    {
        return Err(DataError::Range {
            uri: uri.as_str().into(),
            offset,
            length,
            size,
        });
    }
    Ok(())
}
#[cfg(test)]
mod tests;
