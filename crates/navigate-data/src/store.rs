//! Storage ports and the logical layout of a store.
//!
//! The layout follows the content-addressed store of Pilotage ADR-0044:
//! each blob is at `chunks/<sha256>`, and each manifest is at
//! `packages/<sha256>.json`. Packages that cover the same ground share
//! blobs. The host owns the store and its root; Navigate names only these
//! logical paths.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;

use crate::{DataError, Sha256Digest};

#[cfg(feature = "std")]
mod directory;
#[cfg(feature = "std")]
pub use directory::DirectoryStore;

/// Logical path of a blob.
#[must_use]
pub fn blob_path(digest: Sha256Digest) -> String {
    alloc::format!("chunks/{digest}")
}

/// Logical path of a package manifest.
#[must_use]
pub fn manifest_path(identity: Sha256Digest) -> String {
    alloc::format!("packages/{identity}.json")
}

/// Read access to a store. A browser host can implement it on OPFS; a
/// native host on a directory. `Ok(None)` marks a file that the store does
/// not hold.
pub trait ReadPort {
    /// The bytes at a logical path.
    fn read(&self, path: &str) -> impl Future<Output = Result<Option<Vec<u8>>, DataError>>;
}

/// Write access to a store, for the host and for builders.
pub trait WritePort {
    /// Writes a file. Each path holds content that never changes, so a
    /// second write of the same path is accepted and has no effect.
    fn write(&mut self, path: &str, bytes: &[u8]) -> impl Future<Output = Result<(), DataError>>;

    /// Whether the store holds a file at the path.
    fn contains(&self, path: &str) -> impl Future<Output = Result<bool, DataError>>;
}

/// A store in memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryStore {
    /// The files, by logical path.
    pub files: BTreeMap<String, Vec<u8>>,
}

impl ReadPort for MemoryStore {
    async fn read(&self, path: &str) -> Result<Option<Vec<u8>>, DataError> {
        checked_path(path)?;
        Ok(self.files.get(path).cloned())
    }
}

impl WritePort for MemoryStore {
    async fn write(&mut self, path: &str, bytes: &[u8]) -> Result<(), DataError> {
        checked_path(path)?;
        self.files
            .entry(String::from(path))
            .or_insert_with(|| bytes.to_vec());
        Ok(())
    }

    async fn contains(&self, path: &str) -> Result<bool, DataError> {
        checked_path(path)?;
        Ok(self.files.contains_key(path))
    }
}

/// Refuses a logical path that is empty, absolute, or that has an empty,
/// `.`, or `..` part or a `\` separator.
///
/// # Errors
///
/// Returns [`DataError::Invalid`] for such a path.
pub fn checked_path(path: &str) -> Result<&str, DataError> {
    let unsafe_part = |part: &str| part.is_empty() || part == "." || part == "..";
    if path.is_empty() || path.contains('\\') || path.split('/').any(unsafe_part) {
        return Err(DataError::Invalid {
            field: "path",
            detail: String::from(path),
        });
    }
    Ok(path)
}

#[cfg(test)]
mod tests;
