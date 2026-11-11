//! Storage ports and the logical layout of a store.
//!
//! The layout follows the content-addressed store of Pilotage ADR-0044:
//! each blob, tile index included, is at `chunks/<sha256>`; each package
//! manifest is at `packages/<sha256>.json` with its descriptor at
//! `packages/<sha256>.descriptor.json`. Packages that cover the same ground
//! share blobs. The host owns the store, its root, and the deletion of
//! blobs that no installed document refers to; Navigate names only these
//! logical paths and never deletes.
//!
//! [`ReadPort`] has the shape of the ADR-0044 read port: asynchronous
//! `open` of a logical path, then [`RandomAccess::len`] and
//! [`RandomAccess::read_at`] of a byte range.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;

use crate::{DataError, Sha256Digest};

#[cfg(feature = "std")]
mod directory;
#[cfg(feature = "std")]
pub use directory::{DirectoryFile, DirectoryStore};

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

/// Logical path of the descriptor of a package.
#[must_use]
pub fn descriptor_path(identity: Sha256Digest) -> String {
    alloc::format!("packages/{identity}.descriptor.json")
}

/// An open file of a store with a fixed length.
pub trait RandomAccess {
    /// Length of the file in bytes.
    fn len(&self) -> u64;

    /// Whether the file has no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads exactly `length` bytes at `offset`. A short read is an error.
    fn read_at(
        &self,
        offset: u64,
        length: usize,
    ) -> impl Future<Output = Result<Vec<u8>, DataError>>;
}

/// Read access to a store. A browser host can implement it on OPFS; a
/// native host on a directory.
pub trait ReadPort {
    /// The open file type.
    type File: RandomAccess;

    /// Opens the file at a logical path. `Ok(None)` marks a file that the
    /// store does not hold.
    fn open(&self, path: &str) -> impl Future<Output = Result<Option<Self::File>, DataError>>;
}

/// Write access to a store, for the host and for builders.
pub trait WritePort {
    /// Writes a file. Each path holds content that never changes, so a
    /// second write of the same path is accepted and has no effect.
    fn write(&mut self, path: &str, bytes: &[u8]) -> impl Future<Output = Result<(), DataError>>;

    /// Whether the store holds a file at the path.
    fn contains(&self, path: &str) -> impl Future<Output = Result<bool, DataError>>;
}

/// Reads a whole file, or `None` when the store does not hold it.
///
/// # Errors
///
/// Returns the store refusals.
pub async fn read_all<S: ReadPort + ?Sized>(
    store: &S,
    path: &str,
) -> Result<Option<Vec<u8>>, DataError> {
    let Some(file) = store.open(path).await? else {
        return Ok(None);
    };
    let length = usize::try_from(file.len()).map_err(|_| DataError::Store {
        path: String::from(path),
        message: String::from("file too large for this target"),
    })?;
    file.read_at(0, length).await.map(Some)
}

/// A store in memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryStore {
    files: BTreeMap<String, Arc<[u8]>>,
}

/// An open file of a [`MemoryStore`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryFile(Arc<[u8]>);

impl RandomAccess for MemoryFile {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    async fn read_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, DataError> {
        let start = usize::try_from(offset).ok();
        start
            .and_then(|s| self.0.get(s..s.checked_add(length)?))
            .map(<[u8]>::to_vec)
            .ok_or_else(|| DataError::Store {
                path: String::from("<memory>"),
                message: alloc::format!("range {offset}+{length} past {} bytes", self.0.len()),
            })
    }
}

impl MemoryStore {
    /// The bytes at a path.
    #[must_use]
    pub fn get(&self, path: &str) -> Option<&[u8]> {
        self.files.get(path).map(|b| &**b)
    }

    /// The paths that the store holds, in order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }

    /// Replaces the file at a path. A host uses it to repair a file; a test
    /// uses it to change one.
    pub fn replace(&mut self, path: &str, bytes: &[u8]) {
        self.files.insert(String::from(path), Arc::from(bytes));
    }

    /// Removes the file at a path.
    pub fn remove(&mut self, path: &str) -> bool {
        self.files.remove(path).is_some()
    }
}

impl ReadPort for MemoryStore {
    type File = MemoryFile;

    async fn open(&self, path: &str) -> Result<Option<MemoryFile>, DataError> {
        checked_path(path)?;
        Ok(self.files.get(path).cloned().map(MemoryFile))
    }
}

impl WritePort for MemoryStore {
    async fn write(&mut self, path: &str, bytes: &[u8]) -> Result<(), DataError> {
        checked_path(path)?;
        self.files
            .entry(String::from(path))
            .or_insert_with(|| Arc::from(bytes));
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
