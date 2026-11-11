//! A package copied from a store into memory, for a reader that cannot
//! wait on its store.
//!
//! A navigator step is not asynchronous: it reads its map by blocking on
//! each read, which only works when a read completes without help from the
//! calling thread. A directory store and a [`MemoryStore`] do; a browser
//! store, whose reads complete on the page's event loop, does not, and
//! blocking on it there never returns. A host with such a store awaits
//! [`resident_copy`] once and gives the navigator the memory store.

use alloc::string::String;

use crate::index::IndexEntry;
use crate::product::{chunk_path, list_path, open_chunk, open_list};
use crate::{
    DataError, MemoryStore, Package, ReadPort, Sha256Digest, blob_path, descriptor_path,
    manifest_path, read_all,
};

#[cfg(test)]
mod tests;

/// A package in memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentPackage {
    /// Every file of the package at its path in the source store.
    pub store: MemoryStore,
    /// Bytes of those files.
    pub bytes: u64,
}

/// Copies the package `identity` from `store` into memory: the manifest,
/// the descriptor, the tile indexes and tiles, and with `products` that
/// product list, its chunks, their recipes and entries.
///
/// The copy holds the same bytes at the same paths as the source, and a
/// file the source lacks is lacking in the copy too, so a reader of the
/// copy accepts and refuses what it would on the source. The manifest and
/// the indexes are checked here; tiles and products are checked when they
/// are read, as on any store.
///
/// # Errors
///
/// Refuses a package that does not open on `store`, and a failed read.
pub async fn resident_copy<S: ReadPort + ?Sized>(
    store: &S,
    identity: Sha256Digest,
    products: Option<Sha256Digest>,
) -> Result<ResidentPackage, DataError> {
    let package = Package::open(store, identity).await?;
    let mut copy = Copy {
        store,
        resident: ResidentPackage {
            store: MemoryStore::default(),
            bytes: 0,
        },
    };
    copy.path(manifest_path(identity)).await?;
    copy.path(descriptor_path(identity)).await?;
    for layer in &package.manifest().layers {
        copy.path(blob_path(layer.index.blob)).await?;
        for (_, entry) in package.layer_entries(layer.key).await? {
            if let IndexEntry::Tile { blob, .. } = entry {
                copy.path(blob_path(blob)).await?;
            }
        }
    }
    if let Some(list) = products {
        copy.products(identity, list).await?;
    }
    Ok(copy.resident)
}

struct Copy<'a, S: ?Sized> {
    store: &'a S,
    resident: ResidentPackage,
}

impl<S: ReadPort + ?Sized> Copy<'_, S> {
    /// Copies the file at `path`, once, if the source holds it.
    async fn path(&mut self, path: String) -> Result<(), DataError> {
        if self.resident.store.get(&path).is_some() {
            return Ok(());
        }
        if let Some(bytes) = read_all(self.store, &path).await? {
            self.resident.bytes = self.resident.bytes.saturating_add(bytes.len() as u64);
            self.resident.store.replace(&path, &bytes);
        }
        Ok(())
    }

    /// Copies the product list `list` and what it names. A list or chunk
    /// that does not open is copied as it is and not followed: the reader
    /// refuses it on the copy as on the source. A failed store read is an
    /// error.
    async fn products(
        &mut self,
        package: Sha256Digest,
        list: Sha256Digest,
    ) -> Result<(), DataError> {
        self.path(list_path(list)).await?;
        let opened = match open_list(self.store, list, package).await {
            Ok(opened) => opened,
            Err(error @ DataError::Store { .. }) => return Err(error),
            Err(_) => return Ok(()),
        };
        for entry in &opened.chunks {
            self.path(chunk_path(entry.chunk)).await?;
            let chunk = match open_chunk(self.store, entry.chunk).await {
                Ok(chunk) => chunk,
                Err(error @ DataError::Store { .. }) => return Err(error),
                Err(_) => continue,
            };
            self.path(blob_path(chunk.recipe)).await?;
            for item in &chunk.entries {
                self.path(blob_path(item.blob)).await?;
            }
        }
        Ok(())
    }
}
