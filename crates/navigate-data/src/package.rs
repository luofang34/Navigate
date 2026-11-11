//! Opening a package and reading its tiles.

use alloc::string::String;
use alloc::vec::Vec;

use alloc::collections::BTreeSet;

use crate::index::{IndexEntry, entries, find, verify_index};
use crate::store::read_all;
use crate::{
    DataError, Descriptor, LayerKey, Manifest, RandomAccess, ReadPort, Sha256Digest, TileAddress,
    blob_path, descriptor_path, manifest_path,
};

mod writer;
pub use writer::{LayerSpec, PackageWriter};

/// A package whose manifest and tile indexes have been checked.
pub struct Package<F> {
    manifest: Manifest,
    identity: Sha256Digest,
    /// The open index of each layer, in manifest order.
    indexes: Vec<F>,
}

/// The answer of a package for one tile of a layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileRead {
    /// The tile bytes, checked against their digest and size.
    Present(Vec<u8>),
    /// The address is in the declared coverage, but the source did not have
    /// the tile.
    Gap,
    /// The address is outside the declared coverage.
    Outside,
}

impl<F: RandomAccess> Package<F> {
    /// Opens the package `identity` in `store`: the manifest must have that
    /// digest and be canonical, and each tile index must agree with the
    /// manifest.
    ///
    /// # Errors
    ///
    /// Refuses a missing or changed manifest, a manifest that
    /// [`Manifest::from_canonical`] refuses, and a missing or changed index.
    pub async fn open<S: ReadPort<File = F> + ?Sized>(
        store: &S,
        identity: Sha256Digest,
    ) -> Result<Self, DataError> {
        let path = manifest_path(identity);
        let bytes = read_all(store, &path)
            .await?
            .ok_or_else(|| DataError::Missing { path: path.clone() })?;
        let actual = Sha256Digest::of(&bytes);
        if actual != identity {
            return Err(DataError::Integrity {
                path,
                detail: alloc::format!("digest is {actual}"),
            });
        }
        let manifest = Manifest::from_canonical(&bytes, &path)?;
        let mut indexes = Vec::with_capacity(manifest.layers.len());
        for layer in &manifest.layers {
            let index_path = blob_path(layer.index.blob);
            let file = store
                .open(&index_path)
                .await?
                .ok_or_else(|| DataError::Missing {
                    path: index_path.clone(),
                })?;
            let counts = verify_index(&file, layer.index.blob, &index_path).await?;
            if counts.tiles != layer.index.tiles || counts.gaps != layer.index.gaps {
                return Err(DataError::Integrity {
                    path: index_path,
                    detail: String::from("counts do not agree with the manifest"),
                });
            }
            indexes.push(file);
        }
        Ok(Self {
            manifest,
            identity,
            indexes,
        })
    }

    /// The manifest.
    #[must_use]
    pub const fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The package identity: the digest of the canonical manifest. A visual
    /// fix names it as the `manifest_sha256` of its map revision.
    #[must_use]
    pub const fn identity(&self) -> Sha256Digest {
        self.identity
    }

    /// The index entry of one address of a layer.
    ///
    /// # Errors
    ///
    /// Refuses a layer that the package does not hold and an index read that
    /// fails.
    pub async fn entry(
        &self,
        layer: LayerKey,
        address: TileAddress,
    ) -> Result<Option<IndexEntry>, DataError> {
        let position = self
            .manifest
            .layers
            .iter()
            .position(|l| l.key == layer)
            .ok_or_else(|| DataError::Invalid {
                field: "layer",
                detail: alloc::format!("{layer:?}"),
            })?;
        let (Some(file), Some(spec)) = (
            self.indexes.get(position),
            self.manifest.layers.get(position),
        ) else {
            return Err(DataError::Invalid {
                field: "layer",
                detail: alloc::format!("{layer:?}"),
            });
        };
        find(
            file,
            spec.index.records(),
            address,
            &blob_path(spec.index.blob),
        )
        .await
    }

    /// Reads one tile of a layer and checks its blob.
    ///
    /// # Errors
    ///
    /// Refuses a layer that the package does not hold, a blob that the
    /// store does not hold, and a blob whose size or digest differs.
    pub async fn tile<S: ReadPort + ?Sized>(
        &self,
        store: &S,
        layer: LayerKey,
        address: TileAddress,
    ) -> Result<TileRead, DataError> {
        match self.entry(layer, address).await? {
            None => Ok(TileRead::Outside),
            Some(IndexEntry::Gap) => Ok(TileRead::Gap),
            Some(IndexEntry::Tile { blob, bytes }) => {
                read_blob(store, blob, bytes).await.map(TileRead::Present)
            }
        }
    }

    /// Every record of the index of a layer, in index order.
    ///
    /// # Errors
    ///
    /// Refuses a layer that the package does not hold and an index read that
    /// fails.
    pub async fn layer_entries(
        &self,
        layer: LayerKey,
    ) -> Result<Vec<(TileAddress, IndexEntry)>, DataError> {
        let position = self
            .manifest
            .layers
            .iter()
            .position(|l| l.key == layer)
            .ok_or_else(|| DataError::Invalid {
                field: "layer",
                detail: alloc::format!("{layer:?}"),
            })?;
        let (Some(file), Some(spec)) = (
            self.indexes.get(position),
            self.manifest.layers.get(position),
        ) else {
            return Err(DataError::Invalid {
                field: "layer",
                detail: alloc::format!("{layer:?}"),
            });
        };
        entries(file, spec.index.records(), &blob_path(spec.index.blob)).await
    }

    /// The digests of every tile blob of the package.
    ///
    /// # Errors
    ///
    /// Returns index read failures.
    pub async fn blobs(&self) -> Result<BTreeSet<Sha256Digest>, DataError> {
        let mut out = BTreeSet::new();
        for layer in &self.manifest.layers {
            for (_, entry) in self.layer_entries(layer.key).await? {
                if let IndexEntry::Tile { blob, .. } = entry {
                    out.insert(blob);
                }
            }
        }
        Ok(out)
    }

    /// Reads and checks each tile blob once. Returns the count of distinct
    /// blobs.
    ///
    /// # Errors
    ///
    /// Returns the first blob that is missing or does not agree.
    pub async fn verify_all<S: ReadPort + ?Sized>(&self, store: &S) -> Result<usize, DataError> {
        let mut seen = BTreeSet::new();
        for layer in &self.manifest.layers {
            for (_, entry) in self.layer_entries(layer.key).await? {
                if let IndexEntry::Tile { blob, bytes } = entry
                    && seen.insert(blob)
                {
                    read_blob(store, blob, bytes).await?;
                }
            }
        }
        Ok(seen.len())
    }

    /// The descriptor, if the store holds one.
    ///
    /// # Errors
    ///
    /// Refuses a descriptor that does not decode.
    pub async fn descriptor<S: ReadPort + ?Sized>(
        &self,
        store: &S,
    ) -> Result<Option<Descriptor>, DataError> {
        let path = descriptor_path(self.identity);
        let Some(bytes) = read_all(store, &path).await? else {
            return Ok(None);
        };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|error| DataError::Document {
                path,
                detail: alloc::format!("{error}"),
            })
    }
}

/// Reads one blob and checks its size and digest.
///
/// # Errors
///
/// Refuses a missing blob and a blob that does not agree.
pub async fn read_blob<S: ReadPort + ?Sized>(
    store: &S,
    digest: Sha256Digest,
    size: u64,
) -> Result<Vec<u8>, DataError> {
    let path = blob_path(digest);
    let bytes = read_all(store, &path)
        .await?
        .ok_or_else(|| DataError::Missing { path: path.clone() })?;
    if bytes.len() as u64 != size || Sha256Digest::of(&bytes) != digest {
        return Err(DataError::Integrity {
            path,
            detail: alloc::format!("{} bytes do not agree with the index", bytes.len()),
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests;
