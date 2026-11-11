//! Opening a package and reading its tiles.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::manifest::Declared;
use crate::{DataError, Manifest, ReadPort, Sha256Digest, TileAddress, blob_path, manifest_path};

mod writer;
pub use writer::PackageWriter;

/// A package whose manifest has been checked.
#[derive(Clone, Debug, PartialEq)]
pub struct Package {
    manifest: Manifest,
    identity: Sha256Digest,
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

impl Package {
    /// Opens the package `identity` in `store`. The manifest bytes must have
    /// that digest and be in canonical form.
    ///
    /// # Errors
    ///
    /// Refuses a missing manifest, a digest that differs, and a manifest
    /// that [`Manifest::from_canonical`] refuses.
    pub async fn open<S: ReadPort + ?Sized>(
        store: &S,
        identity: Sha256Digest,
    ) -> Result<Self, DataError> {
        let path = manifest_path(identity);
        let bytes = store
            .read(&path)
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
        Ok(Self { manifest, identity })
    }

    /// The manifest.
    #[must_use]
    pub const fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The package identity: the digest of the canonical manifest. A
    /// visual fix names it as the `manifest_sha256` of its map revision.
    #[must_use]
    pub const fn identity(&self) -> Sha256Digest {
        self.identity
    }

    /// The release identifier.
    #[must_use]
    pub fn release_id(&self) -> &str {
        &self.manifest.release_id
    }

    /// Reads one tile of layer `layer_id` and checks its blob.
    ///
    /// # Errors
    ///
    /// Refuses an unknown layer, a blob that the store does not hold, and a
    /// blob whose size or digest differs from the manifest.
    pub async fn tile<S: ReadPort + ?Sized>(
        &self,
        store: &S,
        layer_id: &str,
        address: TileAddress,
    ) -> Result<TileRead, DataError> {
        let layer = self
            .manifest
            .layer(layer_id)
            .ok_or_else(|| DataError::Invalid {
                field: "layer",
                detail: String::from(layer_id),
            })?;
        match layer.declared(address) {
            Declared::Outside => Ok(TileRead::Outside),
            Declared::Gap => Ok(TileRead::Gap),
            Declared::Tile(tile) => read_blob(store, tile.blob, tile.bytes)
                .await
                .map(TileRead::Present),
        }
    }

    /// Reads and checks each blob once. Returns the count of blobs.
    ///
    /// # Errors
    ///
    /// Returns the first blob that is missing or does not agree.
    pub async fn verify_all<S: ReadPort + ?Sized>(&self, store: &S) -> Result<usize, DataError> {
        let blobs: BTreeSet<(Sha256Digest, u64)> = self.manifest.blobs().collect();
        for &(digest, bytes) in &blobs {
            read_blob(store, digest, bytes).await?;
        }
        Ok(blobs.len())
    }
}

async fn read_blob<S: ReadPort + ?Sized>(
    store: &S,
    digest: Sha256Digest,
    size: u64,
) -> Result<Vec<u8>, DataError> {
    let path = blob_path(digest);
    let bytes = store
        .read(&path)
        .await?
        .ok_or_else(|| DataError::Missing { path: path.clone() })?;
    if bytes.len() as u64 != size || Sha256Digest::of(&bytes) != digest {
        return Err(DataError::Integrity {
            path,
            detail: alloc::format!("{} bytes do not agree with the manifest", bytes.len()),
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests;
