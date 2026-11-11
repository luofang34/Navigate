//! Product chunks and product lists as canonical documents.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::product::{
    CHUNK_FORMAT, ChunkBinding, ChunkEntry, KindName, LIST_FORMAT, PRODUCT_FORMAT_VERSION,
    RECIPE_FORMAT, Recipe, canonical,
};
use crate::store::read_all;
use crate::{DataError, ReadPort, Sha256Digest, blob_path, read_blob};

/// Logical path of a product chunk.
#[must_use]
pub fn chunk_path(digest: Sha256Digest) -> String {
    alloc::format!("products/chunks/{digest}.json")
}

/// Logical path of a product list.
#[must_use]
pub fn list_path(digest: Sha256Digest) -> String {
    alloc::format!("products/lists/{digest}.json")
}

/// One product chunk.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductChunk {
    /// [`CHUNK_FORMAT`].
    pub format: String,
    /// [`PRODUCT_FORMAT_VERSION`].
    pub version: u32,
    /// What the chunk holds.
    pub kind: KindName,
    /// Digest of the recipe document, a blob in `chunks/`.
    pub recipe: Sha256Digest,
    /// What the chunk is for.
    pub binding: ChunkBinding,
    /// The entries, in key order.
    pub entries: Vec<ChunkEntry>,
}

/// One chunk of a product list.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ListEntry {
    /// The kind of the chunk.
    pub kind: KindName,
    /// The chunk digest.
    pub chunk: Sha256Digest,
}

/// The products of one mission package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductList {
    /// [`LIST_FORMAT`].
    pub format: String,
    /// [`PRODUCT_FORMAT_VERSION`].
    pub version: u32,
    /// The package identity.
    pub package: Sha256Digest,
    /// The chunks, in (kind, chunk) order.
    pub chunks: Vec<ListEntry>,
}

fn check_version(
    document: &'static str,
    format: &str,
    expected: &str,
    version: u32,
) -> Result<(), DataError> {
    if format != expected {
        return Err(DataError::Invalid {
            field: "format",
            detail: String::from(format),
        });
    }
    if version != PRODUCT_FORMAT_VERSION {
        return Err(DataError::Version {
            document,
            found: version,
            supported: PRODUCT_FORMAT_VERSION,
        });
    }
    Ok(())
}

impl ProductChunk {
    /// Checks the format, the version, and the order.
    ///
    /// # Errors
    ///
    /// Refuses another format or version, entries out of key order, and
    /// inputs that are not sorted and unique.
    pub fn validate(&self) -> Result<(), DataError> {
        check_version("product chunk", &self.format, CHUNK_FORMAT, self.version)?;
        let sorted = |ok: bool, field| {
            if ok {
                Ok(())
            } else {
                Err(DataError::Invalid {
                    field,
                    detail: String::from("not in strict order"),
                })
            }
        };
        sorted(
            self.entries
                .windows(2)
                .all(|w| matches!(w, [a, b] if a.key < b.key)),
            "entries",
        )?;
        sorted(
            self.binding
                .inputs
                .windows(2)
                .all(|w| matches!(w, [a, b] if a < b)),
            "inputs",
        )
    }

    /// The canonical bytes and the chunk digest.
    ///
    /// # Errors
    ///
    /// Refuses a chunk that [`Self::validate`] refuses.
    pub fn canonical(&self) -> Result<(Vec<u8>, Sha256Digest), DataError> {
        self.validate()?;
        let bytes = canonical(self, "<chunk>")?;
        let digest = Sha256Digest::of(&bytes);
        Ok((bytes, digest))
    }

    /// Whether a package that holds `blobs` holds each input of the chunk.
    #[must_use]
    pub fn applies_to(&self, blobs: &BTreeSet<Sha256Digest>) -> bool {
        self.binding
            .inputs
            .iter()
            .all(|input| blobs.contains(input))
    }

    /// Reads the blob of one entry and checks it.
    ///
    /// # Errors
    ///
    /// Refuses a missing or changed blob.
    pub async fn entry_bytes<S: ReadPort + ?Sized>(
        &self,
        store: &S,
        entry: &ChunkEntry,
    ) -> Result<Vec<u8>, DataError> {
        read_blob(store, entry.blob, entry.bytes).await
    }
}

impl ProductList {
    /// The canonical bytes and the list digest.
    ///
    /// # Errors
    ///
    /// Refuses another format or version and chunks out of order.
    pub fn canonical(&self) -> Result<(Vec<u8>, Sha256Digest), DataError> {
        check_version("product list", &self.format, LIST_FORMAT, self.version)?;
        if !self.chunks.windows(2).all(|w| matches!(w, [a, b] if a < b)) {
            return Err(DataError::Invalid {
                field: "chunks",
                detail: String::from("not in strict order"),
            });
        }
        let bytes = canonical(self, "<list>")?;
        let digest = Sha256Digest::of(&bytes);
        Ok((bytes, digest))
    }

    /// The chunks of kinds that this crate does not know: carried, never
    /// read.
    pub fn unknown_kinds(&self) -> impl Iterator<Item = &ListEntry> {
        self.chunks
            .iter()
            .filter(|c| matches!(c.kind, KindName::Unknown(_)))
    }
}

/// Reads a canonical document, checks its digest and its canonical form.
async fn read_document<S, T>(
    store: &S,
    path: String,
    digest: Sha256Digest,
) -> Result<(T, Vec<u8>), DataError>
where
    S: ReadPort + ?Sized,
    T: Serialize + for<'de> Deserialize<'de>,
{
    let bytes = read_all(store, &path)
        .await?
        .ok_or_else(|| DataError::Missing { path: path.clone() })?;
    if Sha256Digest::of(&bytes) != digest {
        return Err(DataError::Integrity {
            path,
            detail: String::from("digest"),
        });
    }
    let document: T = serde_json::from_slice(&bytes).map_err(|error| DataError::Document {
        path: path.clone(),
        detail: alloc::format!("{error}"),
    })?;
    if canonical(&document, &path)? != bytes {
        return Err(DataError::Document {
            path,
            detail: String::from("the bytes are not in canonical form"),
        });
    }
    Ok((document, bytes))
}

/// Opens one chunk of a known kind.
///
/// # Errors
///
/// Refuses a missing, changed, or non-canonical chunk, another version, and
/// a kind that this crate does not know.
pub async fn open_chunk<S: ReadPort + ?Sized>(
    store: &S,
    digest: Sha256Digest,
) -> Result<ProductChunk, DataError> {
    let (chunk, _): (ProductChunk, _) = read_document(store, chunk_path(digest), digest).await?;
    chunk.validate()?;
    if matches!(chunk.kind, KindName::Unknown(_)) {
        return Err(DataError::Unsupported {
            field: "product kind",
        });
    }
    Ok(chunk)
}

/// Opens the product list `digest` of the package `package`.
///
/// # Errors
///
/// Refuses a missing, changed, or non-canonical list, another version, and
/// a list of another package.
pub async fn open_list<S: ReadPort + ?Sized>(
    store: &S,
    digest: Sha256Digest,
    package: Sha256Digest,
) -> Result<ProductList, DataError> {
    let (list, _): (ProductList, _) = read_document(store, list_path(digest), digest).await?;
    list.canonical()?;
    if list.package != package {
        return Err(DataError::Invalid {
            field: "product list",
            detail: alloc::format!("for package {}, not {package}", list.package),
        });
    }
    Ok(list)
}

/// Reads the recipe of a chunk.
///
/// # Errors
///
/// Refuses a missing, changed, or non-canonical recipe and another version.
pub async fn read_recipe<S: ReadPort + ?Sized>(
    store: &S,
    digest: Sha256Digest,
) -> Result<Recipe, DataError> {
    let (recipe, _): (Recipe, _) = read_document(store, blob_path(digest), digest).await?;
    check_version("recipe", &recipe.format, RECIPE_FORMAT, recipe.version)?;
    Ok(recipe)
}
