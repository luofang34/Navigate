//! Writing product chunks and product lists: blobs first, documents last.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::product::document::{ListEntry, chunk_path, list_path};
use crate::product::{
    CHUNK_FORMAT, ChunkBinding, ChunkEntry, KindName, LIST_FORMAT, PRODUCT_FORMAT_VERSION,
    PlannedPose, ProductChunk, ProductKind, ProductList, Recipe, canonical,
};
use crate::{DataError, Sha256Digest, WritePort, blob_path};

/// Builds one product chunk.
#[derive(Clone, Debug)]
pub struct ProductWriter {
    kind: ProductKind,
    recipe: Recipe,
    binding: ChunkBinding,
    entries: BTreeMap<u32, ChunkEntry>,
}

impl ProductWriter {
    /// A writer for a chunk of `kind` made with `recipe` for `binding`. The
    /// binding inputs are sorted and repeats removed.
    #[must_use]
    pub fn new(kind: ProductKind, recipe: Recipe, mut binding: ChunkBinding) -> Self {
        binding.inputs.sort_unstable();
        binding.inputs.dedup();
        Self {
            kind,
            recipe,
            binding,
            entries: BTreeMap::new(),
        }
    }

    /// Stores the blob of one entry and declares it.
    ///
    /// # Errors
    ///
    /// Refuses a key used before and a failed write.
    pub async fn add_entry<W: WritePort + ?Sized>(
        &mut self,
        store: &mut W,
        key: u32,
        pose: Option<PlannedPose>,
        bytes: &[u8],
    ) -> Result<Sha256Digest, DataError> {
        if self.entries.contains_key(&key) {
            return Err(DataError::Invalid {
                field: "key",
                detail: alloc::format!("{key} twice"),
            });
        }
        let blob = Sha256Digest::of(bytes);
        store.write(&blob_path(blob), bytes).await?;
        self.entries.insert(
            key,
            ChunkEntry {
                key,
                pose,
                blob,
                bytes: bytes.len() as u64,
            },
        );
        Ok(blob)
    }

    /// The chunk that [`Self::seal`] would write, with its recipe bytes.
    ///
    /// # Errors
    ///
    /// Refuses a chunk that breaks a rule.
    pub fn chunk(&self) -> Result<(ProductChunk, Vec<u8>), DataError> {
        let recipe = canonical(&self.recipe, "<recipe>")?;
        let chunk = ProductChunk {
            format: String::from(CHUNK_FORMAT),
            version: PRODUCT_FORMAT_VERSION,
            kind: KindName::Known(self.kind),
            recipe: Sha256Digest::of(&recipe),
            binding: self.binding.clone(),
            entries: self.entries.values().copied().collect(),
        };
        chunk.validate()?;
        Ok((chunk, recipe))
    }

    /// Checks that the store holds each entry blob, writes the recipe, then
    /// the chunk. Returns the chunk digest.
    ///
    /// # Errors
    ///
    /// Refuses a chunk that breaks a rule, a missing blob, and a failed
    /// write.
    pub async fn seal<W: WritePort + ?Sized>(
        &self,
        store: &mut W,
    ) -> Result<Sha256Digest, DataError> {
        let (chunk, recipe) = self.chunk()?;
        for entry in &chunk.entries {
            let path = blob_path(entry.blob);
            if !store.contains(&path).await? {
                return Err(DataError::Missing { path });
            }
        }
        store.write(&blob_path(chunk.recipe), &recipe).await?;
        let (bytes, digest) = chunk.canonical()?;
        store.write(&chunk_path(digest), &bytes).await?;
        Ok(digest)
    }
}

/// Writes the product list of `package` that names `chunks`. Returns the
/// list digest.
///
/// # Errors
///
/// Refuses a chunk that the store does not hold and a failed write.
pub async fn write_list<W: WritePort + ?Sized>(
    store: &mut W,
    package: Sha256Digest,
    chunks: &[(KindName, Sha256Digest)],
) -> Result<Sha256Digest, DataError> {
    let mut entries: Vec<ListEntry> = chunks
        .iter()
        .map(|(kind, chunk)| ListEntry {
            kind: kind.clone(),
            chunk: *chunk,
        })
        .collect();
    entries.sort();
    entries.dedup();
    for entry in &entries {
        let path = chunk_path(entry.chunk);
        if matches!(entry.kind, KindName::Known(_)) && !store.contains(&path).await? {
            return Err(DataError::Missing { path });
        }
    }
    let list = ProductList {
        format: String::from(LIST_FORMAT),
        version: PRODUCT_FORMAT_VERSION,
        package,
        chunks: entries,
    };
    let (bytes, digest) = list.canonical()?;
    store.write(&list_path(digest), &bytes).await?;
    Ok(digest)
}
