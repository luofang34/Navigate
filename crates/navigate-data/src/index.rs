//! Tile index blobs.
//!
//! Each layer has one tile index: a canonical binary blob in the
//! `chunks/` store. It is the magic `NVDIDX01`, then one record of
//! [`RECORD_BYTES`] for each address of the declared coverage, sorted by
//! zoom, row, and column, with no repeats:
//!
//! | Bytes | Field |
//! |---|---|
//! | 1 | zoom |
//! | 4 | column, little endian |
//! | 4 | row, little endian |
//! | 8 | blob size, little endian; zero marks a gap |
//! | 32 | blob SHA-256; zero for a gap |
//!
//! The manifest holds the digest, size, and counts of the index. A reader
//! checks the index once at open and then reads records by range with a
//! binary search.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{DataError, MAX_ZOOM, RandomAccess, Sha256Digest, TileAddress};

/// Magic of a tile index.
pub const INDEX_MAGIC: &[u8; 8] = b"NVDIDX01";
/// Bytes of one index record.
pub const RECORD_BYTES: usize = 49;
/// Records that one verification read covers.
const RECORDS_PER_READ: usize = 1_024;

/// What an index declares at one address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexEntry {
    /// The package holds this tile.
    Tile {
        /// Digest of the blob.
        blob: Sha256Digest,
        /// Size of the blob in bytes, not zero.
        bytes: u64,
    },
    /// The address is in the coverage, but the source did not have it.
    Gap,
}

/// The sort key of a record: zoom, row, column.
pub(crate) type IndexKey = (u32, u32, u32);

/// The sort key of an address.
pub(crate) const fn key(a: TileAddress) -> IndexKey {
    (a.zoom, a.y, a.x)
}

/// Encodes an index. The map order is the canonical order.
#[must_use]
pub fn encode_index(entries: &BTreeMap<IndexKey, IndexEntry>) -> Vec<u8> {
    let mut out = Vec::with_capacity(INDEX_MAGIC.len() + entries.len() * RECORD_BYTES);
    out.extend_from_slice(INDEX_MAGIC);
    for (&(zoom, y, x), entry) in entries {
        out.push(zoom as u8);
        out.extend_from_slice(&x.to_le_bytes());
        out.extend_from_slice(&y.to_le_bytes());
        let (bytes, blob) = match entry {
            IndexEntry::Tile { blob, bytes } => (*bytes, blob.0),
            IndexEntry::Gap => (0, [0; 32]),
        };
        out.extend_from_slice(&bytes.to_le_bytes());
        out.extend_from_slice(&blob);
    }
    out
}

/// Decodes one record.
fn decode_record(record: &[u8], path: &str) -> Result<(TileAddress, IndexEntry), DataError> {
    let bad = |detail: &str| DataError::Document {
        path: String::from(path),
        detail: String::from(detail),
    };
    let (zoom, rest) = record.split_first().ok_or_else(|| bad("short record"))?;
    let (x, rest) = rest
        .split_first_chunk::<4>()
        .ok_or_else(|| bad("short record"))?;
    let (y, rest) = rest
        .split_first_chunk::<4>()
        .ok_or_else(|| bad("short record"))?;
    let (size, rest) = rest
        .split_first_chunk::<8>()
        .ok_or_else(|| bad("short record"))?;
    let digest = rest
        .first_chunk::<32>()
        .ok_or_else(|| bad("short record"))?;
    let (zoom, x, y) = (
        u32::from(*zoom),
        u32::from_le_bytes(*x),
        u32::from_le_bytes(*y),
    );
    if zoom > MAX_ZOOM {
        return Err(bad("zoom past the limit"));
    }
    let address = TileAddress::new(zoom, x, y).map_err(|_| bad("address outside its zoom"))?;
    let bytes = u64::from_le_bytes(*size);
    let entry = match (bytes, *digest == [0; 32]) {
        (0, true) => IndexEntry::Gap,
        (0, false) | (_, true) => return Err(bad("gap with a digest or tile without one")),
        (bytes, false) => IndexEntry::Tile {
            blob: Sha256Digest(*digest),
            bytes,
        },
    };
    Ok((address, entry))
}

/// What the check of an index found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexCounts {
    /// Records that are tiles.
    pub tiles: u64,
    /// Records that are gaps.
    pub gaps: u64,
}

/// Reads a whole index by ranges and checks its digest, its size, its magic,
/// the order of its records, and each record.
///
/// # Errors
///
/// Refuses an index that does not agree with `digest` or that breaks a
/// rule.
pub async fn verify_index<F: RandomAccess>(
    file: &F,
    digest: Sha256Digest,
    path: &str,
) -> Result<IndexCounts, DataError> {
    use sha2::{Digest as _, Sha256};
    let len = file.len();
    let bad = |detail: String| DataError::Integrity {
        path: String::from(path),
        detail,
    };
    let records = len
        .checked_sub(INDEX_MAGIC.len() as u64)
        .filter(|body| body % RECORD_BYTES as u64 == 0)
        .ok_or_else(|| bad(alloc::format!("{len} bytes is not an index")))?
        / RECORD_BYTES as u64;
    let magic = file.read_at(0, INDEX_MAGIC.len()).await?;
    if magic.as_slice() != INDEX_MAGIC {
        return Err(bad(String::from("magic")));
    }
    let mut hasher = Sha256::new();
    hasher.update(&magic);
    let (mut counts, mut previous, mut done) = (IndexCounts { tiles: 0, gaps: 0 }, None, 0_u64);
    while done < records {
        let take = (records - done).min(RECORDS_PER_READ as u64);
        let offset = INDEX_MAGIC.len() as u64 + done * RECORD_BYTES as u64;
        let chunk = file.read_at(offset, take as usize * RECORD_BYTES).await?;
        hasher.update(&chunk);
        for record in chunk.chunks(RECORD_BYTES) {
            let (address, entry) = decode_record(record, path)?;
            if previous.is_some_and(|p| p >= key(address)) {
                return Err(bad(String::from("records out of order")));
            }
            previous = Some(key(address));
            match entry {
                IndexEntry::Tile { .. } => counts.tiles = counts.tiles.wrapping_add(1),
                IndexEntry::Gap => counts.gaps = counts.gaps.wrapping_add(1),
            }
        }
        done = done.wrapping_add(take);
    }
    if Sha256Digest(hasher.finalize().into()) != digest {
        return Err(bad(String::from("digest")));
    }
    Ok(counts)
}

/// Finds the record of `address` by binary search over ranged reads.
/// `None` means the address is outside the declared coverage.
///
/// # Errors
///
/// Returns read failures and records that do not decode.
pub async fn find<F: RandomAccess>(
    file: &F,
    records: u64,
    address: TileAddress,
    path: &str,
) -> Result<Option<IndexEntry>, DataError> {
    let target = key(address);
    let (mut low, mut high) = (0_u64, records);
    while low < high {
        let middle = low + (high - low) / 2;
        let offset = INDEX_MAGIC.len() as u64 + middle * RECORD_BYTES as u64;
        let record = file.read_at(offset, RECORD_BYTES).await?;
        let (found, entry) = decode_record(&record, path)?;
        match key(found).cmp(&target) {
            core::cmp::Ordering::Equal => return Ok(Some(entry)),
            core::cmp::Ordering::Less => low = middle + 1,
            core::cmp::Ordering::Greater => high = middle,
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests;
