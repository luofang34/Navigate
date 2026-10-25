//! Decode verified heights without a second full raster allocation.
use crate::Error;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

const CHUNK_BYTES: usize = 64 * 1024;

pub fn heights_verified_blocking(
    path: &Path,
    count: usize,
    expected: &str,
) -> Result<Vec<f32>, Error> {
    let file = File::open(path).map_err(|source| Error::Io {
        operation: "open heights",
        path: path.into(),
        source,
    })?;
    decode_blocking(file, path, count, expected)
}

fn decode_blocking(
    mut reader: impl Read,
    path: &Path,
    count: usize,
    expected: &str,
) -> Result<Vec<f32>, Error> {
    let length = count.checked_mul(4).ok_or_else(|| {
        Error::Input(format!(
            "height byte count overflows for {}",
            path.display()
        ))
    })?;
    let mut values = Vec::with_capacity(count);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; CHUNK_BYTES];
    let mut remaining = length;
    while remaining > 0 {
        let size = remaining.min(CHUNK_BYTES);
        reader
            .read_exact(&mut buffer[..size])
            .map_err(|source| Error::Io {
                operation: "read heights",
                path: path.into(),
                source,
            })?;
        digest.update(&buffer[..size]);
        values.extend(
            buffer[..size]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v)),
        );
        remaining -= size;
    }
    let mut trailing = Vec::new();
    reader
        .take(1)
        .read_to_end(&mut trailing)
        .map_err(|source| Error::Io {
            operation: "check height length",
            path: path.into(),
            source,
        })?;
    if !trailing.is_empty() {
        return Err(Error::Input(format!(
            "{} exceeds {length} height bytes",
            path.display()
        )));
    }
    // No decoded values leave this function until the full source is verified.
    let actual = hex::encode(digest.finalize());
    if actual != expected {
        return Err(Error::Input(format!(
            "height digest mismatch for {}: expected {expected}, got {actual}",
            path.display()
        )));
    }
    Ok(values)
}

#[cfg(test)]
mod tests;
