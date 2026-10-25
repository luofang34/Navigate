//! Bound file reads and bind evidence to content digests.
use crate::Error;
mod heights;
pub use heights::heights_verified_blocking;
use image::{GrayImage, ImageReader};
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Cursor, Read, Write},
    path::Path,
};

pub fn read_blocking(path: &Path, limit: usize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|f| {
            f.take(limit.saturating_add(1) as u64)
                .read_to_end(&mut bytes)
        })
        .map_err(|source| Error::Io {
            operation: "read",
            path: path.into(),
            source,
        })?;
    if bytes.len() > limit {
        return Err(Error::Input(format!(
            "{} exceeds {limit} bytes",
            path.display()
        )));
    }
    Ok(bytes)
}
pub fn read_json_blocking<T: DeserializeOwned>(path: &Path) -> Result<T, Error> {
    serde_json::from_slice(&read_blocking(path, 4 * 1024 * 1024)?).map_err(|source| Error::Json {
        path: path.into(),
        source,
    })
}
pub fn image_blocking(path: &Path, max_pixels: usize) -> Result<GrayImage, Error> {
    image_verified_blocking(path, max_pixels, None)
}
pub fn image_verified_blocking(
    path: &Path,
    max_pixels: usize,
    expected: Option<&str>,
) -> Result<GrayImage, Error> {
    let bytes = read_blocking(path, 64 * 1024 * 1024)?;
    if let Some(expected) = expected {
        verify(&bytes, expected)?;
    }
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|source| Error::Io {
            operation: "detect image format",
            path: path.into(),
            source,
        })?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(max_pixels.saturating_mul(16) as u64);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|source| Error::Image {
            path: path.into(),
            source,
        })?
        .to_luma8();
    if image.as_raw().len() > max_pixels {
        return Err(Error::Input(format!(
            "{} exceeds {max_pixels} pixels",
            path.display()
        )));
    }
    Ok(image)
}
pub fn write_blocking(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| Error::Io {
            operation: "create",
            path: path.into(),
            source,
        })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| Error::Io {
            operation: "write",
            path: path.into(),
            source,
        })
}
pub fn write_json_blocking(path: &Path, value: &impl Serialize) -> Result<(), Error> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|source| Error::Json {
        path: path.into(),
        source,
    })?;
    write_blocking(path, &bytes)
}
pub fn verify(bytes: &[u8], expected: &str) -> Result<(), Error> {
    let actual = hex::encode(Sha256::digest(bytes));
    if expected != actual {
        return Err(Error::Input(format!(
            "content digest mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}
