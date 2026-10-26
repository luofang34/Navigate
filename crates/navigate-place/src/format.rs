//! The index files: a JSON manifest and binary shards of fixed-size records.
//!
//! A shard holds the records of one crop size in one [`Cell`]. It starts with a
//! 32-byte header. Each record is the crop centre as two `i32` values in 1e-7
//! degrees, then the descriptor as half-precision values. Records follow a Hilbert
//! curve inside the cell, so that a small area is a few contiguous byte ranges.
use crate::{CELL_ZOOM, Cell, PlaceError, region::tile_xy};
use half::f16;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"NPLS";
const VERSION: u16 = 1;
const HEADER: usize = 32;
/// Hilbert grid side inside one cell.
const CURVE: u32 = 1 << 16;

/// The manifest of one place index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexManifest {
    /// Always `navigate-place-index`.
    pub format: String,
    /// Manifest version.
    pub version: u32,
    /// Short name of the indexed region.
    pub region: String,
    /// Name for a person.
    pub label: String,
    /// South, west, north and east edges of the indexed area, degrees.
    pub bounds: [f64; 4],
    /// Imagery source identity, for example `usgs-imagery-only`.
    pub imagery: String,
    /// Licence of the imagery that the index derives from.
    pub imagery_licence: String,
    /// The descriptor model and its input.
    pub model: ModelEntry,
    /// The crop sizes in the index.
    pub scales: Vec<ScaleEntry>,
}

/// The descriptor model of an index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Model name, for example `dinov2-small`.
    pub id: String,
    /// SHA-256 of the model file.
    pub sha256: String,
    /// Side of a rendered canvas, pixels.
    pub render_px: u32,
    /// Side of the model input, pixels.
    pub encode_px: u32,
}

/// One crop size of an index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleEntry {
    /// Side of a crop on the ground, metres. Crop centres are half a side apart.
    pub side_m: f64,
    /// Descriptor width of the records.
    pub width: usize,
    /// The query head that brings a query descriptor into the record space. Without
    /// a head, query and records use the plain pooled descriptor.
    pub query_head: Option<QueryHead>,
    /// One shard for each cell that holds records.
    pub shards: Vec<ShardEntry>,
}

/// A linear map from a pooled query descriptor to the record space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueryHead {
    /// Path of the head file next to the manifest: little-endian `f16`, row by row.
    pub path: String,
    /// SHA-256 of the head file.
    pub sha256: String,
    /// Output width.
    pub rows: usize,
    /// Input width.
    pub columns: usize,
}

/// One shard file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardEntry {
    /// The cell that the shard covers.
    pub cell: Cell,
    /// Path next to the manifest.
    pub path: String,
    /// Records in the shard.
    pub records: usize,
    /// SHA-256 of the shard file.
    pub sha256: String,
}

/// One map crop of the index.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    /// Latitude and longitude of the crop centre, degrees.
    pub centre: [f64; 2],
    /// Unit-length descriptor.
    pub descriptor: Vec<f32>,
}

impl IndexManifest {
    /// Read a manifest from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid JSON or a different format name.
    pub fn from_json(text: &str) -> Result<Self, PlaceError> {
        let manifest: Self = serde_json::from_str(text)?;
        if manifest.format != "navigate-place-index" || manifest.version != 1 {
            return Err(PlaceError::Request(format!(
                "unknown index format {} {}",
                manifest.format, manifest.version
            )));
        }
        Ok(manifest)
    }
}

fn record_bytes(width: usize) -> usize {
    8 + 2 * width
}

/// The shard file of `records` in `cell`, records in Hilbert order.
pub fn encode_shard(
    cell: Cell,
    side_m: f64,
    width: usize,
    records: &[Record],
) -> Result<Vec<u8>, PlaceError> {
    let mut ordered: Vec<(u64, &Record)> = Vec::with_capacity(records.len());
    for record in records {
        if record.descriptor.len() != width {
            return Err(PlaceError::Width {
                expected: width,
                found: record.descriptor.len(),
            });
        }
        let [x, y] = tile_xy(record.centre, CELL_ZOOM);
        let along = |v: f64, origin: u32| {
            ((v - f64::from(origin)) * f64::from(CURVE)).clamp(0.0, f64::from(CURVE - 1)) as u32
        };
        ordered.push((
            crate::hilbert::index(CURVE, along(x, cell.x), along(y, cell.y)),
            record,
        ));
    }
    ordered.sort_by_key(|(d, _)| *d);
    let mut out = Vec::with_capacity(HEADER + records.len() * record_bytes(width));
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(
        &u16::try_from(width)
            .map_err(|_| PlaceError::Width {
                expected: 65_535,
                found: width,
            })?
            .to_le_bytes(),
    );
    out.extend_from_slice(
        &u32::try_from(records.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    out.extend_from_slice(&(side_m as f32).to_le_bytes());
    out.extend_from_slice(&cell.x.to_le_bytes());
    out.extend_from_slice(&cell.y.to_le_bytes());
    out.resize(HEADER, 0);
    for (_, record) in ordered {
        for v in record.centre {
            out.extend_from_slice(&((v * 1e7).round() as i32).to_le_bytes());
        }
        for v in &record.descriptor {
            out.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
        }
    }
    Ok(out)
}

/// The records of a shard file; `path` names it in errors.
pub fn decode_shard(path: &str, bytes: &[u8]) -> Result<(Cell, f64, Vec<Record>), PlaceError> {
    let bad = |reason: &str| PlaceError::Shard {
        path: path.to_owned(),
        reason: reason.to_owned(),
    };
    let header = bytes
        .get(..HEADER)
        .ok_or_else(|| bad("shorter than its header"))?;
    let u16_at = |i: usize| u16::from_le_bytes([header[i], header[i + 1]]);
    let u32_at =
        |i: usize| u32::from_le_bytes([header[i], header[i + 1], header[i + 2], header[i + 3]]);
    if &header[..4] != MAGIC || u16_at(4) != VERSION {
        return Err(bad("unknown magic or version"));
    }
    let width = usize::from(u16_at(6));
    let count = u32_at(8) as usize;
    let side_m = f64::from(f32::from_bits(u32_at(12)));
    let cell = Cell {
        x: u32_at(16),
        y: u32_at(20),
    };
    let size = record_bytes(width);
    let body = &bytes[HEADER..];
    if body.len() != count * size {
        return Err(bad(&format!(
            "{} body bytes for {count} records of {size}",
            body.len()
        )));
    }
    let records = body
        .chunks_exact(size)
        .map(|r| {
            let coord = |i: usize| {
                f64::from(i32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]])) / 1e7
            };
            Record {
                centre: [coord(0), coord(4)],
                descriptor: r[8..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16::from_le_bytes(*b).to_f32())
                    .collect(),
            }
        })
        .collect();
    Ok((cell, side_m, records))
}

#[cfg(test)]
mod tests;
