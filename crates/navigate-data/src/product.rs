//! Derived products in two levels (ADR-0022 §12).
//!
//! - A **product chunk** (`products/chunks/<sha256>.json`) is identified by
//!   its kind, the digest of its complete [`Recipe`], its grid cell, its
//!   altitude band, the sorted digests of the input blobs it was computed
//!   from, and, for route products, the route digest and the validity
//!   envelope. It names no package, so any mission whose package holds the
//!   same input blobs can use it.
//! - A **product list** (`products/lists/<sha256>.json`) is per mission. It
//!   names the package identity and the chunks. It is outside the core
//!   identity.
//!
//! The documents follow the canonical rule of the manifest: integers only,
//! closed names or digests as strings, sorted collections, explicit nulls.
//! A product kind that this crate does not know is carried in a list and
//! enumerated as unknown, but never read.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::{DataError, GridCell, Sha256Digest};

mod document;
mod writer;
pub use document::{
    ListEntry, ProductChunk, ProductList, chunk_path, list_path, open_chunk, open_list, read_recipe,
};
pub use writer::{ProductWriter, write_list};

/// The `format` of a product chunk.
pub const CHUNK_FORMAT: &str = "navigate.data.product_chunk";
/// The `format` of a product list.
pub const LIST_FORMAT: &str = "navigate.data.product_list";
/// The `format` of a recipe.
pub const RECIPE_FORMAT: &str = "navigate.data.recipe";
/// The one version of the three product documents.
pub const PRODUCT_FORMAT_VERSION: u32 = 1;

/// The product kinds that this crate knows. Kinds after
/// [`ProductKind::CorridorLandmarks`] are reserved names with no builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductKind {
    /// Route keyframes: reference views at planned poses (representation K).
    Keyframes,
    /// Sparse 3D landmarks with descriptors, for example the `NVKTILE1`
    /// corridor bank (representation L).
    CorridorLandmarks,
    /// Reserved: terrain slope grids.
    TerrainSlopes,
    /// Reserved: horizon profiles.
    HorizonProfiles,
    /// Reserved: scalar fields, for example magnetic anomaly or gravity.
    ScalarField,
    /// Reserved: star pair tables.
    StarPairs,
    /// Reserved: station lists.
    StationList,
    /// Reserved: ephemeris segments.
    EphemerisSegments,
}

/// A product kind as a list carries it: a known kind, or the name of a kind
/// that this crate does not know.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KindName {
    /// A known kind.
    Known(ProductKind),
    /// An unknown kind, carried and enumerated, never read.
    Unknown(String),
}

/// Numeric precision of a recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// 16-bit floats.
    F16,
    /// 32-bit floats.
    F32,
    /// 64-bit floats.
    F64,
}

/// The complete recipe of a product: what made it, with each parameter as
/// an integer in the unit that its name states (for example `spacing_mm`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// [`RECIPE_FORMAT`].
    pub format: String,
    /// [`PRODUCT_FORMAT_VERSION`].
    pub version: u32,
    /// Name of the algorithm, a constant of the builder code.
    pub algorithm: String,
    /// Version of the builder code.
    pub code_version: String,
    /// Digest of the model weights, if the algorithm uses a model.
    pub weights: Option<Sha256Digest>,
    /// Numeric precision.
    pub precision: Precision,
    /// Each parameter, by name.
    pub parameters: BTreeMap<String, i64>,
}

impl Recipe {
    /// A recipe of the current format.
    #[must_use]
    pub fn new(
        algorithm: &str,
        code_version: &str,
        weights: Option<Sha256Digest>,
        precision: Precision,
    ) -> Self {
        Self {
            format: String::from(RECIPE_FORMAT),
            version: PRODUCT_FORMAT_VERSION,
            algorithm: String::from(algorithm),
            code_version: String::from(code_version),
            weights,
            precision,
            parameters: BTreeMap::new(),
        }
    }

    /// The same recipe with one more parameter.
    #[must_use]
    pub fn with(mut self, name: &str, value: i64) -> Self {
        self.parameters.insert(String::from(name), value);
        self
    }
}

/// Pinhole calibration in micro-pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    /// Image width, pixels.
    pub width: u32,
    /// Image height, pixels.
    pub height: u32,
    /// Focal length along `x`, micro-pixels.
    pub fx_upx: i64,
    /// Focal length along `y`, micro-pixels.
    pub fy_upx: i64,
    /// Principal point `x`, micro-pixels.
    pub cx_upx: i64,
    /// Principal point `y`, micro-pixels.
    pub cy_upx: i64,
}

/// The poses in which a route product entry is valid, about its planned
/// pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidityEnvelope {
    /// Largest horizontal distance, millimetres.
    pub lateral_mm: u32,
    /// Largest height difference, millimetres.
    pub vertical_mm: u32,
    /// Largest heading difference, microradians.
    pub heading_urad: u32,
}

/// A planned pose in integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedPose {
    /// Latitude, nanodegrees.
    pub lat_ndeg: i64,
    /// Longitude, nanodegrees.
    pub lon_ndeg: i64,
    /// Height above the WGS84 ellipsoid, millimetres.
    pub height_mm: i64,
    /// Heading clockwise from north, microradians.
    pub heading_urad: i64,
}

impl PlannedPose {
    /// The pose of `[lat, lon]` in degrees, a height in metres, and a heading
    /// in radians, rounded to the integer units.
    #[must_use]
    pub fn from_si([lat, lon]: [f64; 2], height_m: f64, heading_rad: f64) -> Self {
        let round = |v: f64| navigate_geodesy::float_math::round(v) as i64;
        Self {
            lat_ndeg: round(lat * 1e9),
            lon_ndeg: round(lon * 1e9),
            height_mm: round(height_m * 1e3),
            heading_urad: round(heading_rad * 1e6),
        }
    }

    /// `([lat, lon]` degrees, height metres, heading radians`)`.
    #[must_use]
    pub fn to_si(&self) -> ([f64; 2], f64, f64) {
        (
            [self.lat_ndeg as f64 * 1e-9, self.lon_ndeg as f64 * 1e-9],
            self.height_mm as f64 * 1e-3,
            self.heading_urad as f64 * 1e-6,
        )
    }
}

/// One entry of a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkEntry {
    /// Key of the entry, unique and increasing in the chunk.
    pub key: u32,
    /// The planned pose, where the entry has one.
    pub pose: Option<PlannedPose>,
    /// Digest of the blob.
    pub blob: Sha256Digest,
    /// Size of the blob in bytes.
    pub bytes: u64,
}

/// What a chunk is for: the binding that decides whether a mission can use
/// it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkBinding {
    /// The grid cell, for cell products.
    pub cell: Option<GridCell>,
    /// The altitude band.
    pub altitude_band: Option<u32>,
    /// The sorted digests of the input blobs.
    pub inputs: Vec<Sha256Digest>,
    /// The digest of the route, for route products.
    pub route: Option<Sha256Digest>,
    /// The validity envelope of each entry, for route products.
    pub envelope: Option<ValidityEnvelope>,
    /// The camera calibration, where it applies.
    pub calibration: Option<Calibration>,
}

/// What a navigator does with one product at load.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductUse {
    /// The navigator uses the product.
    Use,
    /// The navigator does not use the product and falls back to another
    /// representation, for the stated reason.
    Fallback(String),
    /// The navigator refuses the product, for the stated reason.
    Refusal(String),
}

/// The digest of a route: the canonical JSON of its points as
/// `[lat_ndeg, lon_ndeg, height_mm]`, in fly order.
#[must_use]
pub fn route_digest(points: &[[f64; 3]]) -> Sha256Digest {
    let round = |v: f64| navigate_geodesy::float_math::round(v) as i64;
    let integers: Vec<[i64; 3]> = points
        .iter()
        .map(|[lat, lon, h]| [round(lat * 1e9), round(lon * 1e9), round(h * 1e3)])
        .collect();
    Sha256Digest::of(&serde_json::to_vec(&integers).unwrap_or_default())
}

/// The digest of a set of input blobs: the SHA-256 of their sorted digests
/// in binary. A builder that writes one source identity into its own
/// format (for example the `NVKTILE1` bank) writes this one.
#[must_use]
pub fn inputs_digest(inputs: &[Sha256Digest]) -> Sha256Digest {
    let mut sorted = inputs.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let bytes: Vec<u8> = sorted.iter().flat_map(|d| d.0).collect();
    Sha256Digest::of(&bytes)
}

/// Canonical bytes of a product document.
pub(crate) fn canonical<T: Serialize>(document: &T, path: &str) -> Result<Vec<u8>, DataError> {
    serde_json::to_vec(document).map_err(|error| DataError::Document {
        path: String::from(path),
        detail: alloc::format!("{error}"),
    })
}

#[cfg(test)]
mod tests;
