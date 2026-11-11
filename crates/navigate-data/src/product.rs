//! Derived products: files that a builder makes from the layers of a core
//! package.
//!
//! A product is outside the core identity. Its manifest binds the core
//! package identity, the complete recipe, the camera calibration where it
//! applies, the altitude band, and the validity envelope of each entry.
//! The SHA-256 of the canonical product manifest is the product identity.
//! Its blobs are in the same `chunks/` store as the core blobs, and its
//! manifest is at `products/<sha256>.json`.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::{DataError, Package, ReadPort, Sha256Digest, blob_path};

mod writer;
pub use writer::ProductWriter;

/// The `format` of a product manifest.
pub const PRODUCT_FORMAT: &str = "navigate.data.product";
/// The `version` that this crate writes and reads.
pub const PRODUCT_FORMAT_VERSION: u32 = 1;

/// Logical path of a product manifest.
#[must_use]
pub fn product_path(identity: Sha256Digest) -> String {
    alloc::format!("products/{identity}.json")
}

/// What a product holds. The kinds after `CorridorLandmarks` are reserved
/// names: no builder makes them yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProductKind {
    /// Route keyframes: reference views or features made at planned poses,
    /// each with a validity envelope (representation K).
    Keyframes,
    /// Sparse 3D landmarks with descriptors (representation L), for example
    /// the `NVKTILE1` corridor bank.
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

/// The complete recipe of a product: what made it, with each parameter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// Name of the algorithm, for example `sparse-ray-march`.
    pub algorithm: String,
    /// Version of the algorithm code, for example the crate version.
    pub version: String,
    /// Digest of the model weights, if the algorithm uses a model.
    pub weights: Option<Sha256Digest>,
    /// Numeric precision, for example `f32`.
    pub precision: String,
    /// Each parameter, by name, as text.
    pub parameters: BTreeMap<String, String>,
}

/// Pinhole calibration of the camera that a product is made for.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Focal length along `x` in pixels.
    pub fx: f64,
    /// Focal length along `y` in pixels.
    pub fy: f64,
    /// Principal point `x` in pixels.
    pub cx: f64,
    /// Principal point `y` in pixels.
    pub cy: f64,
}

/// The poses in which an entry is valid, about its planned pose.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidityEnvelope {
    /// Largest horizontal distance from the planned position, metres.
    pub lateral_m: f64,
    /// Largest height difference from the planned position, metres.
    pub vertical_m: f64,
    /// Largest heading difference from the planned heading, radians.
    pub heading_rad: f64,
}

/// The planned pose of an entry.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedPose {
    /// `[lat, lon]` in degrees and the height above the WGS84 ellipsoid in
    /// metres.
    pub position: [f64; 3],
    /// Heading in radians, clockwise from north.
    pub heading_rad: f64,
}

/// One entry of a product.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductEntry {
    /// Key of the entry, unique in the product, for example a grid cell
    /// identifier or a keyframe index.
    pub key: String,
    /// The planned pose, where the entry has one.
    pub pose: Option<PlannedPose>,
    /// Digest of the blob.
    pub blob: Sha256Digest,
    /// Size of the blob in bytes.
    pub bytes: u64,
}

/// The manifest of one product.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductManifest {
    /// [`PRODUCT_FORMAT`].
    pub format: String,
    /// [`PRODUCT_FORMAT_VERSION`].
    pub version: u32,
    /// What the product holds.
    pub kind: ProductKind,
    /// The identity of the core package that the product comes from.
    pub core: Sha256Digest,
    /// The complete recipe.
    pub recipe: Recipe,
    /// The camera calibration, where it applies.
    pub calibration: Option<Calibration>,
    /// The altitude band, where it applies.
    pub altitude_band: Option<u32>,
    /// The validity envelope of each entry about its planned pose.
    pub envelope: Option<ValidityEnvelope>,
    /// The entries, in key order.
    pub entries: Vec<ProductEntry>,
}

impl ProductManifest {
    /// Checks the format, the order, and the values.
    ///
    /// # Errors
    ///
    /// Refuses another format or version, an empty algorithm, entries out
    /// of key order or repeated, and values that are not finite.
    pub fn validate(&self) -> Result<(), DataError> {
        let invalid = |field: &'static str, detail: String| DataError::Invalid { field, detail };
        if self.format != PRODUCT_FORMAT || self.version != PRODUCT_FORMAT_VERSION {
            return Err(invalid(
                "format",
                alloc::format!("{} {}", self.format, self.version),
            ));
        }
        if self.recipe.algorithm.is_empty() || self.recipe.version.is_empty() {
            return Err(invalid(
                "recipe",
                String::from("empty algorithm or version"),
            ));
        }
        if self
            .entries
            .windows(2)
            .any(|w| matches!(w, [a, b] if a.key >= b.key))
        {
            return Err(invalid("entries", String::from("not in strict key order")));
        }
        let envelope_ok = self.envelope.is_none_or(|e| {
            [e.lateral_m, e.vertical_m, e.heading_rad]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        });
        let poses_ok = self
            .entries
            .iter()
            .filter_map(|e| e.pose)
            .all(|p| p.position.iter().all(|v| v.is_finite()) && p.heading_rad.is_finite());
        if !envelope_ok || !poses_ok {
            return Err(invalid("entries", String::from("a value is not finite")));
        }
        Ok(())
    }

    /// The canonical bytes: compact JSON in declaration order.
    ///
    /// # Errors
    ///
    /// Refuses a manifest that [`Self::validate`] refuses.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, DataError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| DataError::Manifest {
            path: String::from("<memory>"),
            detail: alloc::format!("{error}"),
        })
    }
}

/// A product whose manifest has been checked against its core package.
#[derive(Clone, Debug, PartialEq)]
pub struct Product {
    manifest: ProductManifest,
    identity: Sha256Digest,
}

impl Product {
    /// Opens the product `identity` for the core package `core`.
    ///
    /// # Errors
    ///
    /// Refuses a missing manifest, a digest that differs, a manifest that
    /// is not canonical, and a product of another core package.
    pub async fn open<S: ReadPort + ?Sized>(
        store: &S,
        identity: Sha256Digest,
        core: &Package,
    ) -> Result<Self, DataError> {
        let path = product_path(identity);
        let bytes = store
            .read(&path)
            .await?
            .ok_or_else(|| DataError::Missing { path: path.clone() })?;
        if Sha256Digest::of(&bytes) != identity {
            return Err(DataError::Integrity {
                path,
                detail: String::from("product manifest digest differs"),
            });
        }
        let refuse = |detail: String| DataError::Manifest {
            path: path.clone(),
            detail,
        };
        let manifest: ProductManifest =
            serde_json::from_slice(&bytes).map_err(|e| refuse(alloc::format!("{e}")))?;
        if manifest.canonical_bytes()? != bytes {
            return Err(refuse(String::from("the bytes are not in canonical form")));
        }
        if manifest.core != core.identity() {
            return Err(refuse(alloc::format!(
                "made from core {}, not {}",
                manifest.core,
                core.identity()
            )));
        }
        Ok(Self { manifest, identity })
    }

    /// The manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ProductManifest {
        &self.manifest
    }

    /// The product identity.
    #[must_use]
    pub const fn identity(&self) -> Sha256Digest {
        self.identity
    }

    /// Reads the blob of one entry and checks it.
    ///
    /// # Errors
    ///
    /// Refuses a missing or changed blob.
    pub async fn entry_bytes<S: ReadPort + ?Sized>(
        &self,
        store: &S,
        entry: &ProductEntry,
    ) -> Result<Vec<u8>, DataError> {
        let path = blob_path(entry.blob);
        let bytes = store
            .read(&path)
            .await?
            .ok_or_else(|| DataError::Missing { path: path.clone() })?;
        if bytes.len() as u64 != entry.bytes || Sha256Digest::of(&bytes) != entry.blob {
            return Err(DataError::Integrity {
                path,
                detail: alloc::format!("entry {} does not agree with the manifest", entry.key),
            });
        }
        Ok(bytes)
    }
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

#[cfg(test)]
mod tests;
