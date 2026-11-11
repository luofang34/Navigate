//! Writing a product: blobs first, the manifest last.

use alloc::collections::BTreeMap;
use alloc::string::String;

use crate::product::{
    Calibration, PRODUCT_FORMAT, PRODUCT_FORMAT_VERSION, PlannedPose, ProductEntry, ProductKind,
    ProductManifest, Recipe, ValidityEnvelope, product_path,
};
use crate::{DataError, Sha256Digest, WritePort, blob_path};

/// Builds one product for one core package.
#[derive(Clone, Debug)]
pub struct ProductWriter {
    kind: ProductKind,
    core: Sha256Digest,
    recipe: Recipe,
    calibration: Option<Calibration>,
    altitude_band: Option<u32>,
    envelope: Option<ValidityEnvelope>,
    entries: BTreeMap<String, ProductEntry>,
}

impl ProductWriter {
    /// A writer for a product of `kind` from the core package `core`.
    #[must_use]
    pub const fn new(kind: ProductKind, core: Sha256Digest, recipe: Recipe) -> Self {
        Self {
            kind,
            core,
            recipe,
            calibration: None,
            altitude_band: None,
            envelope: None,
            entries: BTreeMap::new(),
        }
    }

    /// Records the camera calibration that the product is made for.
    #[must_use]
    pub const fn with_calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = Some(calibration);
        self
    }

    /// Records the altitude band.
    #[must_use]
    pub const fn with_altitude_band(mut self, band: u32) -> Self {
        self.altitude_band = Some(band);
        self
    }

    /// Records the validity envelope of each entry.
    #[must_use]
    pub const fn with_envelope(mut self, envelope: ValidityEnvelope) -> Self {
        self.envelope = Some(envelope);
        self
    }

    /// Stores the blob of one entry and declares it.
    ///
    /// # Errors
    ///
    /// Refuses a key used before and a failed write.
    pub async fn add_entry<W: WritePort + ?Sized>(
        &mut self,
        store: &mut W,
        key: &str,
        pose: Option<PlannedPose>,
        bytes: &[u8],
    ) -> Result<Sha256Digest, DataError> {
        if self.entries.contains_key(key) {
            return Err(DataError::Invalid {
                field: "key",
                detail: String::from(key),
            });
        }
        let blob = Sha256Digest::of(bytes);
        store.write(&blob_path(blob), bytes).await?;
        self.entries.insert(
            String::from(key),
            ProductEntry {
                key: String::from(key),
                pose,
                blob,
                bytes: bytes.len() as u64,
            },
        );
        Ok(blob)
    }

    /// The manifest that [`Self::seal`] would write.
    ///
    /// # Errors
    ///
    /// Refuses a manifest that breaks the rules.
    pub fn manifest(&self) -> Result<ProductManifest, DataError> {
        let manifest = ProductManifest {
            format: String::from(PRODUCT_FORMAT),
            version: PRODUCT_FORMAT_VERSION,
            kind: self.kind,
            core: self.core,
            recipe: self.recipe.clone(),
            calibration: self.calibration,
            altitude_band: self.altitude_band,
            envelope: self.envelope,
            entries: self.entries.values().cloned().collect(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Checks that the store holds each blob, then writes the manifest.
    /// Returns the product identity.
    ///
    /// # Errors
    ///
    /// Refuses a manifest that breaks the rules, a missing blob, and a
    /// failed write.
    pub async fn seal<W: WritePort + ?Sized>(
        &self,
        store: &mut W,
    ) -> Result<Sha256Digest, DataError> {
        let manifest = self.manifest()?;
        for entry in &manifest.entries {
            let path = blob_path(entry.blob);
            if !store.contains(&path).await? {
                return Err(DataError::Missing { path });
            }
        }
        let bytes = manifest.canonical_bytes()?;
        let identity = Sha256Digest::of(&bytes);
        store.write(&product_path(identity), &bytes).await?;
        Ok(identity)
    }
}
