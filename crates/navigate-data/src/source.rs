//! Layer kinds, encodings, and the registry of sources.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{DataError, DayNumber, Sha256Digest};

/// What a layer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    /// Orthoimagery.
    Imagery,
    /// Elevation of the surface.
    Elevation,
    /// A kind that this crate does not know. A core package that holds one
    /// is refused.
    #[serde(other)]
    Unsupported,
}

/// How the blobs of a layer are encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    /// PNG image.
    Png,
    /// JPEG image.
    Jpeg,
    /// WebP image.
    Webp,
    /// PNG with Terrarium heights: `R·256 + G + B/256 − 32768` metres.
    TerrariumPng,
    /// An encoding that this crate does not know.
    #[serde(other)]
    Unsupported,
}

/// A source: a registered name, or a declared source. The key, not a URL
/// or a host label, names the source in the package identity. Its text
/// form is the snake-case registry name (`esri_world_imagery`) or
/// `declared:<sha256>`, the digest of a canonical
/// [`SourceDeclaration`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceKey {
    /// Esri World Imagery.
    EsriWorldImagery,
    /// Mapzen Terrarium elevation tiles (the AWS terrain tiles).
    MapzenTerrarium,
    /// Data that a test or a simulator makes. A navigator never takes it
    /// as evidence of the real world.
    Synthetic,
    /// A source outside the registry, by the digest of its declaration:
    /// a private survey, a national orthophoto, a simulated world.
    Declared(Sha256Digest),
    /// A name that this crate does not know.
    Unsupported,
}

impl SourceKey {
    const NAMES: [(&'static str, Self); 3] = [
        ("esri_world_imagery", Self::EsriWorldImagery),
        ("mapzen_terrarium", Self::MapzenTerrarium),
        ("synthetic", Self::Synthetic),
    ];

    /// The key of a text form; an unknown text gives
    /// [`SourceKey::Unsupported`].
    #[must_use]
    pub fn parse(text: &str) -> Self {
        if let Some(hex) = text.strip_prefix("declared:") {
            return hex.parse().map_or(Self::Unsupported, Self::Declared);
        }
        Self::NAMES
            .iter()
            .find(|(name, _)| *name == text)
            .map_or(Self::Unsupported, |(_, key)| *key)
    }
}

impl fmt::Display for SourceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Declared(digest) => write!(f, "declared:{digest}"),
            Self::Unsupported => f.write_str("unsupported"),
            known => {
                let name = Self::NAMES
                    .iter()
                    .find(|(_, key)| key == known)
                    .map_or("unsupported", |(n, _)| n);
                f.write_str(name)
            }
        }
    }
}

impl Serialize for SourceKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for SourceKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <&str>::deserialize(deserializer)?;
        Ok(Self::parse(text))
    }
}

/// The `format` of a source declaration.
pub const SOURCE_FORMAT: &str = "navigate.data.source";
/// The one version of a source declaration.
pub const SOURCE_FORMAT_VERSION: u32 = 1;

/// The declaration of a source outside the registry. Its canonical bytes
/// (the rule of the manifest: integers only, declared field order, explicit
/// nulls) give the digest of [`SourceKey::Declared`]. Two hosts that
/// declare the same source with the same values get the same key, so the
/// correlation keys of their packages agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceDeclaration {
    /// [`SOURCE_FORMAT`] is implied; the version is [`SOURCE_FORMAT_VERSION`].
    pub version: u32,
    /// What the source holds.
    pub kind: LayerKind,
    /// Nominal ground sample distance, millimetres.
    pub ground_sample_mm: u32,
    /// Horizontal one-sigma that the provider states, millimetres.
    pub horizontal_mm: Option<u32>,
    /// Vertical one-sigma that the provider states, millimetres.
    pub vertical_mm: Option<u32>,
    /// The day of collection.
    pub collected: Option<DayNumber>,
    /// The digest of a stable identifier that the provider chooses, for
    /// example of the text `"agency:ortho-2024:sheet-17"`. A digest, not
    /// text, so the declaration holds no free string.
    pub provider_id: Sha256Digest,
}

impl SourceDeclaration {
    /// The canonical bytes.
    ///
    /// # Errors
    ///
    /// Refuses another version, an unsupported kind, and a zero ground
    /// sample distance.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, DataError> {
        if self.version != SOURCE_FORMAT_VERSION {
            return Err(DataError::Version {
                document: "source declaration",
                found: self.version,
                supported: SOURCE_FORMAT_VERSION,
            });
        }
        if self.kind == LayerKind::Unsupported {
            return Err(DataError::Unsupported {
                field: "layer kind",
            });
        }
        if self.ground_sample_mm == 0 {
            return Err(DataError::Invalid {
                field: "ground_sample_mm",
                detail: String::from("zero"),
            });
        }
        serde_json::to_vec(self).map_err(|error| DataError::Document {
            path: String::from(SOURCE_FORMAT),
            detail: alloc::format!("{error}"),
        })
    }

    /// The source key of this declaration.
    ///
    /// # Errors
    ///
    /// As [`Self::canonical_bytes`].
    pub fn key(&self) -> Result<SourceKey, DataError> {
        self.canonical_bytes()
            .map(|bytes| SourceKey::Declared(Sha256Digest::of(&bytes)))
    }
}

/// What the provider states about the version of the content.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Revision {
    /// The provider states no version. The blob digests still identify the
    /// content.
    Unversioned,
    /// A numbered release of the provider.
    Release(u32),
    /// The collection date that the provider states.
    Collected(DayNumber),
}

#[cfg(test)]
mod tests;
