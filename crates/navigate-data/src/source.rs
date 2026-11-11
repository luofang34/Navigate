//! Layer kinds, encodings, and the registry of sources.

use serde::{Deserialize, Serialize};

use crate::DayNumber;

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

/// A source from the registry of known sources. A new provider is a new
/// value here. The key, not a URL or a host label, names the source in the
/// package identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKey {
    /// Esri World Imagery.
    EsriWorldImagery,
    /// Mapzen Terrarium elevation tiles (the AWS terrain tiles).
    MapzenTerrarium,
    /// Data that a test or a simulator makes. A navigator never takes it
    /// as evidence of the real world.
    Synthetic,
    /// A source that this crate does not know.
    #[serde(other)]
    Unsupported,
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
