//! Layers and their sources.

use alloc::string::String;

use serde::{Deserialize, Serialize};

/// What a layer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum LayerKind {
    /// Orthoimagery.
    Imagery,
    /// Elevation of the surface.
    Elevation,
}

/// How the blobs of a layer are encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Encoding {
    /// PNG image.
    Png,
    /// JPEG image.
    Jpeg,
    /// WebP image.
    Webp,
    /// PNG with Terrarium heights: `R·256 + G + B/256 − 32768` metres.
    TerrariumPng,
}

/// The source of a layer. The source identity and the revision name the
/// content, not a URL, so two hosts that fetch from different addresses
/// give the same identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceRef {
    /// Stable identifier of the provider product, for example
    /// `mapzen-terrarium`.
    pub id: String,
    /// The provider release or revision. `unknown` when the provider does
    /// not state one.
    pub revision: String,
    /// The attribution text that the licence requires.
    pub attribution: String,
}
