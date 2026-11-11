//! The descriptor: what a package says about itself outside its identity.

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::LayerKey;

/// Labels, attribution, licence text, and host notes of a package. It is
/// stored beside the manifest, a host shows it, and it never changes the
/// package identity. A navigator never uses it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Descriptor {
    /// Release identifier that the host gives, for example for
    /// `MapRevision::release_id`.
    pub release_id: String,
    /// Labels and terms of each layer.
    pub layers: Vec<LayerLabel>,
    /// Free text from the host, for example the fetch time.
    pub note: String,
}

/// The label and terms of one layer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerLabel {
    /// The layer.
    pub key: LayerKey,
    /// A human label.
    pub label: String,
    /// The attribution text that the licence requires.
    pub attribution: String,
    /// The licence or terms of use.
    pub licence: String,
}
