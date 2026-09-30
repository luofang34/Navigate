//! Image retrieval proposes reference IDs. It does not accept a camera pose.
use crate::MapRevision;
use image::RgbImage;

/// An entry in one pinned reference catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceId(pub u64);

/// Identity of the reference entries and their map data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceCatalog {
    /// Map data used to prepare the reference entries.
    pub map: MapRevision,
    /// SHA-256 of the catalog that maps IDs to reference data and geometry.
    pub manifest_sha256: String,
}

/// Retrieval input or backend failure, before geometric verification.
#[derive(Debug, thiserror::Error)]
pub enum RetrievalError {
    /// Invalid reference IDs or limits.
    #[error("invalid retrieval request: {0}")]
    Invalid(String),
    /// Model or device failure.
    #[error("image retrieval backend {backend} failed: {source}")]
    Backend {
        /// Backend, model and index identity.
        backend: String,
        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Replaceable appearance retrieval over a pinned reference catalog.
///
/// The host applies the navigation prior to select eligible reference IDs.
/// An adapter can use learned descriptors, classical features or another method.
/// Scores, model loading, preprocessing and device setup stay in the adapter.
pub trait ImageRetriever {
    /// Backend identity, including model and index content when applicable.
    fn identity(&self) -> &str;
    /// Catalog identity that the host must check before it resolves returned IDs.
    fn catalog(&self) -> &ReferenceCatalog;
    /// Rank up to `limit` eligible references for these image pixels.
    ///
    /// Empty results are allowed. Rank is not a geographic probability or a pose.
    /// Different queries or repeated calls can use the same underlying evidence.
    /// The host retains observation identity and performs geometric verification.
    ///
    /// # Errors
    /// Rejects invalid IDs, limits, images or execution failures.
    fn rank_blocking(
        &mut self,
        query: &RgbImage,
        eligible: &[ReferenceId],
        limit: usize,
    ) -> Result<Vec<ReferenceId>, RetrievalError>;
}
