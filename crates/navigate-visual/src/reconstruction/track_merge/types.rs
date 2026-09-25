//! Provenance and failures for bounded image-track joining.
use super::{ImageTracks, ReconstructionError};
use crate::VisualError;
use crate::reconstruction::ScenePointMatch;

/// The exact source feature that contributes to a joined track.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackSource {
    /// Verified source image-group digest. The host retains its full document.
    pub group_sha256: String,
    /// Feature identity inside that source group.
    pub feature_id: u64,
}

/// A selected graph for one bounded geometry calculation.
#[derive(Clone, Debug)]
pub struct MergedTrackSelection {
    /// Selected source cameras and tracks. Feature IDs are stable in this merger.
    pub graph: ImageTracks,
    /// Source feature identities, in the same order as the selected tracks.
    pub source_tracks: Vec<Vec<TrackSource>>,
    /// Tracks with enough selected observations that were omitted by the budget
    /// or spatial sampling. Omission does not reject their geometric support.
    pub deferred_tracks: usize,
}

/// Scene point links for one exact source observation.
#[derive(Clone, Debug)]
pub struct SceneTrackLinks {
    /// Unique source pixels linked to the supplied estimated scene points.
    pub matches: Vec<ScenePointMatch>,
    /// Point IDs omitted because multiple features propose the same source pixel.
    pub ambiguous_feature_ids: Vec<u64>,
}

/// Image groups cannot be joined or selected with these inputs.
#[derive(Debug, thiserror::Error)]
pub enum TrackMergeError {
    /// Intrinsic validation failed.
    #[error("invalid track-merge camera: {source}")]
    Camera {
        /// Calibration error.
        #[source]
        source: VisualError,
    },
    /// A source group failed validation.
    #[error("invalid source graph {group_sha256}: {source}")]
    Graph {
        /// Source group digest.
        group_sha256: String,
        /// Graph validation failure.
        #[source]
        source: ReconstructionError,
    },
    /// A source group identity is invalid or repeated.
    #[error("invalid source group {group_sha256}: {reason}")]
    Group {
        /// Supplied digest.
        group_sha256: String,
        /// Validation reason.
        reason: &'static str,
    },
    /// The complete graph would exceed its resource budget.
    #[error(
        "track merge exceeds bounds: {groups} groups, {observations} images, {tracks} tracks, {links} links"
    )]
    Limits {
        /// Number of groups after insertion.
        groups: usize,
        /// Number of distinct observations after insertion.
        observations: usize,
        /// Upper bound on tracks after insertion.
        tracks: usize,
        /// Number of supplied image links after insertion.
        links: usize,
    },
    /// A scene point does not identify a valid feature in this graph.
    #[error("invalid merged feature {feature_id}: {reason}")]
    Feature {
        /// Supplied feature identity.
        feature_id: u64,
        /// Validation reason.
        reason: &'static str,
    },
    /// A camera selection or its resource budget is invalid.
    #[error("invalid track selection: {reason}")]
    Selection {
        /// Validation reason.
        reason: &'static str,
    },
}
