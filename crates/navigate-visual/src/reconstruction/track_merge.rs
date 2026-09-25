//! Join image groups through shared observations without changing their geometry.
use super::{ImageTracks, ReconstructionError, validation};
use crate::CameraModel;
use nalgebra::Vector2;
use std::collections::{BTreeMap, BTreeSet};
mod association;
mod point_links;
mod selection;
mod types;
pub use types::{MergedTrackSelection, SceneTrackLinks, TrackMergeError, TrackSource};

struct Track {
    pixels: BTreeMap<usize, Vector2<f64>>,
    sources: Vec<TrackSource>,
}

/// A bounded graph of image links shared across successive source groups.
///
/// Joining requires unique pixel agreement in two shared observations. All
/// shared pixels must agree within one pixel. Ambiguous links remain separate.
/// These are association proposals, not verified 3D points or independent
/// measurements. Supply one connected candidate's source groups in order.
pub struct ImageTrackMerger {
    camera: CameraModel,
    observations: Vec<String>,
    observation_indices: BTreeMap<String, usize>,
    tracks: Vec<Track>,
    previous: Vec<usize>,
    groups: BTreeSet<String>,
    source_links: usize,
}

impl ImageTrackMerger {
    /// Start a graph with one common pinhole camera model.
    ///
    /// # Errors
    /// Rejects invalid calibration.
    pub fn new(camera: CameraModel) -> Result<Self, TrackMergeError> {
        camera
            .validate()
            .map_err(|source| TrackMergeError::Camera { source })?;
        Ok(Self {
            camera,
            observations: Vec::new(),
            observation_indices: BTreeMap::new(),
            tracks: Vec::new(),
            previous: Vec::new(),
            groups: BTreeSet::new(),
            source_links: 0,
        })
    }

    /// Append one verified source group. On error the graph stays unchanged.
    ///
    /// The host verifies the group digest and retains matcher provenance.
    /// Digests are case insensitive. The same group cannot add evidence twice.
    ///
    /// # Errors
    /// Rejects invalid groups, repeated group identities, or resource limits.
    /// Limits are 256 groups, 4,096 observations, 1,048,576 joined tracks, and
    /// 4,000,000 source image links. Local group bounds also apply.
    pub fn push(&mut self, group_sha256: &str, graph: &ImageTracks) -> Result<(), TrackMergeError> {
        let links = self.validate(group_sha256, graph)?;
        let digest = group_sha256.to_ascii_lowercase();
        let indices: Vec<_> = graph
            .observation_sha256
            .iter()
            .map(|id| {
                let id = id.to_ascii_lowercase();
                if let Some(&index) = self.observation_indices.get(&id) {
                    return index;
                }
                let index = self.observations.len();
                self.observations.push(id.clone());
                self.observation_indices.insert(id, index);
                index
            })
            .collect();
        let shared: BTreeSet<_> = indices.iter().copied().collect();
        let grid = association::grid(&self.tracks, &self.previous, &shared);
        let mut next = BTreeSet::new();
        for source in &graph.tracks {
            let pixels: BTreeMap<_, _> = source
                .observations
                .iter()
                .map(|o| (indices[o.camera_index], o.pixel))
                .collect();
            let proposed = association::candidate(&self.tracks, &grid, &pixels);
            let index = if let Some(index) = proposed.filter(|i| !next.contains(i)) {
                index
            } else {
                let index = self.tracks.len();
                self.tracks.push(Track {
                    pixels: BTreeMap::new(),
                    sources: Vec::new(),
                });
                index
            };
            let target = &mut self.tracks[index];
            for (camera, pixel) in pixels {
                target.pixels.entry(camera).or_insert(pixel);
            }
            target.sources.push(TrackSource {
                group_sha256: digest.clone(),
                feature_id: source.feature_id,
            });
            next.insert(index);
        }
        self.previous = next.into_iter().collect();
        self.groups.insert(digest);
        self.source_links = self.source_links.saturating_add(links);
        Ok(())
    }

    /// Source observation identities in first-seen order.
    pub fn observation_sha256(&self) -> &[String] {
        &self.observations
    }

    fn validate(&self, digest: &str, graph: &ImageTracks) -> Result<usize, TrackMergeError> {
        if digest.len() != 64
            || !digest.bytes().all(|c| c.is_ascii_hexdigit())
            || self.groups.contains(&digest.to_ascii_lowercase())
        {
            return Err(TrackMergeError::Group {
                group_sha256: digest.to_owned(),
                reason: "invalid or repeated group digest",
            });
        }
        validation::graph(&self.camera, graph).map_err(|source| TrackMergeError::Graph {
            group_sha256: digest.to_owned(),
            source,
        })?;
        let links = graph
            .tracks
            .iter()
            .fold(0usize, |n, t| n.saturating_add(t.observations.len()));
        let observations = self.observations.len().saturating_add(
            graph
                .observation_sha256
                .iter()
                .filter(|id| {
                    !self
                        .observation_indices
                        .contains_key(&id.to_ascii_lowercase())
                })
                .count(),
        );
        if self.groups.len() >= 256
            || observations > 4096
            || self.tracks.len().saturating_add(graph.tracks.len()) > 1_048_576
            || self.source_links.saturating_add(links) > 4_000_000
        {
            return Err(TrackMergeError::Limits {
                groups: self.groups.len().saturating_add(1),
                observations,
                tracks: self.tracks.len().saturating_add(graph.tracks.len()),
                links: self.source_links.saturating_add(links),
            });
        }
        Ok(links)
    }
}

#[cfg(test)]
mod tests;
