//! Bounded image-track selection with exact source camera identities.
use super::{
    BTreeMap, BTreeSet, ImageTrackMerger, ImageTracks, MergedTrackSelection, TrackMergeError,
};
use crate::ScenePointObservation;
use crate::reconstruction::ImageTrack;
impl ImageTrackMerger {
    /// Select at most 65,536 tracks for 2 to 129 requested source cameras.
    ///
    /// Longer tracks take priority. A track must add coverage in at least one
    /// image cell. Feature IDs stay stable across repeated selections. The full
    /// source groups remain necessary evidence; the result is a work selection.
    ///
    /// # Errors
    /// Rejects unknown or repeated cameras and invalid work budgets.
    pub fn select(
        &self,
        observation_sha256: &[String],
        max_tracks: usize,
        cell_size_px: u32,
    ) -> Result<MergedTrackSelection, TrackMergeError> {
        let indices = self.selection_indices(observation_sha256, max_tracks, cell_size_px)?;
        let mut candidates: Vec<_> = self
            .tracks
            .iter()
            .enumerate()
            .filter_map(|(id, track)| {
                let mut observations: Vec<_> = track
                    .pixels
                    .iter()
                    .filter_map(|(source, &pixel)| {
                        indices
                            .get(source)
                            .map(|&camera_index| ScenePointObservation {
                                camera_index,
                                pixel,
                            })
                    })
                    .collect();
                observations.sort_by_key(|o| o.camera_index);
                (observations.len() >= 2).then_some((id, id.to_string(), observations))
            })
            .collect();
        candidates.sort_by(|a, b| b.2.len().cmp(&a.2.len()).then_with(|| a.1.cmp(&b.1)));
        let supported = candidates.len();
        let mut occupied = BTreeSet::new();
        let mut tracks = Vec::new();
        let mut source_tracks = Vec::new();
        for (id, _, observations) in candidates {
            let cells: Vec<_> = observations
                .iter()
                .map(|o| {
                    (
                        o.camera_index,
                        (o.pixel.x / f64::from(cell_size_px)).floor() as u32,
                        (o.pixel.y / f64::from(cell_size_px)).floor() as u32,
                    )
                })
                .collect();
            if cells.iter().all(|cell| occupied.contains(cell)) {
                continue;
            }
            occupied.extend(cells);
            tracks.push(ImageTrack {
                feature_id: id as u64,
                observations,
            });
            source_tracks.push(self.tracks[id].sources.clone());
            if tracks.len() == max_tracks {
                break;
            }
        }
        Ok(MergedTrackSelection {
            deferred_tracks: supported.saturating_sub(tracks.len()),
            source_tracks,
            graph: ImageTracks {
                observation_sha256: observation_sha256
                    .iter()
                    .map(|s| s.to_ascii_lowercase())
                    .collect(),
                tracks,
            },
        })
    }
    fn selection_indices(
        &self,
        ids: &[String],
        max_tracks: usize,
        cell_size_px: u32,
    ) -> Result<BTreeMap<usize, usize>, TrackMergeError> {
        let invalid = || TrackMergeError::Selection {
            reason: "expected unique known cameras, 1–65,536 tracks, and 1–4,096 pixel cells",
        };
        if !(2..=129).contains(&ids.len())
            || !(1..=65536).contains(&max_tracks)
            || !(1..=4096).contains(&cell_size_px)
        {
            return Err(invalid());
        }
        let mut indices = BTreeMap::new();
        for (index, id) in ids.iter().enumerate() {
            let Some(&source) = self.observation_indices.get(&id.to_ascii_lowercase()) else {
                return Err(invalid());
            };
            if indices.insert(source, index).is_some() {
                return Err(invalid());
            }
        }
        Ok(indices)
    }
}
