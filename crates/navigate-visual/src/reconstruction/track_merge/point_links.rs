//! Source pixels for camera fitting with conditional scene points.
use super::{BTreeMap, BTreeSet, ImageTrackMerger, SceneTrackLinks, TrackMergeError};
use crate::{LocalScenePoint, reconstruction::ScenePointMatch};
impl ImageTrackMerger {
    /// Link one source image to a set of estimated points from this graph.
    ///
    /// Features with an identical query pixel are all omitted as ambiguous.
    /// Repeated calls do not change the graph or its image support.
    ///
    /// # Errors
    /// Rejects unknown observations, unknown or repeated feature IDs, invalid
    /// point positions, and point sets larger than 65,536 entries.
    pub fn scene_point_links(
        &self,
        observation_sha256: &str,
        points: &[LocalScenePoint],
    ) -> Result<SceneTrackLinks, TrackMergeError> {
        let Some(&camera) = self
            .observation_indices
            .get(&observation_sha256.to_ascii_lowercase())
        else {
            return Err(TrackMergeError::Selection {
                reason: "unknown camera for scene point links",
            });
        };
        if points.len() > 65536 {
            return Err(TrackMergeError::Selection {
                reason: "too many scene points for one camera fit",
            });
        }
        let mut features = BTreeSet::new();
        let mut pixels = BTreeMap::<_, usize>::new();
        let mut matches = Vec::new();
        let bits = |v: f64| if v == 0.0 { 0 } else { v.to_bits() };
        for point in points {
            let invalid = || TrackMergeError::Feature {
                feature_id: point.feature_id,
                reason: "unknown or repeated feature ID, or nonfinite point position",
            };
            let index = usize::try_from(point.feature_id).map_err(|_| invalid())?;
            let track = self.tracks.get(index).ok_or_else(invalid)?;
            if !features.insert(point.feature_id) || !point.position.iter().all(|v| v.is_finite()) {
                return Err(invalid());
            }
            if let Some(&pixel) = track.pixels.get(&camera) {
                let count = pixels.entry((bits(pixel.x), bits(pixel.y))).or_default();
                *count = count.saturating_add(1);
                matches.push(ScenePointMatch {
                    feature_id: point.feature_id,
                    position: point.position,
                    pixel,
                });
            }
        }
        let mut ambiguous_feature_ids = Vec::new();
        matches.retain(|p| {
            let unique = pixels.get(&(bits(p.pixel.x), bits(p.pixel.y))) == Some(&1);
            if !unique {
                ambiguous_feature_ids.push(p.feature_id);
            }
            unique
        });
        Ok(SceneTrackLinks {
            matches,
            ambiguous_feature_ids,
        })
    }
}
