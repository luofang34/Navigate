//! Associate dense pair proposals while keeping feature and image identities.
use super::dense_flow::Flow;
use super::*;
use crate::PixelMatch;
/// A bounded image-track adapter for dense pair correspondences.
///
/// Local affine interpolation links nearby proposals. This is an association
/// estimate, not geometry validation. Reverse interpolation uses the same pairs
/// and is not independent evidence. Other trackers can supply [`ImageTracks`]
/// directly. A builder owns one group; start a new one at a group boundary.
/// Completed tracks need three observations. Active two-image links remain
/// available for initialization; abandoned two-image links are discarded.
pub struct DenseTrackBuilder {
    camera: CameraModel,
    graph: ImageTracks,
    active: Vec<usize>,
    next_feature: u64,
}
fn cell(p: Vector2<f64>) -> (i32, i32) {
    ((p.x / 12.0).floor() as i32, (p.y / 12.0).floor() as i32)
}
impl DenseTrackBuilder {
    /// Start an empty association group with one common calibrated image size.
    ///
    /// # Errors
    /// Rejects invalid camera intrinsics.
    pub fn new(camera: CameraModel) -> Result<Self, ReconstructionError> {
        camera
            .validate()
            .map_err(|source| ReconstructionError::Camera { source })?;
        Ok(Self {
            camera,
            graph: ImageTracks {
                observation_sha256: Vec::new(),
                tracks: Vec::new(),
            },
            active: Vec::new(),
            next_feature: 0,
        })
    }
    /// Append a new observation and pairs from the immediately preceding image.
    /// Supply no pairs for the first image. Empty pairs break active links.
    ///
    /// # Errors
    /// Rejects repeated digests, invalid pixels, or a full group. On error the
    /// builder stays unchanged. A group accepts at most 129 cameras.
    pub fn push(
        &mut self,
        digest: String,
        pairs: &[PixelMatch],
    ) -> Result<(), ReconstructionError> {
        self.validate(&digest, pairs)?;
        let frame = self.graph.observation_sha256.len();
        if frame > 0 {
            self.advance(pairs, frame);
        }
        self.graph.observation_sha256.push(digest);
        Ok(())
    }
    /// Copy linked features and exact source digests. Retain this graph with its
    /// matcher provenance. Exporting it does not add independent evidence.
    pub fn snapshot(&self) -> ImageTracks {
        self.graph.clone()
    }
    fn validate(&self, digest: &str, pairs: &[PixelMatch]) -> Result<(), ReconstructionError> {
        let index = self.graph.observation_sha256.len();
        if index >= 129
            || self
                .graph
                .tracks
                .len()
                .saturating_add(pairs.len().min(4096))
                > 65536
            || pairs.len() > 65536
        {
            return Err(ReconstructionError::Limits {
                cameras: index.saturating_add(1),
                tracks: self.graph.tracks.len(),
                observations: pairs.len(),
            });
        }
        if digest.len() != 64
            || !digest.bytes().all(|c| c.is_ascii_hexdigit())
            || self
                .graph
                .observation_sha256
                .iter()
                .any(|v| v.eq_ignore_ascii_case(digest))
            || (index == 0 && !pairs.is_empty())
        {
            return Err(ReconstructionError::Observation {
                index,
                reason: "invalid or repeated digest, or first image has preceding pairs",
            });
        }
        if pairs.iter().any(|p| {
            [p.reference, p.query].iter().any(|p| {
                !(0.0..f64::from(self.camera.width)).contains(&p.x)
                    || !(0.0..f64::from(self.camera.height)).contains(&p.y)
            })
        }) {
            return Err(ReconstructionError::Observation {
                index,
                reason: "correspondence outside calibrated images",
            });
        }
        Ok(())
    }
    fn advance(&mut self, pairs: &[PixelMatch], frame: usize) {
        let flow = Flow::new(pairs);
        let reverse_pairs: Vec<_> = pairs
            .iter()
            .map(|p| PixelMatch {
                reference: p.query,
                query: p.reference,
            })
            .collect();
        let reverse = Flow::new(&reverse_pairs);
        let mut next = Vec::new();
        let mut occupied = BTreeSet::new();
        for &id in &self.active {
            let Some(last) = self.graph.tracks[id].observations.last() else {
                continue;
            };
            let pixel = last.pixel;
            let Some(query) = flow.at(pixel) else {
                continue;
            };
            let Some(back) = reverse.at(query) else {
                continue;
            };
            if (back - pixel).norm() > 1.0
                || occupied.contains(&cell(query))
                || !(0.0..f64::from(self.camera.width)).contains(&query.x)
                || !(0.0..f64::from(self.camera.height)).contains(&query.y)
            {
                continue;
            }
            occupied.insert(cell(query));
            self.graph.tracks[id]
                .observations
                .push(ScenePointObservation {
                    camera_index: frame,
                    pixel: query,
                });
            next.push(id);
        }
        for pair in pairs {
            if next.len() >= 4096 {
                break;
            }
            if occupied.contains(&cell(pair.query)) {
                continue;
            }
            let Some(back) = reverse.at(pair.query) else {
                continue;
            };
            if (back - pair.reference).norm() > 1.0 {
                continue;
            }
            occupied.insert(cell(pair.query));
            next.push(self.graph.tracks.len());
            self.graph.tracks.push(ImageTrack {
                feature_id: self.next_feature,
                observations: vec![
                    ScenePointObservation {
                        camera_index: frame - 1,
                        pixel: pair.reference,
                    },
                    ScenePointObservation {
                        camera_index: frame,
                        pixel: pair.query,
                    },
                ],
            });
            self.next_feature = self.next_feature.wrapping_add(1);
        }
        self.active = next;
        self.retain_continuing_tracks();
    }
    fn retain_continuing_tracks(&mut self) {
        let active: BTreeSet<_> = self.active.iter().copied().collect();
        let mut next = Vec::new();
        let mut retained = Vec::new();
        for (id, track) in self.graph.tracks.drain(..).enumerate() {
            if track.observations.len() >= 3 || active.contains(&id) {
                if active.contains(&id) {
                    next.push(retained.len());
                }
                retained.push(track);
            }
        }
        self.graph.tracks = retained;
        self.active = next;
    }
}
#[cfg(test)]
mod tests;
