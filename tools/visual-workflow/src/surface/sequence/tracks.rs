//! A tracked query pixel keeps its original map pixel and surface sample.
use crate::Error;
use navigate_visual::{
    Frame, LocalizerConfig, OrthographicEstimate, OrthographicReference, PixelMatch, PointTracker,
};

pub(in crate::surface) const MAX_TRACK_GAP_NS: u64 = 200_000_000;

pub(in crate::surface) struct Tracks {
    pub previous: Frame,
    pub origin: String,
    pub pairs: Vec<PixelMatch>,
    pub age: usize,
}
impl Tracks {
    pub fn eligible(&self, frame: &Frame, interval: usize) -> bool {
        self.age < interval
            && frame.camera == self.previous.camera
            && frame.stamp.sequence > self.previous.stamp.sequence
            && frame.stamp.capture_time_ns > self.previous.stamp.capture_time_ns
            && frame.stamp.capture_time_ns - self.previous.stamp.capture_time_ns <= MAX_TRACK_GAP_NS
    }
    pub fn propagate_blocking(
        &self,
        frame: &Frame,
        tracker: &mut dyn PointTracker,
    ) -> Result<Vec<PixelMatch>, Error> {
        let points: Vec<_> = self.pairs.iter().map(|p| p.query).collect();
        let tracked = tracker.track_points_blocking(&self.previous.image, &frame.image, &points)?;
        if tracked.len() != points.len() {
            return Err(Error::Input(
                "surface tracker returned a different feature count".into(),
            ));
        }
        Ok(self
            .pairs
            .iter()
            .zip(tracked)
            .filter_map(|(p, q)| {
                q.map(|query| PixelMatch {
                    reference: p.reference,
                    query,
                })
            })
            .collect())
    }
    pub fn new(
        frame: Frame,
        reference: &OrthographicReference,
        estimate: &OrthographicEstimate,
        pairs: Vec<PixelMatch>,
        old: Option<&Self>,
        refreshed: bool,
    ) -> Self {
        let pairs = retain_inliers(&frame, reference, estimate, pairs);
        let origin = if refreshed {
            frame.evidence_sha256()
        } else {
            old.map_or_else(|| frame.evidence_sha256(), |t| t.origin.clone())
        };
        let age = if refreshed {
            1
        } else {
            old.map_or(1, |t| t.age.wrapping_add(1))
        };
        Self {
            previous: frame,
            origin,
            pairs,
            age,
        }
    }
}
pub(in crate::surface) fn retain_inliers(
    frame: &Frame,
    reference: &OrthographicReference,
    estimate: &OrthographicEstimate,
    pairs: Vec<PixelMatch>,
) -> Vec<PixelMatch> {
    let threshold = LocalizerConfig::default().inlier_threshold_px;
    pairs
        .into_iter()
        .filter(|pair| {
            reference
                .world_at(pair.reference)
                .and_then(|world| frame.camera.project(&estimate.pose, world))
                .is_some_and(|pixel| (pixel - pair.query).norm() <= threshold)
        })
        .collect()
}
#[cfg(test)]
mod tests;
