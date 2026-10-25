//! Carry fixed map correspondences without adding independent map evidence.
use super::metrics;
use crate::Error;
use navigate_visual::{CameraModel, Estimate, Frame, PixelMatch, PointTracker, ReferenceView};

pub(super) struct Tracks {
    previous: Frame,
    pairs: Vec<PixelMatch>,
    age: usize,
}
impl Tracks {
    pub fn inputs(&self) -> (&Frame, &[PixelMatch]) {
        (&self.previous, &self.pairs)
    }
    pub fn refresh_due(&self, interval: usize) -> bool {
        self.age.is_multiple_of(interval)
    }
    pub fn eligible(&self, frame: &Frame, interval: usize) -> bool {
        self.age < interval
            && frame.camera == self.previous.camera
            && frame.stamp.capture_time_ns > self.previous.stamp.capture_time_ns
            && frame.stamp.capture_time_ns - self.previous.stamp.capture_time_ns <= 200_000_000
    }
    pub fn propagate_blocking(
        &self,
        frame: &Frame,
        tracker: &mut dyn PointTracker,
    ) -> Result<Vec<PixelMatch>, Error> {
        let points: Vec<_> = self.pairs.iter().map(|p| p.query).collect();
        let tracked = tracker.track_points_blocking(&self.previous.image, &frame.image, &points)?;
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
    pub fn update(
        previous: Frame,
        pairs: Vec<PixelMatch>,
        reference: (&ReferenceView, &CameraModel),
        estimate: &Estimate,
        old: Option<&Self>,
        refreshed: bool,
    ) -> Self {
        let pairs = pairs
            .into_iter()
            .filter(|p| {
                metrics::residual(*p, &previous.camera, reference, &estimate.pose)
                    .is_some_and(|e| e <= 2.5)
            })
            .collect();
        Self {
            previous,
            pairs,
            age: if refreshed {
                1
            } else {
                old.map_or(1, |old| old.age.wrapping_add(1))
            },
        }
    }
}

pub(super) fn select_map(
    track: &Result<Estimate, Error>,
    map: &Result<Estimate, Error>,
    limit: Option<f64>,
) -> (bool, Option<f64>) {
    let distance = match (track, map) {
        (Ok(track), Ok(map)) => Some((track.pose.position - map.pose.position).norm()),
        _ => None,
    };
    let conflict = distance
        .zip(limit)
        .is_some_and(|(distance, limit)| distance > limit);
    (track.is_err() || (map.is_ok() && !conflict), distance)
}
