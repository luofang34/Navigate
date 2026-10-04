//! Connectivity of keyframes and position uncertainty bounds.
//!
//! A bound is the smallest sum of declared one-sigma errors along a path of
//! odometry and closure links from an anchor. Linear addition does not assume
//! independent errors, so the bound is conservative. It is not a covariance.

use super::VisualSession;
use crate::store::KeyframeId;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

#[derive(PartialEq)]
struct Visit(f64, KeyframeId);

impl Eq for Visit {}

impl PartialOrd for Visit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Visit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .total_cmp(&self.0)
            .then_with(|| other.1.cmp(&self.1))
    }
}

/// One undirected link with its declared drift.
#[derive(Clone, Copy, Debug)]
pub(super) struct Link {
    pub to: KeyframeId,
    pub m: f64,
    pub rad: f64,
    /// Camera travel along the link, in metres.
    pub length_m: f64,
}

impl VisualSession {
    /// Undirected odometry and closure links with their declared errors.
    pub(super) fn links(&self) -> BTreeMap<KeyframeId, Vec<Link>> {
        let mut links: BTreeMap<KeyframeId, Vec<Link>> = BTreeMap::new();
        let mut add = |a: KeyframeId, b: KeyframeId, m: f64, rad: f64, length_m: f64| {
            links.entry(a).or_default().push(Link {
                to: b,
                m,
                rad,
                length_m,
            });
            links.entry(b).or_default().push(Link {
                to: a,
                m,
                rad,
                length_m,
            });
        };
        for segment in self.segments.values() {
            for pair in segment.keyframes.windows(2) {
                let (Some(a), Some(b)) = (pair.first(), pair.get(1)) else {
                    continue;
                };
                if let (Some(ka), Some(kb)) = (self.keyframes.get(a), self.keyframes.get(b)) {
                    let (m, rad) = ka.odometry.drift_to(&kb.odometry, &self.config.drift);
                    add(
                        *a,
                        *b,
                        m,
                        rad,
                        (kb.odometry.path_m - ka.odometry.path_m).abs(),
                    );
                }
            }
        }
        for closure in &self.closures {
            let length = closure.measured.translation.vector.norm();
            add(
                closure.earlier,
                closure.later,
                closure.sigma_m,
                closure.sigma_rad,
                length,
            );
        }
        links
    }

    /// Position bound of each keyframe, or `None` when no anchor reaches it.
    ///
    /// Relative motion is measured in the camera frame, so an attitude error
    /// turns each metre of travel into a position error. Along each link the
    /// bound grows by the declared drift and by the travel times the sine of
    /// the attitude bound at the start of the link. Ground planes do not
    /// observe heading, so this uses the attitude bound from map evidence.
    pub(super) fn bounds(&self) -> BTreeMap<KeyframeId, Option<f64>> {
        let attitude = self.attitude_bounds();
        let sources = self
            .anchors
            .iter()
            .map(|(id, a)| (*id, a.budget.horizontal_m() + a.transfer_m))
            .chain(self.frozen.iter().map(|(id, f)| (*id, f.sigma_m)));
        self.propagate(sources, |from, link| {
            let angle = attitude
                .get(&from)
                .copied()
                .flatten()
                .unwrap_or(1.0)
                .min(1.0);
            link.m + link.length_m * angle.sin()
        })
    }

    /// Attitude bound from map evidence: anchors and kept priors. Anchors
    /// give their heading error here; ground planes do not observe heading.
    pub(super) fn attitude_bounds(&self) -> BTreeMap<KeyframeId, Option<f64>> {
        let anchors = self
            .anchors
            .iter()
            .map(|(id, a)| (*id, a.budget.heading_rad + a.transfer_rad));
        self.propagate(anchors.chain(self.frozen_sources()), |_, link| link.rad)
    }

    /// Tilt bound from map evidence only: anchors and kept priors.
    pub(super) fn map_tilt_bounds(&self) -> BTreeMap<KeyframeId, Option<f64>> {
        self.propagate(self.map_tilt_sources(), |_, link| link.rad)
    }

    /// Tilt bound from map evidence and ground-plane observations.
    pub(super) fn tilt_bounds(&self) -> BTreeMap<KeyframeId, Option<f64>> {
        let ground = self.grounds.iter().map(|(id, g)| (*id, g.sigma_rad));
        self.propagate(self.map_tilt_sources().chain(ground), |_, link| link.rad)
    }

    fn map_tilt_sources(&self) -> impl Iterator<Item = (KeyframeId, f64)> + '_ {
        self.anchors
            .iter()
            .map(|(id, a)| (*id, a.budget.tilt_rad + a.transfer_rad))
            .chain(self.frozen_sources())
    }

    fn frozen_sources(&self) -> impl Iterator<Item = (KeyframeId, f64)> + '_ {
        self.frozen.iter().map(|(id, f)| (*id, f.sigma_rad))
    }

    /// Smallest source value plus link weights along any path.
    pub(super) fn propagate(
        &self,
        sources: impl Iterator<Item = (KeyframeId, f64)>,
        weight: impl Fn(KeyframeId, &Link) -> f64,
    ) -> BTreeMap<KeyframeId, Option<f64>> {
        let links = self.links();
        let mut best: BTreeMap<KeyframeId, f64> = BTreeMap::new();
        let mut heap = BinaryHeap::new();
        for (id, value) in sources {
            if best.get(&id).is_none_or(|b| value < *b) {
                best.insert(id, value);
                heap.push(Visit(value, id));
            }
        }
        while let Some(Visit(value, id)) = heap.pop() {
            if best.get(&id).is_some_and(|b| value > *b) {
                continue;
            }
            for link in links.get(&id).map(Vec::as_slice).unwrap_or_default() {
                let candidate = value + weight(id, link);
                if best.get(&link.to).is_none_or(|b| candidate < *b) {
                    best.insert(link.to, candidate);
                    heap.push(Visit(candidate, link.to));
                }
            }
        }
        self.keyframes
            .keys()
            .map(|id| (*id, best.get(id).copied()))
            .collect()
    }

    /// Capture times of the anchors in each component, in order.
    pub(super) fn anchor_times(&self) -> BTreeMap<KeyframeId, Vec<u64>> {
        let components = self.components();
        let mut times: BTreeMap<KeyframeId, Vec<u64>> = BTreeMap::new();
        for (id, anchor) in &self.anchors {
            if let Some(root) = components.get(id) {
                times.entry(*root).or_default().push(anchor.capture_ns);
            }
        }
        times.values_mut().for_each(|t| t.sort_unstable());
        times
    }

    /// Connected component of each keyframe, named by its smallest keyframe.
    pub(super) fn components(&self) -> BTreeMap<KeyframeId, KeyframeId> {
        let links = self.links();
        let mut component = BTreeMap::new();
        for start in self.keyframes.keys() {
            if component.contains_key(start) {
                continue;
            }
            let mut stack = vec![*start];
            while let Some(id) = stack.pop() {
                if component.insert(id, *start).is_some() {
                    continue;
                }
                for link in links.get(&id).map(Vec::as_slice).unwrap_or_default() {
                    if !component.contains_key(&link.to) {
                        stack.push(link.to);
                    }
                }
            }
        }
        component
    }

    /// Components with an anchor or a kept prior: they have a map gauge.
    pub(super) fn anchored_roots(&self) -> BTreeMap<KeyframeId, usize> {
        self.count_roots(self.anchors.keys().chain(self.frozen.keys()))
    }

    fn count_roots<'a>(
        &self,
        ids: impl Iterator<Item = &'a KeyframeId>,
    ) -> BTreeMap<KeyframeId, usize> {
        let components = self.components();
        let mut counts = BTreeMap::new();
        for id in ids {
            if let Some(root) = components.get(id) {
                *counts.entry(*root).or_insert(0) += 1;
            }
        }
        counts
    }
}
