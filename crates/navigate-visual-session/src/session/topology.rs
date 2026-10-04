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

impl VisualSession {
    /// Undirected links with their declared position error, in metres.
    pub(super) fn links(&self) -> BTreeMap<KeyframeId, Vec<(KeyframeId, f64)>> {
        let mut links: BTreeMap<KeyframeId, Vec<(KeyframeId, f64)>> = BTreeMap::new();
        let mut add = |a: KeyframeId, b: KeyframeId, w: f64| {
            links.entry(a).or_default().push((b, w));
            links.entry(b).or_default().push((a, w));
        };
        for segment in self.segments.values() {
            for pair in segment.keyframes.windows(2) {
                let (Some(a), Some(b)) = (pair.first(), pair.get(1)) else {
                    continue;
                };
                if let (Some(ka), Some(kb)) = (self.keyframes.get(a), self.keyframes.get(b)) {
                    add(
                        *a,
                        *b,
                        ka.odometry.drift_to(&kb.odometry, &self.config.drift).0,
                    );
                }
            }
        }
        for closure in &self.closures {
            add(closure.earlier, closure.later, closure.sigma_m);
        }
        links
    }

    /// Position bound of each keyframe, or `None` when no anchor reaches it.
    pub(super) fn bounds(&self) -> BTreeMap<KeyframeId, Option<f64>> {
        let links = self.links();
        let mut best: BTreeMap<KeyframeId, f64> = BTreeMap::new();
        let mut heap = BinaryHeap::new();
        let sources = self
            .anchors
            .iter()
            .map(|(id, a)| (*id, a.budget.horizontal_m() + a.transfer_m))
            .chain(self.frozen.iter().map(|(id, f)| (*id, f.sigma_m)));
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
            for (next, weight) in links.get(&id).map(Vec::as_slice).unwrap_or_default() {
                let candidate = value + weight;
                if best.get(next).is_none_or(|b| candidate < *b) {
                    best.insert(*next, candidate);
                    heap.push(Visit(candidate, *next));
                }
            }
        }
        self.keyframes
            .keys()
            .map(|id| (*id, best.get(id).copied()))
            .collect()
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
                for (next, _) in links.get(&id).map(Vec::as_slice).unwrap_or_default() {
                    if !component.contains_key(next) {
                        stack.push(*next);
                    }
                }
            }
        }
        component
    }

    /// Number of accepted anchors in each component. Kept priors repeat
    /// earlier anchor evidence, so they do not count.
    pub(super) fn anchor_counts(&self) -> BTreeMap<KeyframeId, usize> {
        self.count_roots(self.anchors.keys())
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
