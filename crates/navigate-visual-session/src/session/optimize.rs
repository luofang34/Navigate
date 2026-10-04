//! Pose-graph construction, solution, outlier retraction, and revision events.

use super::VisualSession;
use crate::{
    RevisedRange, RevisionCause, SessionEvent, TrajectoryRevision,
    graph::{Factor, Problem},
    pose::Pose,
    store::KeyframeId,
};
use nalgebra::{Matrix3, Vector3};
use std::collections::{BTreeMap, BTreeSet};

const MAX_ITERATIONS: usize = 30;
const MAX_RETRACTIONS: usize = 8;
const GAUGE_SIGMA_M: f64 = 1_000.0;
const GAUGE_SIGMA_RAD: f64 = 1.0;
const CHANGE_M: f64 = 1e-3;
const CHANGE_RAD: f64 = 1e-5;

/// What a factor stands for, so that an outlier can be retracted.
#[derive(Clone, Copy, Debug)]
enum Source {
    Fixed,
    Closure(usize),
    Anchor(KeyframeId),
    Frozen(KeyframeId),
}

struct Built {
    problem: Problem,
    ids: Vec<KeyframeId>,
    index: BTreeMap<KeyframeId, usize>,
    sources: Vec<Source>,
}

impl Built {
    fn push(&mut self, factor: Factor, source: Source) {
        self.problem.factors.push(factor);
        self.sources.push(source);
    }

    fn node(&self, id: &KeyframeId) -> Option<usize> {
        self.index.get(id).copied()
    }
}

impl VisualSession {
    fn build(&self) -> Built {
        let ids: Vec<KeyframeId> = self.keyframes.keys().copied().collect();
        let mut built = Built {
            problem: Problem {
                poses: self.keyframes.values().map(|k| k.estimate).collect(),
                biases: self.biases.values().copied().collect(),
                ..Problem::default()
            },
            index: ids.iter().enumerate().map(|(i, id)| (*id, i)).collect(),
            ids,
            sources: Vec::new(),
        };
        self.add_motion(&mut built);
        self.add_closures(&mut built);
        self.add_anchors(&mut built);
        self.add_priors(&mut built);
        built.problem.disabled = vec![false; built.problem.factors.len()];
        built
    }

    fn add_motion(&self, built: &mut Built) {
        for segment in self.segments.values() {
            for pair in segment.keyframes.windows(2) {
                let (Some(a), Some(b)) = (pair.first(), pair.get(1)) else {
                    continue;
                };
                let (Some(ka), Some(kb)) = (self.keyframes.get(a), self.keyframes.get(b)) else {
                    continue;
                };
                let (Some(ia), Some(ib)) = (built.node(a), built.node(b)) else {
                    continue;
                };
                let (sigma_m, sigma_rad) = ka.odometry.drift_to(&kb.odometry, &self.config.drift);
                let a_to_b = ka.odometry.pose.inv_mul(&kb.odometry.pose);
                let factor = Factor::Relative {
                    a: ia,
                    b: ib,
                    a_to_b,
                    sigma_m,
                    sigma_rad,
                    robust: false,
                };
                built.push(factor, Source::Fixed);
            }
        }
    }

    fn add_closures(&self, built: &mut Built) {
        for (i, closure) in self.closures.iter().enumerate() {
            let (Some(a), Some(b)) = (built.node(&closure.earlier), built.node(&closure.later))
            else {
                continue;
            };
            let factor = Factor::Relative {
                a,
                b,
                a_to_b: closure.measured,
                sigma_m: closure.sigma_m,
                sigma_rad: closure.sigma_rad,
                robust: true,
            };
            built.push(factor, Source::Closure(i));
        }
    }

    /// Anchors and one zero-mean prior for each map-cell bias they share.
    fn add_anchors(&self, built: &mut Built) {
        let cells: Vec<_> = self.biases.keys().cloned().collect();
        let mut shared: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
        for (id, anchor) in &self.anchors {
            let Some(node) = built.node(id) else { continue };
            let bias = cells.iter().position(|c| *c == anchor.cell);
            let mut covariance =
                anchor.budget.independent_m2 + Matrix3::identity() * anchor.transfer_m.powi(2);
            match bias {
                Some(j) => {
                    let entry = shared.entry(j).or_insert((0.0, 0.0));
                    entry.0 = entry.0.max(anchor.budget.shared_horizontal_m);
                    entry.1 = entry.1.max(anchor.budget.shared_vertical_m);
                }
                // Without a bias variable, the anchor carries the shared error itself.
                None => {
                    let h = anchor.budget.shared_horizontal_m.powi(2);
                    let v = anchor.budget.shared_vertical_m.powi(2);
                    covariance += Matrix3::from_diagonal(&Vector3::new(h, h, v));
                }
            }
            let Some(info) = sqrt_info(&covariance) else {
                continue;
            };
            let factor = Factor::Anchor {
                node,
                bias,
                pose: anchor.keyframe_pose,
                position_sqrt_info: info,
                sigma_rad: anchor.budget.rotation_rad + anchor.transfer_rad,
            };
            built.push(factor, Source::Anchor(*id));
        }
        for (j, (h, v)) in shared {
            let sqrt_info = Matrix3::from_diagonal(&Vector3::new(1.0 / h, 1.0 / h, 1.0 / v));
            built.push(Factor::BiasPrior { bias: j, sqrt_info }, Source::Fixed);
        }
    }

    /// Kept priors of evicted keyframes, and a weak gauge for unanchored components.
    fn add_priors(&self, built: &mut Built) {
        for (id, frozen) in &self.frozen {
            let Some(node) = built.node(id) else { continue };
            let factor = Factor::Prior {
                node,
                pose: frozen.pose,
                sigma_m: frozen.sigma_m,
                sigma_rad: frozen.sigma_rad,
            };
            built.push(factor, Source::Frozen(*id));
        }
        let counts = self.anchored_roots();
        let roots: BTreeSet<KeyframeId> = self.components().values().copied().collect();
        for root in roots.into_iter().filter(|r| !counts.contains_key(r)) {
            let (Some(node), Some(kf)) = (built.node(&root), self.keyframes.get(&root)) else {
                continue;
            };
            let factor = Factor::Prior {
                node,
                pose: kf.estimate,
                sigma_m: GAUGE_SIGMA_M,
                sigma_rad: GAUGE_SIGMA_RAD,
            };
            built.push(factor, Source::Fixed);
        }
    }

    /// Keyframe estimates before a change, for the revision report.
    pub(super) fn snapshot(&self) -> BTreeMap<KeyframeId, Pose> {
        self.keyframes
            .iter()
            .map(|(id, k)| (*id, k.estimate))
            .collect()
    }

    /// Optimize the graph, retract outliers, and report map poses that differ
    /// from `before`. Keyframes created after the snapshot start a range but
    /// do not count in its largest change.
    pub(super) fn optimize(
        &mut self,
        cause: RevisionCause,
        before: &BTreeMap<KeyframeId, Pose>,
    ) -> Option<TrajectoryRevision> {
        self.prune_biases();
        let mut retracted = false;
        for _ in 0..=MAX_RETRACTIONS {
            let mut built = self.build();
            built.problem.solve(MAX_ITERATIONS);
            for (id, pose) in built.ids.iter().zip(&built.problem.poses) {
                if let Some(kf) = self.keyframes.get_mut(id) {
                    kf.estimate = *pose;
                }
            }
            for (value, solved) in self.biases.values_mut().zip(&built.problem.biases) {
                *value = *solved;
            }
            if !self.retract_worst(&built) {
                break;
            }
            retracted = true;
        }
        self.refresh();
        let cause = if retracted {
            RevisionCause::Retraction
        } else {
            cause
        };
        self.revise(before, cause)
    }

    fn retract_worst(&mut self, built: &Built) -> bool {
        let limit = self.config.revisits.retract_sigma;
        let worst = built
            .sources
            .iter()
            .enumerate()
            .filter(|(_, s)| !matches!(s, Source::Fixed))
            .filter_map(|(i, s)| Some((*s, built.problem.squared_residual(i)?.sqrt())))
            .filter(|(_, r)| *r > limit)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match worst {
            Some((Source::Closure(i), residual)) if i < self.closures.len() => {
                let closure = self.closures.remove(i);
                self.event(SessionEvent::ClosureRetracted {
                    earlier: closure.constraint.earlier,
                    later: closure.constraint.later,
                    residual_sigma: residual,
                });
                true
            }
            Some((Source::Anchor(id), _)) => {
                if let Some(anchor) = self.anchors.remove(&id) {
                    self.event(SessionEvent::AnchorRetracted {
                        frame: anchor.observation.frame,
                    });
                }
                self.prune_biases();
                true
            }
            // A kept prior repeats evidence that the trajectory now contradicts.
            Some((Source::Frozen(id), _)) => self.frozen.remove(&id).is_some(),
            _ => false,
        }
    }

    fn revise(
        &mut self,
        before: &BTreeMap<KeyframeId, Pose>,
        cause: RevisionCause,
    ) -> Option<TrajectoryRevision> {
        let mut ranges = Vec::new();
        for (segment_id, segment) in &self.segments {
            let mut first_changed = None;
            let (mut shift, mut rotation) = (0.0_f64, 0.0_f64);
            for (position, id) in segment.keyframes.iter().enumerate() {
                let Some(kf) = self.keyframes.get(id) else {
                    continue;
                };
                // A new keyframe has no earlier map pose. It starts the range
                // but does not set the size of the change.
                let Some(old) = before.get(id) else {
                    first_changed.get_or_insert(position);
                    continue;
                };
                let d = (kf.estimate.translation.vector - old.translation.vector).norm();
                let r = old.rotation.angle_to(&kf.estimate.rotation);
                if d > CHANGE_M || r > CHANGE_RAD {
                    first_changed.get_or_insert(position);
                    shift = shift.max(d);
                    rotation = rotation.max(r);
                }
            }
            let Some(position) = first_changed else {
                continue;
            };
            let from = position
                .checked_sub(1)
                .and_then(|p| segment.keyframes.get(p))
                .and_then(|id| self.keyframes.get(id))
                .map_or(segment.first, |kf| kf.frame);
            let Some(map_from_odom) = segment.keyframes.last().and_then(|id| self.correction(*id))
            else {
                continue;
            };
            ranges.push(RevisedRange {
                segment: *segment_id,
                from,
                to: segment.head,
                max_shift_m: shift,
                max_rotation_rad: rotation,
                map_from_odom,
            });
        }
        if ranges.is_empty() {
            return None;
        }
        self.revision = self.revision.wrapping_add(1);
        let revision = TrajectoryRevision {
            revision: self.revision,
            cause,
            ranges,
        };
        self.event(SessionEvent::TrajectoryRevised(revision.clone()));
        Some(revision)
    }
}

/// Inverse of the lower Cholesky factor: whitens a residual with this covariance.
fn sqrt_info(covariance: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let symmetric = (covariance + covariance.transpose()) * 0.5;
    symmetric.cholesky()?.l().try_inverse()
}
