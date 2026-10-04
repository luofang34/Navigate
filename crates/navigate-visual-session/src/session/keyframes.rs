//! Keyframe creation, frame attachment, and the keyframe memory bound.

use super::{Frozen, VisualSession};
use crate::{
    FrameKey, SegmentId, SessionError, SessionEvent,
    pose::Pose,
    store::{Carry, Keyframe, KeyframeId, Odometry, wants_keyframe},
};

/// Rotation error of a kept prior, in radians.
const FROZEN_SIGMA_RAD: f64 = 0.05;

/// A frame joined to a keyframe of its segment.
pub(super) struct Attachment {
    pub keyframe: KeyframeId,
    /// Pose of the keyframe camera in the frame camera.
    pub frame_to_keyframe: Pose,
    /// Declared drift between the frame and the keyframe, in metres.
    pub drift_m: f64,
    pub drift_rad: f64,
}

impl VisualSession {
    pub(super) fn push_keyframe(
        &mut self,
        segment: SegmentId,
        frame: FrameKey,
        odometry: &Odometry,
        estimate: Pose,
        protect: &[KeyframeId],
    ) -> KeyframeId {
        let id = KeyframeId(self.next_keyframe);
        self.next_keyframe = self.next_keyframe.wrapping_add(1);
        let (capture_ns, observation_sha256) = self
            .frames
            .get(&frame)
            .map(|s| (s.record.capture.at_ns, s.record.observation_sha256.clone()))
            .unwrap_or_default();
        self.keyframes.insert(
            id,
            Keyframe {
                frame,
                capture_ns,
                observation_sha256,
                odometry: odometry.clone(),
                estimate,
                descriptor: None,
            },
        );
        let keyframes = &self.keyframes;
        if let Some(s) = self.segments.get_mut(&segment) {
            let at = s
                .keyframes
                .iter()
                .position(|k| {
                    keyframes
                        .get(k)
                        .is_some_and(|kf| kf.frame.index > frame.index)
                })
                .unwrap_or(s.keyframes.len());
            s.keyframes.insert(at, id);
        }
        let mut kept = protect.to_vec();
        kept.push(id);
        self.evict_keyframes(&kept);
        self.refresh();
        id
    }

    /// Map-from-odometry correction implied by one keyframe.
    pub(super) fn correction(&self, id: KeyframeId) -> Option<Pose> {
        let kf = self.keyframes.get(&id)?;
        Some(kf.estimate * kf.odometry.pose.inverse())
    }

    pub(super) fn extend_keyframes(
        &mut self,
        segment: SegmentId,
        frame: FrameKey,
        odometry: &Odometry,
    ) -> bool {
        let Some(last) = self
            .segments
            .get(&segment)
            .and_then(|s| s.keyframes.last().copied())
        else {
            self.restart_keyframes(segment, frame, odometry);
            return true;
        };
        let wanted = self
            .keyframes
            .get(&last)
            .is_some_and(|kf| wants_keyframe(kf, odometry, &self.config.keyframes));
        if wanted {
            let estimate = self.correction(last).unwrap_or_else(Pose::identity) * odometry.pose;
            self.push_keyframe(segment, frame, odometry, estimate, &[]);
        }
        wanted
    }

    /// First keyframe of a segment, or the next one after the memory bound
    /// removed all of them. A carried correction keeps the map reference.
    fn restart_keyframes(&mut self, segment: SegmentId, frame: FrameKey, odometry: &Odometry) {
        let carry = self.segments.get_mut(&segment).and_then(|s| s.carry.take());
        let Some(carry) = carry else {
            self.push_keyframe(segment, frame, odometry, odometry.pose, &[]);
            return;
        };
        let estimate = carry.correction * odometry.pose;
        let id = self.push_keyframe(segment, frame, odometry, estimate, &[]);
        let (drift_m, drift_rad) = carry.odometry.drift_to(odometry, &self.config.drift);
        let frozen = Frozen {
            pose: estimate,
            sigma_m: carry.sigma_m + drift_m,
            sigma_rad: FROZEN_SIGMA_RAD + drift_rad,
        };
        self.frozen.insert(id, frozen);
        self.refresh();
    }

    /// The keyframes of a segment before and after a frame index.
    pub(super) fn neighbours(
        &self,
        segment: SegmentId,
        index: u64,
    ) -> (Option<KeyframeId>, Option<KeyframeId>) {
        let Some(s) = self.segments.get(&segment) else {
            return (None, None);
        };
        let split = s
            .keyframes
            .iter()
            .position(|k| {
                self.keyframes
                    .get(k)
                    .is_some_and(|kf| kf.frame.index > index)
            })
            .unwrap_or(s.keyframes.len());
        let before = split
            .checked_sub(1)
            .and_then(|i| s.keyframes.get(i))
            .copied();
        (before, s.keyframes.get(split).copied())
    }

    /// Join a frame to a nearby keyframe, or make it a keyframe.
    ///
    /// `protect` names keyframes that the memory bound must keep during the call.
    pub(super) fn attach(
        &mut self,
        frame: FrameKey,
        protect: &[KeyframeId],
    ) -> Result<Attachment, SessionError> {
        let odometry = self
            .frames
            .get(&frame)
            .and_then(|s| s.odometry.clone())
            .ok_or(SessionError::UnknownFrame(frame))?;
        let (before, after) = self.neighbours(odometry.segment, frame.index);
        let near = [before, after].into_iter().flatten().find(|id| {
            self.keyframes.get(id).is_some_and(|kf| {
                kf.frame == frame || !wants_keyframe(kf, &odometry, &self.config.keyframes)
            })
        });
        let id = match near {
            Some(id) => id,
            None => {
                let base = before
                    .or(after)
                    .and_then(|k| self.correction(k))
                    .unwrap_or_else(Pose::identity);
                let estimate = base * odometry.pose;
                self.push_keyframe(odometry.segment, frame, &odometry, estimate, protect)
            }
        };
        let kf = self
            .keyframes
            .get(&id)
            .ok_or(SessionError::UnknownFrame(frame))?;
        let (drift_m, drift_rad) = if kf.frame == frame {
            (0.0, 0.0)
        } else {
            odometry.drift_to(&kf.odometry, &self.config.drift)
        };
        Ok(Attachment {
            keyframe: id,
            frame_to_keyframe: odometry.pose.inv_mul(&kf.odometry.pose),
            drift_m,
            drift_rad,
        })
    }

    /// Remove segments without keyframes that can no longer grow: a newer
    /// segment replaced them, or the frame ring keeps no frame of their epoch.
    pub(super) fn prune_segments(&mut self) {
        let finished: Vec<_> = self
            .segments
            .iter()
            .filter(|(id, s)| {
                let track = s.first.track();
                s.keyframes.is_empty()
                    && (self.tracks.get(&track) != Some(id) || !self.frames.keeps(&track))
            })
            .map(|(id, s)| (*id, s.first.track()))
            .collect();
        for (id, track) in finished {
            self.segments.remove(&id);
            self.pending.remove(&id);
            if self.tracks.get(&track) == Some(&id) {
                self.tracks.remove(&track);
            }
        }
    }

    fn evict_keyframes(&mut self, protect: &[KeyframeId]) {
        while self.keyframes.len() > self.config.limits.max_keyframes {
            let Some(oldest) = self
                .keyframes
                .iter()
                .filter(|(id, _)| !protect.contains(id))
                .min_by_key(|(id, kf)| (kf.capture_ns, **id))
                .map(|(id, _)| *id)
            else {
                break;
            };
            // The frozen prior keeps the current estimate, so the map poses
            // do not change and no solve is needed.
            self.evict(oldest);
        }
    }

    fn evict(&mut self, id: KeyframeId) {
        let bounds = self.bounds();
        let Some(kf) = self.keyframes.remove(&id) else {
            return;
        };
        let segment = kf.odometry.segment;
        let next = self.segments.get_mut(&segment).and_then(|s| {
            let at = s.keyframes.iter().position(|k| *k == id)?;
            s.keyframes.remove(at);
            s.keyframes.get(at).copied()
        });
        // The next keyframe keeps the located estimate as a prior, so the
        // remaining trajectory keeps its map reference. The bound is a
        // conservative sum, which limits the weight of the repeated evidence.
        if let Some(bound) = bounds.get(&id).copied().flatten() {
            match next.and_then(|n| self.keyframes.get(&n).map(|k| (n, k))) {
                Some((next, n)) => {
                    let (drift_m, drift_rad) =
                        kf.odometry.drift_to(&n.odometry, &self.config.drift);
                    let frozen = Frozen {
                        pose: n.estimate,
                        sigma_m: bound + drift_m,
                        sigma_rad: FROZEN_SIGMA_RAD + drift_rad,
                    };
                    self.frozen.entry(next).or_insert(frozen);
                }
                None => {
                    let carry = Carry {
                        correction: kf.estimate * kf.odometry.pose.inverse(),
                        sigma_m: bound,
                        odometry: kf.odometry.clone(),
                    };
                    if let Some(s) = self.segments.get_mut(&segment) {
                        s.carry = Some(carry);
                    }
                }
            }
        }
        self.anchors.remove(&id);
        self.frozen.remove(&id);
        let before = self.closures.len();
        self.closures.retain(|c| c.earlier != id && c.later != id);
        if self.closures.len() != before {
            self.event(SessionEvent::ClosureEvicted { keyframe: kf.frame });
        }
        self.prune_biases();
        self.event(SessionEvent::KeyframeEvicted { frame: kf.frame });
        self.prune_segments();
    }

    /// Remove bias variables that no anchor uses.
    pub(super) fn prune_biases(&mut self) {
        let cells: std::collections::BTreeSet<_> =
            self.anchors.values().map(|a| a.cell.clone()).collect();
        self.biases.retain(|cell, _| cells.contains(cell));
    }
}
