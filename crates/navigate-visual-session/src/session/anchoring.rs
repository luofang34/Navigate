//! Map-anchor admission: gate, agreement between frames, relocation, and transfer.

use super::{Anchor, VisualSession};
use crate::{
    AnchorObservation, AnchorRejection, FrameKey, FusionEligibility, RegionChange, RevisionCause,
    SegmentId, SegmentStart, SessionError, SessionEvent,
    anchor::budget,
    evidence::MapCell,
    pose::{Pose, finite},
    store::KeyframeId,
};
use nalgebra::Vector3;

/// An anchor that waits for agreement from other frames.
#[derive(Clone, Debug)]
pub(super) struct Pending {
    anchor: AnchorObservation,
}

/// The result of an anchor submission.
#[derive(Clone, Debug, PartialEq)]
pub enum AnchorDecision {
    /// The anchor entered the trajectory.
    Accepted {
        /// Fusion admission label.
        eligibility: FusionEligibility,
    },
    /// Agreeing anchors moved the segment to a new map position.
    Relocated {
        /// Anchors in the agreeing set.
        agreeing: usize,
        /// Earlier anchors that left the trajectory.
        retracted: usize,
    },
    /// An anchor from a nearby frame already constrains the same keyframe.
    SameViewpoint,
    /// The anchor did not enter the trajectory.
    Rejected(AnchorRejection),
}

impl VisualSession {
    /// Submit a camera pose matched against map imagery.
    ///
    /// The first anchor of an unlocated track is accepted without a check and
    /// is reported as unconfirmed. Later anchors must agree with the
    /// trajectory. Anchors that disagree wait until enough frames agree with
    /// each other; that agreement moves the track and retracts the anchors
    /// it contradicts. Anchors are never pulled toward the trajectory.
    ///
    /// # Errors
    /// Rejects unknown frames, digest mismatches, repeated evidence, invalid
    /// error declarations, and frames without odometry inside a tracked run.
    pub fn submit_anchor(
        &mut self,
        mut anchor: AnchorObservation,
    ) -> Result<AnchorDecision, SessionError> {
        // Every check that can refuse the input runs before any state change.
        self.check_anchor(&anchor)?;
        let start = self.segment_start(anchor.frame)?;
        anchor.pose = self.pose_in_session_frame(&anchor)?;
        if anchor.reliability.change == RegionChange::Changed {
            return Ok(self.reject(anchor.frame, AnchorRejection::ChangedRegion));
        }
        self.local_frame.get_or_insert(anchor.local_frame);
        let segment = match start {
            Some(segment) => segment,
            None => self.start_segment(anchor.frame, SegmentStart::AnchorAfterGap),
        };
        let Some((predicted, bound)) = self.predict(&anchor.frame) else {
            return self.accept(anchor, true);
        };
        let horizontal = budget(&anchor, self.half_fov()).horizontal_m();
        let distance = horizontal_distance(&predicted, &anchor.pose);
        let policy = self.config.anchors;
        let gate = policy.gate_sigma * (bound.powi(2) + horizontal.powi(2)).sqrt();
        if distance <= gate && bound <= policy.direct_bound_m {
            return self.accept(anchor, false);
        }
        let frame = anchor.frame;
        self.queue_pending(segment, anchor);
        if let Some(decision) = self.try_consensus(segment)? {
            return Ok(decision);
        }
        let reason = if distance > gate {
            AnchorRejection::Inconsistent {
                distance_m: distance,
                gate_m: gate,
            }
        } else {
            AnchorRejection::AwaitingConsensus {
                agreeing: self.best_group(segment).len().max(1),
            }
        };
        Ok(self.reject(frame, reason))
    }

    fn check_anchor(&self, anchor: &AnchorObservation) -> Result<(), SessionError> {
        let state = self
            .frames
            .get(&anchor.frame)
            .ok_or(SessionError::UnknownFrame(anchor.frame))?;
        if state.record.observation_sha256 != anchor.observation_sha256 {
            return Err(SessionError::EvidenceMismatch(anchor.frame));
        }
        let used = self
            .anchors
            .values()
            .any(|a| a.observation.observation_sha256 == anchor.observation_sha256)
            || self
                .pending
                .values()
                .flatten()
                .any(|p| p.anchor.observation_sha256 == anchor.observation_sha256);
        if used {
            return Err(SessionError::RepeatedEvidence(
                anchor.observation_sha256.clone(),
            ));
        }
        let geometry_ok = anchor.geometry_position_m2.iter().all(|v| v.is_finite())
            && anchor.geometry_rotation_rad.is_finite();
        if !anchor.reliability.validate() || !finite(&anchor.pose) || !geometry_ok {
            return Err(SessionError::Invalid { field: "anchor" });
        }
        // A covariance without a positive-definite square root cannot weight the anchor.
        let independent = budget(anchor, self.half_fov()).independent_m2;
        if ((independent + independent.transpose()) * 0.5)
            .cholesky()
            .is_none()
        {
            return Err(SessionError::Invalid {
                field: "anchor covariance",
            });
        }
        Ok(())
    }

    fn pose_in_session_frame(&self, anchor: &AnchorObservation) -> Result<Pose, SessionError> {
        let frame = self.local_frame.unwrap_or(anchor.local_frame);
        if frame == anchor.local_frame {
            return Ok(anchor.pose);
        }
        let geodetic = anchor.local_frame.geodetic(anchor.pose.translation.vector);
        let position = frame
            .local(geodetic)
            .map_err(|source| SessionError::Visual {
                frame: anchor.frame,
                source,
            })?;
        Ok(Pose::from_parts(position.into(), anchor.pose.rotation))
    }

    /// The segment of an anchored frame, or `None` for a frame after the
    /// tracked run, which starts a new segment.
    fn segment_start(&self, frame: FrameKey) -> Result<Option<SegmentId>, SessionError> {
        if let Some(odometry) = self.frames.get(&frame).and_then(|s| s.odometry.as_ref()) {
            return Ok(Some(odometry.segment));
        }
        let head = self
            .tracks
            .get(&frame.track())
            .and_then(|s| self.segments.get(s))
            .map(|s| s.head);
        if head.is_some_and(|h| h.index >= frame.index) {
            return Err(SessionError::NoOdometry(frame));
        }
        Ok(None)
    }

    /// The anchor error budget. An unknown lens model adds attitude error,
    /// because uncorrected distortion moves the fitted camera axis.
    fn anchor_budget(&self, anchor: &AnchorObservation) -> crate::AnchorBudget {
        let mut result = budget(anchor, self.half_fov());
        if self.calibration.lens == crate::LensModel::Unknown {
            result.rotation_rad += UNKNOWN_LENS_ROTATION_RAD;
        }
        result
    }

    pub(super) fn half_fov(&self) -> f64 {
        let c = &self.calibration.intrinsics;
        let half = (f64::from(c.width).powi(2) + f64::from(c.height).powi(2)).sqrt() * 0.5;
        (half / c.fx.min(c.fy)).atan()
    }

    fn reject(&mut self, frame: FrameKey, reason: AnchorRejection) -> AnchorDecision {
        self.event(SessionEvent::AnchorRejected { frame, reason });
        AnchorDecision::Rejected(reason)
    }

    /// Transfer the anchor to a keyframe and optimize.
    fn accept(
        &mut self,
        anchor: AnchorObservation,
        first: bool,
    ) -> Result<AnchorDecision, SessionError> {
        let before = self.snapshot();
        let shift = if first { Shift::Component } else { Shift::None };
        let frame = anchor.frame;
        let sha = anchor.observation_sha256.clone();
        let copy = anchor.clone();
        let Some((keyframe, eligibility)) = self.insert_anchor(anchor, shift)? else {
            return Ok(AnchorDecision::SameViewpoint);
        };
        self.optimize(RevisionCause::Anchor, &before);
        let kept = self
            .anchors
            .get(&keyframe)
            .is_some_and(|a| a.observation.observation_sha256 == sha);
        if kept {
            return Ok(AnchorDecision::Accepted { eligibility });
        }
        // Other evidence outvoted it. It still counts as evidence for a
        // relocation if later frames agree with it.
        let segment = self
            .frames
            .get(&frame)
            .and_then(|s| s.odometry.as_ref())
            .map(|o| o.segment);
        if let Some(segment) = segment {
            self.queue_pending(segment, copy);
            if let Some(decision) = self.try_consensus(segment)? {
                return Ok(decision);
            }
        }
        Ok(self.reject(frame, AnchorRejection::Retracted))
    }

    /// Insert without optimization. `None` reports a better anchor at the same keyframe.
    fn insert_anchor(
        &mut self,
        anchor: AnchorObservation,
        shift: Shift,
    ) -> Result<Option<(KeyframeId, FusionEligibility)>, SessionError> {
        let attachment = self.attach(anchor.frame, &[])?;
        let budget = self.anchor_budget(&anchor);
        if let Some(existing) = self.anchors.get(&attachment.keyframe)
            && existing.budget.horizontal_m() + existing.transfer_m
                <= budget.horizontal_m() + attachment.drift_m
        {
            return Ok(None);
        }
        let keyframe_pose = anchor.pose * attachment.frame_to_keyframe;
        if shift != Shift::None {
            self.shift_to(attachment.keyframe, &keyframe_pose, shift);
        }
        let cell = self.cell(&anchor);
        if !self.biases.contains_key(&cell) && self.biases.len() < self.config.limits.max_bias_cells
        {
            self.biases.insert(cell.clone(), Vector3::zeros());
        }
        let eligibility = self.ledger.label(
            cell.clone(),
            anchor.frame,
            self.config.limits.max_ledger_cells,
        );
        let frame = anchor.frame;
        let capture_ns = self
            .frames
            .get(&frame)
            .map_or(0, |s| s.record.capture.at_ns);
        let record = Anchor {
            observation: anchor,
            keyframe_pose,
            budget,
            transfer_m: attachment.drift_m,
            transfer_rad: attachment.drift_rad,
            cell,
            capture_ns,
        };
        if let Some(replaced) = self.anchors.insert(attachment.keyframe, record) {
            self.event(SessionEvent::AnchorRetracted {
                frame: replaced.observation.frame,
            });
        }
        self.event(SessionEvent::AnchorAccepted {
            frame,
            eligibility: eligibility.clone(),
        });
        Ok(Some((attachment.keyframe, eligibility)))
    }

    /// Move the keyframes of a component or a segment so that one keyframe
    /// takes `target`. This gives the solver a start near the new solution.
    pub(super) fn move_component(&mut self, keyframe: KeyframeId, target: &Pose) {
        self.shift_to(keyframe, target, Shift::Component);
    }

    fn shift_to(&mut self, keyframe: KeyframeId, target: &Pose, shift: Shift) {
        let Some(current) = self.keyframes.get(&keyframe) else {
            return;
        };
        let delta = target * current.estimate.inverse();
        let segment = current.odometry.segment;
        let components = self.components();
        let root = components.get(&keyframe).copied();
        for (id, kf) in self.keyframes.iter_mut() {
            let moves = match shift {
                Shift::Component => components.get(id).copied() == root,
                Shift::Segment => kf.odometry.segment == segment,
                Shift::None => false,
            };
            if moves {
                kf.estimate = crate::pose::compose(&delta, &kf.estimate);
            }
        }
    }

    fn cell(&self, anchor: &AnchorObservation) -> MapCell {
        let size = self.config.anchors.shared_error_cell_m;
        let p = anchor.pose.translation.vector;
        MapCell {
            map_manifest_sha256: anchor.map.manifest_sha256.clone(),
            east: (p.x / size).floor() as i64,
            north: (p.y / size).floor() as i64,
        }
    }

    fn queue_pending(&mut self, segment: SegmentId, anchor: AnchorObservation) {
        let limit = self.config.anchors.max_pending;
        let list = self.pending.entry(segment).or_default();
        list.push(Pending { anchor });
        if list.len() > limit {
            let old = list.remove(0);
            self.event(SessionEvent::AnchorRejected {
                frame: old.anchor.frame,
                reason: AnchorRejection::PendingOverflow,
            });
        }
    }

    /// Two anchors agree when they are separate evidence and consistent.
    fn agree(&self, a: &AnchorObservation, b: &AnchorObservation) -> bool {
        let path = |x: &AnchorObservation| {
            self.frames
                .get(&x.frame)
                .and_then(|s| s.odometry.as_ref())
                .map(|o| o.path_m)
        };
        let separated = path(a)
            .zip(path(b))
            .is_some_and(|(pa, pb)| (pa - pb).abs() >= self.config.anchors.consensus_separation_m);
        separated && self.consistent(a, b)
    }

    /// Two anchors are consistent when odometry carries one onto the other within the gate.
    fn consistent(&self, a: &AnchorObservation, b: &AnchorObservation) -> bool {
        let (Some(oa), Some(ob)) = (
            self.frames.get(&a.frame).and_then(|s| s.odometry.clone()),
            self.frames.get(&b.frame).and_then(|s| s.odometry.clone()),
        ) else {
            return false;
        };
        if oa.segment != ob.segment {
            return false;
        }
        let predicted = a.pose * oa.pose.inv_mul(&ob.pose);
        let (drift, _) = oa.drift_to(&ob, &self.config.drift);
        let fov = self.half_fov();
        let (ha, hb) = (budget(a, fov).horizontal_m(), budget(b, fov).horizontal_m());
        let gate =
            self.config.anchors.gate_sigma * (ha.powi(2) + hb.powi(2) + drift.powi(2)).sqrt();
        horizontal_distance(&predicted, &b.pose) <= gate
    }

    /// The largest set of pending anchors that agree with each other.
    fn best_group(&self, segment: SegmentId) -> Vec<usize> {
        let Some(list) = self.pending.get(&segment) else {
            return Vec::new();
        };
        let mut best: Vec<usize> = Vec::new();
        for (i, leader) in list.iter().enumerate() {
            let mut group = vec![i];
            for (j, other) in list.iter().enumerate().filter(|(j, _)| *j != i) {
                let joins = group.iter().all(|&g| {
                    list.get(g)
                        .is_some_and(|member| self.agree(&member.anchor, &other.anchor))
                });
                if joins && self.agree(&leader.anchor, &other.anchor) {
                    group.push(j);
                }
            }
            if group.len() > best.len() {
                best = group;
            }
        }
        best
    }

    fn try_consensus(
        &mut self,
        segment: SegmentId,
    ) -> Result<Option<AnchorDecision>, SessionError> {
        let group = self.best_group(segment);
        if group.len() < self.config.anchors.consensus_anchors {
            return Ok(None);
        }
        let list = self.pending.get(&segment).cloned().unwrap_or_default();
        let Some(leader) = group
            .first()
            .and_then(|i| list.get(*i))
            .map(|p| p.anchor.clone())
        else {
            return Ok(None);
        };
        let conflicting: Vec<KeyframeId> = self
            .anchors
            .iter()
            .filter(|(id, a)| {
                self.keyframes
                    .get(id)
                    .is_some_and(|k| k.odometry.segment == segment)
                    && !self.consistent(&leader, &a.observation)
            })
            .map(|(id, _)| *id)
            .collect();
        if conflicting.len() >= group.len() {
            return Ok(None);
        }
        let before = self.snapshot();
        // Kept priors of this segment repeat the contradicted evidence.
        let keyframes = &self.keyframes;
        self.frozen.retain(|id, _| {
            keyframes
                .get(id)
                .is_none_or(|k| k.odometry.segment != segment)
        });
        for id in &conflicting {
            if let Some(a) = self.anchors.remove(id) {
                self.event(SessionEvent::AnchorRetracted {
                    frame: a.observation.frame,
                });
            }
        }
        let mut members: Vec<AnchorObservation> = group
            .iter()
            .filter_map(|i| list.get(*i))
            .map(|p| p.anchor.clone())
            .collect();
        if let Some(remaining) = self.pending.get_mut(&segment) {
            let frames: Vec<FrameKey> = members.iter().map(|a| a.frame).collect();
            remaining.retain(|p| !frames.contains(&p.anchor.frame));
        }
        let agreeing = members.len();
        let mut shift = Shift::Segment;
        let mut inserted = Vec::new();
        for anchor in members.drain(..) {
            let sha = anchor.observation_sha256.clone();
            if let Some((keyframe, _)) = self.insert_anchor(anchor, shift)? {
                shift = Shift::None;
                inserted.push((keyframe, sha));
            }
        }
        self.optimize(RevisionCause::Relocation, &before);
        let survived = inserted.iter().any(|(keyframe, sha)| {
            self.anchors
                .get(keyframe)
                .is_some_and(|a| a.observation.observation_sha256 == *sha)
        });
        if inserted.is_empty() {
            // The agreeing anchors share keyframes with better anchors.
            return Ok(Some(AnchorDecision::SameViewpoint));
        }
        if !survived {
            return Ok(Some(AnchorDecision::Rejected(AnchorRejection::Retracted)));
        }
        Ok(Some(AnchorDecision::Relocated {
            agreeing,
            retracted: conflicting.len(),
        }))
    }
}

/// Attitude error added for an uncalibrated lens, in radians.
const UNKNOWN_LENS_ROTATION_RAD: f64 = 0.03;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shift {
    None,
    Component,
    Segment,
}

fn horizontal_distance(a: &Pose, b: &Pose) -> f64 {
    (a.translation.vector.xy() - b.translation.vector.xy()).norm()
}
