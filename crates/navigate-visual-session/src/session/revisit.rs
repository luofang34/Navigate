//! Revisits: historical keyframe search, geometric verification, and closures.
//!
//! A closure measures the camera of a later frame relative to an earlier
//! keyframe that is not adjacent in time. Verification uses
//! `PoseVerifier::track_from_prior` with depth rendered at the earlier
//! keyframe, so the closure scale comes from the map depth near both cameras.
//! The fit starts at the predicted later pose, because a return pass often
//! has the opposite heading.

use super::{Closure, VisualSession};
use crate::{
    FrameKey, RevisionCause, ScaleSource, SessionError, SessionEvent,
    pose::{Pose, finite, from_camera, to_camera},
    store::{Keyframe, KeyframeId, Odometry},
};
use navigate_visual::{
    CameraPose, EstimateQuality, Frame, LocalizerConfig, PixelMatch, PosePrior, PoseVerifier,
    ReferenceView, TrackingMotion, TrackingReference,
};

/// An earlier keyframe that a later frame may see again.
#[derive(Clone, Debug)]
pub struct RevisitCandidate {
    /// Earlier keyframe.
    pub earlier: FrameKey,
    /// Evidence digest of the earlier keyframe.
    pub earlier_observation_sha256: String,
    /// Pose at which the host renders depth for the earlier keyframe.
    pub earlier_pose: CameraPose,
    /// Later frame.
    pub later: FrameKey,
    /// Search bounds for the later camera, in the frame of `earlier_pose`.
    pub later_prior: PosePrior,
    /// Cosine similarity of the retrieval descriptors, when both exist.
    pub similarity: Option<f32>,
    /// Predicted camera distance, when both poses share one frame.
    pub distance_m: Option<f64>,
}

/// Host evidence for one revisit check.
pub struct RevisitEvidence<'a> {
    /// Later frame pixels.
    pub later: &'a Frame,
    /// Earlier keyframe pixels.
    pub earlier: &'a Frame,
    /// Depth rendered at `RevisitCandidate::earlier_pose`.
    pub earlier_surface: &'a ReferenceView,
    /// Matches from the earlier image (reference) to the later image (query).
    pub matches: &'a [PixelMatch],
    /// Matcher identity.
    pub matcher_identity: &'a str,
}

/// A verified relative pose between an earlier keyframe and a later frame.
#[derive(Clone, Debug)]
pub struct RevisitConstraint {
    /// Earlier keyframe.
    pub earlier: FrameKey,
    /// Later frame.
    pub later: FrameKey,
    /// Pose of the later camera in the earlier camera frame.
    pub earlier_to_later: Pose,
    /// Declared position error, in metres.
    pub sigma_m: f64,
    /// Declared rotation error, in radians.
    pub sigma_rad: f64,
    /// Geometric support of the check.
    pub quality: EstimateQuality,
    /// Scale source of the check.
    pub scale: ScaleSource,
    /// Matcher identity.
    pub backend: String,
}

/// Why a closure did not enter the trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RevisitDecision {
    /// The closure entered the graph.
    Accepted,
    /// The two frames are adjacent in time or in the keyframe chain.
    Adjacent,
    /// A closure between the same keyframes exists.
    Duplicate,
    /// The closure entered the graph, but it remained an outlier after
    /// optimization and was retracted.
    Retracted,
    /// The closure disagrees with the trajectory by more than the gate allows.
    Inconsistent {
        /// Position disagreement, in metres.
        distance_m: f64,
        /// Gate radius, in metres.
        gate_m: f64,
    },
}

impl VisualSession {
    /// Attach a retrieval descriptor to a keyframe. Returns false for a frame
    /// that is not a keyframe.
    ///
    /// # Errors
    /// Rejects a descriptor that is empty, too long, non-finite, or zero.
    pub fn set_descriptor(
        &mut self,
        frame: &FrameKey,
        descriptor: &[f32],
    ) -> Result<bool, SessionError> {
        let norm = descriptor.iter().map(|v| v * v).sum::<f32>().sqrt();
        if descriptor.is_empty()
            || descriptor.len() > self.config.limits.max_descriptor_len
            || !descriptor.iter().all(|v| v.is_finite())
            || norm <= 0.0
        {
            return Err(SessionError::Invalid {
                field: "retrieval descriptor",
            });
        }
        let Some(kf) = self.keyframes.values_mut().find(|k| k.frame == *frame) else {
            return Ok(false);
        };
        kf.descriptor = Some(descriptor.iter().map(|v| v / norm).collect());
        Ok(true)
    }

    fn keyframe_of(&self, frame: &FrameKey) -> Option<KeyframeId> {
        self.keyframes
            .iter()
            .find(|(_, k)| k.frame == *frame)
            .map(|(id, _)| *id)
    }

    /// True when two frames are close in time or in their keyframe chain.
    fn adjacent(&self, earlier: KeyframeId, later: &FrameKey) -> bool {
        let policy = &self.config.revisits;
        let (Some(kf), Some(state)) = (self.keyframes.get(&earlier), self.frames.get(later)) else {
            return true;
        };
        if state.record.capture.at_ns.abs_diff(kf.capture_ns) < policy.min_time_separation_ns {
            return true;
        }
        let Some(odometry) = state.odometry.as_ref() else {
            return true;
        };
        if odometry.segment != kf.odometry.segment {
            return false;
        }
        let chain = self
            .segments
            .get(&odometry.segment)
            .map(|s| s.keyframes.as_slice())
            .unwrap_or_default();
        let position = |index: u64| {
            chain
                .iter()
                .filter(|k| {
                    self.keyframes
                        .get(k)
                        .is_some_and(|x| x.frame.index <= index)
                })
                .count()
        };
        position(later.index).abs_diff(position(kf.frame.index)) < policy.min_keyframe_separation
    }

    /// Earlier keyframes that a later frame may see again, best first.
    ///
    /// Candidates in the same connected trajectory, or with both trajectories
    /// on the map, need predicted overlap. Candidates in unconnected
    /// trajectories need descriptor similarity. Temporal neighbours never qualify.
    ///
    /// # Errors
    /// Rejects an unknown frame or a frame without odometry.
    pub fn revisit_candidates(
        &self,
        later: &FrameKey,
        descriptor: Option<&[f32]>,
    ) -> Result<Vec<RevisitCandidate>, SessionError> {
        let state = self
            .frames
            .get(later)
            .ok_or(SessionError::UnknownFrame(*later))?;
        let odometry = state
            .odometry
            .clone()
            .ok_or(SessionError::NoOdometry(*later))?;
        let keyframe = self.neighbours(odometry.segment, later.index).0;
        let query = Query {
            later: *later,
            descriptor: descriptor.and_then(normalized),
            estimate: keyframe
                .and_then(|k| self.correction(k))
                .map(|c| crate::pose::compose(&c, &odometry.pose)),
            root: keyframe
                .and_then(|k| self.topology.components.get(&k))
                .copied(),
            bound: self.predict(later).map(|(_, b)| b),
            odometry,
        };
        let mut found: Vec<RevisitCandidate> = self
            .keyframes
            .iter()
            .filter(|(id, _)| !self.adjacent(**id, later))
            .filter_map(|(id, kf)| self.candidate(&query, id, kf))
            .collect();
        found.sort_by(|a, b| {
            let s = b
                .similarity
                .unwrap_or(f32::MIN)
                .total_cmp(&a.similarity.unwrap_or(f32::MIN));
            s.then_with(|| {
                a.distance_m
                    .unwrap_or(f64::MAX)
                    .total_cmp(&b.distance_m.unwrap_or(f64::MAX))
            })
        });
        found.truncate(self.config.revisits.max_candidates);
        Ok(found)
    }

    /// One keyframe as a candidate, or `None` when it cannot overlap the query.
    fn candidate(&self, query: &Query, id: &KeyframeId, kf: &Keyframe) -> Option<RevisitCandidate> {
        let policy = self.config.revisits;
        let similarity = match (&query.descriptor, &kf.descriptor) {
            (Some(q), Some(d)) if q.len() == d.len() => {
                Some(q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>())
            }
            _ => None,
        };
        let root = self.topology.components.get(id).copied();
        let kf_bound = self.topology.bounds.get(id).copied().flatten();
        let comparable = root == query.root || (query.bound.is_some() && kf_bound.is_some());
        let spatial = query.estimate.filter(|_| comparable).map(|e| {
            let distance = (e.translation.vector - kf.estimate.translation.vector).norm();
            let drift = || query.odometry.drift_to(&kf.odometry, &self.config.drift).0;
            let bounds = query.bound.unwrap_or(0.0) + kf_bound.unwrap_or_else(drift);
            let radius = self.config.anchors.gate_sigma * bounds + policy.search_margin_m;
            (distance, axis_angle(&e, &kf.estimate), radius, e)
        });
        let (prior_pose, radius, distance) = match spatial {
            Some((d, view, radius, e)) if d <= radius && view <= policy.max_view_angle_rad => {
                (e, radius, Some(d))
            }
            Some(_) => return None,
            None if similarity.is_some() => (kf.estimate, policy.search_margin_m, None),
            None => return None,
        };
        Some(RevisitCandidate {
            earlier: kf.frame,
            earlier_observation_sha256: kf.observation_sha256.clone(),
            earlier_pose: to_camera(&kf.estimate),
            later: query.later,
            later_prior: PosePrior {
                pose: to_camera(&prior_pose),
                position_radius_m: radius,
                attitude_radius_rad: (2.0 * policy.max_view_angle_rad).min(std::f64::consts::PI),
            },
            similarity,
            distance_m: distance,
        })
    }

    /// Verify a candidate with matches against depth rendered at the earlier keyframe.
    ///
    /// # Errors
    /// Rejects evidence that does not belong to the candidate frames, and
    /// returns the visual geometry error when the check fails.
    pub fn verify_revisit(
        &self,
        candidate: &RevisitCandidate,
        evidence: RevisitEvidence<'_>,
    ) -> Result<RevisitConstraint, SessionError> {
        let later_sha = self
            .frames
            .get(&candidate.later)
            .map(|s| s.record.observation_sha256.clone());
        if later_sha.as_deref() != Some(evidence.later.evidence_sha256().as_str()) {
            return Err(SessionError::EvidenceMismatch(candidate.later));
        }
        if candidate.earlier_observation_sha256 != evidence.earlier.evidence_sha256() {
            return Err(SessionError::EvidenceMismatch(candidate.earlier));
        }
        let visual = |source| SessionError::Visual {
            frame: candidate.later,
            source,
        };
        let verifier = PoseVerifier::new(LocalizerConfig::default()).map_err(visual)?;
        let reference = TrackingReference {
            observation: evidence.earlier,
            surface: evidence.earlier_surface,
        };
        let proposal = verifier
            .track_from_prior(
                evidence.later,
                reference,
                &candidate.later_prior,
                evidence.matches,
                evidence.matcher_identity,
                TrackingMotion::Free,
            )
            .map_err(visual)?;
        let earlier_to_later =
            from_camera(&evidence.earlier_surface.pose).inv_mul(&from_camera(&proposal.pose));
        let policy = self.config.revisits;
        Ok(RevisitConstraint {
            earlier: candidate.earlier,
            later: candidate.later,
            sigma_m: policy.closure_floor_m
                + policy.closure_fraction * earlier_to_later.translation.vector.norm(),
            sigma_rad: policy.closure_rotation_rad,
            earlier_to_later,
            quality: proposal.quality,
            scale: ScaleSource::MapDepth { map: proposal.map },
            backend: proposal.backend,
        })
    }

    /// Add a verified closure, optimize, and retract it if it remains an outlier.
    ///
    /// # Errors
    /// Rejects a non-finite pose or error, and frames that are not in the
    /// session or have no odometry. A refused closure changes no state.
    pub fn submit_revisit(
        &mut self,
        constraint: RevisitConstraint,
    ) -> Result<RevisitDecision, SessionError> {
        let sigmas = [constraint.sigma_m, constraint.sigma_rad];
        if !finite(&constraint.earlier_to_later)
            || !sigmas.iter().all(|v| v.is_finite() && *v > 0.0)
        {
            return Err(SessionError::Invalid {
                field: "revisit constraint",
            });
        }
        let earlier = self
            .keyframe_of(&constraint.earlier)
            .ok_or(SessionError::UnknownFrame(constraint.earlier))?;
        if self
            .frames
            .get(&constraint.later)
            .and_then(|s| s.odometry.as_ref())
            .is_none()
        {
            return Err(SessionError::NoOdometry(constraint.later));
        }
        if self.adjacent(earlier, &constraint.later) {
            return Ok(RevisitDecision::Adjacent);
        }
        let attachment = self.attach(constraint.later, &[earlier])?;
        let later = attachment.keyframe;
        if earlier == later
            || self
                .closures
                .iter()
                .any(|c| (c.earlier, c.later) == (earlier, later))
        {
            return Ok(RevisitDecision::Duplicate);
        }
        let measured = constraint.earlier_to_later * attachment.frame_to_keyframe;
        let sigma_m = constraint.sigma_m + attachment.drift_m;
        let sigma_rad = constraint.sigma_rad + attachment.drift_rad;
        let frames = (constraint.earlier, constraint.later);
        if let Some(decision) = self.closure_gate((earlier, later), frames, &measured, sigma_m)? {
            return Ok(decision);
        }
        let before = self.snapshot();
        self.join_components(earlier, later, &measured);
        if self.closures.len() >= self.config.limits.max_closures && !self.closures.is_empty() {
            let oldest = self.closures.remove(0);
            self.event(SessionEvent::ClosureLimit {
                earlier: oldest.constraint.earlier,
                later: oldest.constraint.later,
            });
        }
        let event = SessionEvent::ClosureAccepted {
            earlier: constraint.earlier,
            later: constraint.later,
        };
        self.closures.push(Closure {
            constraint,
            earlier,
            later,
            measured,
            sigma_m,
            sigma_rad,
        });
        self.event(event);
        self.optimize(RevisionCause::Closure, &before);
        let kept = self
            .closures
            .iter()
            .any(|c| (c.earlier, c.later) == (earlier, later));
        Ok(if kept {
            RevisitDecision::Accepted
        } else {
            RevisitDecision::Retracted
        })
    }

    /// Refuse a closure that disagrees with the trajectory. Missing keyframes fail closed.
    fn closure_gate(
        &self,
        (earlier, later): (KeyframeId, KeyframeId),
        (earlier_frame, later_frame): (FrameKey, FrameKey),
        measured: &Pose,
        sigma_m: f64,
    ) -> Result<Option<RevisitDecision>, SessionError> {
        let a = self
            .keyframes
            .get(&earlier)
            .ok_or(SessionError::UnknownFrame(earlier_frame))?;
        let b = self
            .keyframes
            .get(&later)
            .ok_or(SessionError::UnknownFrame(later_frame))?;
        let same = self.topology.components.get(&earlier) == self.topology.components.get(&later);
        let bound = |id: &KeyframeId| self.topology.bounds.get(id).copied().flatten();
        let anchored = bound(&earlier).zip(bound(&later)).map(|(x, y)| x + y);
        let drift = || {
            self.path_drift(earlier, later)
                .ok_or(SessionError::UnknownFrame(later_frame))
        };
        let uncertainty = match (same, anchored) {
            (true, Some(sum)) => drift()?.min(sum),
            (true, None) => drift()?,
            // Unconnected trajectories: the closure joins them.
            (false, Some(sum)) => sum,
            (false, None) => return Ok(None),
        };
        let predicted = a.estimate.inv_mul(&b.estimate);
        let distance = (predicted.translation.vector - measured.translation.vector).norm();
        let gate = self.config.anchors.gate_sigma * (uncertainty + sigma_m);
        let refused = distance.is_nan() || distance > gate;
        Ok(refused.then_some(RevisitDecision::Inconsistent {
            distance_m: distance,
            gate_m: gate,
        }))
    }

    /// Smallest declared drift along links between two keyframes.
    fn path_drift(&self, from: KeyframeId, to: KeyframeId) -> Option<f64> {
        self.propagate(std::iter::once((from, 0.0)), |_, link| link.m)
            .get(&to)
            .copied()
            .flatten()
    }

    /// Move an unlocated component next to the other end of a new closure.
    fn join_components(&mut self, earlier: KeyframeId, later: KeyframeId, measured: &Pose) {
        if self.topology.components.get(&earlier) == self.topology.components.get(&later) {
            return;
        }
        let located = |id: &KeyframeId| self.topology.bounds.get(id).copied().flatten().is_some();
        let (Some(a), Some(b)) = (
            self.keyframes.get(&earlier).map(|k| k.estimate),
            self.keyframes.get(&later).map(|k| k.estimate),
        ) else {
            return;
        };
        if !located(&later) {
            self.move_component(later, &(a * measured));
        } else if !located(&earlier) {
            self.move_component(earlier, &(b * measured.inverse()));
        }
    }
}

/// A later frame and what the session predicts about it.
struct Query {
    later: FrameKey,
    odometry: Odometry,
    descriptor: Option<Vec<f32>>,
    estimate: Option<Pose>,
    root: Option<KeyframeId>,
    bound: Option<f64>,
}

/// Angle between two optical axes. Rotation about the axis does not reduce overlap.
fn axis_angle(a: &Pose, b: &Pose) -> f64 {
    let axis = nalgebra::Vector3::new(0.0, 0.0, -1.0);
    (a.rotation * axis).angle(&(b.rotation * axis))
}

fn normalized(descriptor: &[f32]) -> Option<Vec<f32>> {
    let norm = descriptor.iter().map(|v| v * v).sum::<f32>().sqrt();
    (norm > 0.0 && norm.is_finite()).then(|| descriptor.iter().map(|v| v / norm).collect())
}
