//! The session state machine.

mod anchoring;
mod ground;
mod keyframes;
mod optimize;
mod query;
mod relocate;
mod revisit;
mod topology;

use crate::{
    Calibration, ContinuityEpoch, EventBatch, FrameKey, FrameRecord, RelativeMotion, SegmentId,
    SegmentStart, SessionConfig, SessionError, SessionEvent, StreamId,
    anchor::{AnchorBudget, AnchorObservation},
    events::EventQueue,
    evidence::{Ledger, MapCell},
    pose::{Pose, finite},
    store::{Frames, Keyframe, KeyframeId, Odometry, Segment},
};
use nalgebra::Vector3;
use navigate_contract::ClockDomainId;
use navigate_visual::LocalFrame;
use std::collections::BTreeMap;

pub use anchoring::AnchorDecision;
pub use ground::{GroundDecision, GroundPlaneObservation, TerrainNormal};
pub use query::{Confirmation, FramePose, MapPose, OdometryPose, Unlocated, Usability};
pub use relocate::{GroundView, PositionBasis, RelocationCandidate, TiltBasis};
pub use revisit::{RevisitCandidate, RevisitConstraint, RevisitDecision, RevisitEvidence};

/// An accepted anchor, transferred to its keyframe.
#[derive(Clone, Debug)]
struct Anchor {
    observation: AnchorObservation,
    /// Map pose of the keyframe implied by the anchor.
    keyframe_pose: Pose,
    budget: AnchorBudget,
    /// Drift between the anchored frame and its keyframe, in metres.
    transfer_m: f64,
    /// Drift between the anchored frame and its keyframe, in radians.
    transfer_rad: f64,
    cell: MapCell,
    /// Capture time of the anchored frame.
    capture_ns: u64,
}

/// A ground-plane direction transferred to its keyframe.
#[derive(Clone, Debug)]
struct Ground {
    /// Terrain normal in the world frame.
    world: Vector3<f64>,
    /// The same normal in the keyframe camera frame.
    camera: Vector3<f64>,
    sigma_rad: f64,
    /// Frame of the observation.
    frame: FrameKey,
}

/// A frozen prior that keeps information from evicted keyframes.
#[derive(Clone, Copy, Debug)]
struct Frozen {
    pose: Pose,
    sigma_m: f64,
    sigma_rad: f64,
}

#[derive(Clone, Debug)]
struct Closure {
    constraint: RevisitConstraint,
    earlier: KeyframeId,
    later: KeyframeId,
    /// Pose of the later keyframe in the earlier keyframe camera.
    measured: Pose,
    sigma_m: f64,
    sigma_rad: f64,
}

/// Memory use of a session, in element counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionUsage {
    /// Frame records.
    pub frames: usize,
    /// Keyframes.
    pub keyframes: usize,
    /// Accepted anchors.
    pub anchors: usize,
    /// Anchors that wait for agreement.
    pub pending_anchors: usize,
    /// Revisit closures.
    pub closures: usize,
    /// Map-cell bias variables.
    pub bias_cells: usize,
    /// Odometry segments.
    pub segments: usize,
    /// Queued events.
    pub events: usize,
    /// Fusion ledger cells.
    pub ledger_cells: usize,
}

/// Result of a motion step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionOutcome {
    /// The end frame became a keyframe. The host can attach a retrieval descriptor.
    pub keyframe: bool,
}

/// Continuous odometry, map anchoring, revisit closure, and pose revision for one session.
///
/// The session is sans-IO and deterministic. The host decodes frames, runs
/// matchers and renders references, then reports results. The session never
/// reads a clock; capture times come with each frame.
pub struct VisualSession {
    config: SessionConfig,
    calibration: Calibration,
    clock: Option<ClockDomainId>,
    frames: Frames,
    segments: BTreeMap<SegmentId, Segment>,
    tracks: BTreeMap<(StreamId, ContinuityEpoch), SegmentId>,
    next_segment: u32,
    keyframes: BTreeMap<KeyframeId, Keyframe>,
    next_keyframe: u64,
    anchors: BTreeMap<KeyframeId, Anchor>,
    pending: BTreeMap<SegmentId, Vec<anchoring::Pending>>,
    closures: Vec<Closure>,
    frozen: BTreeMap<KeyframeId, Frozen>,
    grounds: BTreeMap<KeyframeId, Ground>,
    biases: BTreeMap<MapCell, Vector3<f64>>,
    local_frame: Option<LocalFrame>,
    ledger: Ledger,
    events: EventQueue,
    revision: u64,
    topology: query::Topology,
}

impl VisualSession {
    /// Start a session.
    ///
    /// # Errors
    /// Rejects an invalid configuration or camera intrinsics.
    pub fn new(config: SessionConfig, calibration: Calibration) -> Result<Self, SessionError> {
        config.validate()?;
        calibration
            .intrinsics
            .validate()
            .map_err(|_| SessionError::Invalid {
                field: "camera intrinsics",
            })?;
        Ok(Self {
            config,
            calibration,
            clock: None,
            frames: Frames::default(),
            segments: BTreeMap::new(),
            tracks: BTreeMap::new(),
            next_segment: 0,
            keyframes: BTreeMap::new(),
            next_keyframe: 0,
            anchors: BTreeMap::new(),
            pending: BTreeMap::new(),
            closures: Vec::new(),
            frozen: BTreeMap::new(),
            grounds: BTreeMap::new(),
            biases: BTreeMap::new(),
            local_frame: None,
            ledger: Ledger::default(),
            events: EventQueue::default(),
            revision: 0,
            topology: query::Topology::default(),
        })
    }

    /// The session calibration.
    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    /// Revision number of the map poses. It changes with each trajectory revision.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The local frame of all map poses, after the first anchor.
    pub fn local_frame(&self) -> Option<LocalFrame> {
        self.local_frame
    }

    /// Register a processed frame. The first frame of an epoch starts a segment.
    ///
    /// # Errors
    /// Rejects a frame from another clock domain, an empty digest, or a frame
    /// whose index or capture time does not increase in its epoch.
    pub fn observe_frame(&mut self, record: FrameRecord) -> Result<(), SessionError> {
        if record.observation_sha256.is_empty() {
            return Err(SessionError::Invalid {
                field: "observation digest",
            });
        }
        if self
            .clock
            .is_some_and(|clock| clock != record.capture.clock)
        {
            return Err(SessionError::ClockDomain(record.key));
        }
        let key = record.key;
        let evicted = self
            .frames
            .push(record.clone(), self.config.limits.max_frames)?;
        self.clock = Some(record.capture.clock);
        if evicted.is_some() {
            self.prune_segments();
        }
        if !self.tracks.contains_key(&key.track()) {
            self.start_segment(key, SegmentStart::NewEpoch);
        }
        Ok(())
    }

    /// Integrate relative motion into the odometry of the epoch.
    ///
    /// Odometry is continuous and never revised. A motion step without metric
    /// scale is refused. The host then starts a new continuity epoch, or waits
    /// for a map anchor, which starts a new segment.
    ///
    /// # Errors
    /// Rejects motion that does not extend the segment head, unknown frames,
    /// digest mismatches, non-finite poses, and motion without metric scale.
    pub fn apply_motion(&mut self, motion: RelativeMotion) -> Result<MotionOutcome, SessionError> {
        let (segment_id, from) = self.motion_start(&motion)?;
        let to_state = self
            .frames
            .get(&motion.to)
            .ok_or(SessionError::UnknownFrame(motion.to))?;
        if to_state.record.observation_sha256 != motion.to_observation_sha256 {
            return Err(SessionError::EvidenceMismatch(motion.to));
        }
        if to_state.odometry.is_some()
            || motion.to.track() != motion.from.track()
            || motion.to.index <= motion.from.index
        {
            return Err(SessionError::MotionOrder {
                from: motion.from,
                to: motion.to,
            });
        }
        if !motion.scale.is_metric() {
            return Err(SessionError::ScaleUnobservable {
                from: motion.from,
                to: motion.to,
            });
        }
        if !finite(&motion.from_to) {
            return Err(SessionError::Invalid {
                field: "relative motion",
            });
        }
        let odometry = Odometry {
            segment: segment_id,
            pose: crate::pose::compose(&from.pose, &motion.from_to),
            path_m: from.path_m + motion.from_to.translation.vector.norm(),
            turn_rad: from.turn_rad + motion.from_to.rotation.angle(),
            steps: from.steps.wrapping_add(1),
            scale: Some(motion.scale),
        };
        if let Some(state) = self.frames.get_mut(&motion.to) {
            state.odometry = Some(odometry.clone());
        }
        if let Some(segment) = self.segments.get_mut(&segment_id) {
            segment.head = motion.to;
        }
        let keyframe = self.extend_keyframes(segment_id, motion.to, &odometry);
        Ok(MotionOutcome { keyframe })
    }

    fn motion_start(&self, motion: &RelativeMotion) -> Result<(SegmentId, Odometry), SessionError> {
        let state = self
            .frames
            .get(&motion.from)
            .ok_or(SessionError::UnknownFrame(motion.from))?;
        if state.record.observation_sha256 != motion.from_observation_sha256 {
            return Err(SessionError::EvidenceMismatch(motion.from));
        }
        let from = state.odometry.clone().ok_or(SessionError::MotionOrder {
            from: motion.from,
            to: motion.to,
        })?;
        let head = self.segments.get(&from.segment).map(|s| s.head);
        if head != Some(motion.from) || self.tracks.get(&motion.from.track()) != Some(&from.segment)
        {
            return Err(SessionError::MotionOrder {
                from: motion.from,
                to: motion.to,
            });
        }
        Ok((from.segment, from))
    }

    fn start_segment(&mut self, first: FrameKey, reason: SegmentStart) -> SegmentId {
        let id = SegmentId(self.next_segment);
        self.next_segment = self.next_segment.wrapping_add(1);
        let odometry = Odometry {
            segment: id,
            pose: Pose::identity(),
            path_m: 0.0,
            turn_rad: 0.0,
            steps: 0,
            scale: None,
        };
        if let Some(state) = self.frames.get_mut(&first) {
            state.odometry = Some(odometry.clone());
        }
        self.segments.insert(
            id,
            Segment {
                carry: None,
                first,
                head: first,
                keyframes: Vec::new(),
            },
        );
        self.tracks.insert(first.track(), id);
        self.push_keyframe(id, first, &odometry, Pose::identity(), &[]);
        self.event(SessionEvent::SegmentStarted {
            segment: id,
            first,
            reason,
        });
        id
    }

    /// Take all queued events.
    pub fn drain_events(&mut self) -> EventBatch {
        self.events.drain()
    }

    /// Current memory use.
    pub fn usage(&self) -> SessionUsage {
        SessionUsage {
            frames: self.frames.len(),
            keyframes: self.keyframes.len(),
            anchors: self.anchors.len(),
            pending_anchors: self.pending.values().map(Vec::len).sum(),
            closures: self.closures.len(),
            bias_cells: self.biases.len(),
            segments: self.segments.len(),
            events: self.events.len(),
            ledger_cells: self.ledger.len(),
        }
    }

    /// Heights of a frame at the current revision.
    ///
    /// # Errors
    /// Returns `UnknownFrame` for a frame that is not in the session.
    pub fn height_at(
        &self,
        frame: &FrameKey,
        inputs: &crate::HeightInputs,
    ) -> Result<crate::HeightEstimate, SessionError> {
        Ok(crate::estimate_height(
            &self.pose_at(frame)?,
            &self.calibration,
            inputs,
        ))
    }

    fn event(&mut self, event: SessionEvent) {
        self.events.push(event, self.config.limits.max_events);
    }
}
