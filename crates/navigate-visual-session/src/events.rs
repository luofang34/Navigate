//! Session events and their bounded queue.

use crate::{FrameKey, pose::Pose};
use std::collections::VecDeque;

/// Internal identity of one odometry segment. Values are never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentId(pub u32);

/// Why a new odometry segment started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentStart {
    /// The first frame of a new stream or continuity epoch.
    NewEpoch,
    /// A map anchor arrived for a frame that tracking did not reach.
    AnchorAfterGap,
}

/// Why an anchor did not enter the trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnchorRejection {
    /// The pose disagrees with the trajectory by more than the gate allows.
    Inconsistent {
        /// Horizontal distance from the predicted position, in metres.
        distance_m: f64,
        /// Gate radius at this frame, in metres.
        gate_m: f64,
    },
    /// The host marked the map area as changed.
    ChangedRegion,
    /// The anchor waits for agreement from other frames.
    AwaitingConsensus {
        /// Mutually consistent anchors so far, including this one.
        agreeing: usize,
    },
    /// The waiting list of anchors is full; the oldest entry was removed.
    PendingOverflow,
    /// The anchor entered the graph, but it remained an outlier after
    /// optimization and was retracted.
    Retracted,
}

/// What caused a trajectory revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RevisionCause {
    /// A map anchor entered the graph.
    Anchor,
    /// Agreeing anchors moved a track to a new map position.
    Relocation,
    /// A revisit closure entered the graph.
    Closure,
    /// A closure or an anchor left the graph after optimization.
    Retraction,
    /// A ground-plane observation constrained a keyframe tilt.
    GroundPlane,
}

/// Frames of one segment whose map pose changed.
#[derive(Clone, Debug, PartialEq)]
pub struct RevisedRange {
    /// Revised segment.
    pub segment: SegmentId,
    /// First frame with a changed map pose.
    pub from: FrameKey,
    /// Newest frame of the segment at revision time.
    pub to: FrameKey,
    /// Largest keyframe position change, in metres.
    pub max_shift_m: f64,
    /// Largest keyframe rotation change, in radians.
    pub max_rotation_rad: f64,
    /// New map-from-odometry transform at the segment head.
    pub map_from_odom: Pose,
}

/// A change of past map poses. Consumers re-project the frames in each range.
///
/// Odometry poses do not change. A revision is not vehicle motion.
#[derive(Clone, Debug, PartialEq)]
pub struct TrajectoryRevision {
    /// Revision number after this change.
    pub revision: u64,
    /// Cause of the change.
    pub cause: RevisionCause,
    /// Changed frame ranges.
    pub ranges: Vec<RevisedRange>,
}

/// One session event.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionEvent {
    /// A new odometry segment started.
    SegmentStarted {
        /// New segment.
        segment: SegmentId,
        /// Its first frame.
        first: FrameKey,
        /// Reason for the start.
        reason: SegmentStart,
    },
    /// A map anchor entered the trajectory.
    AnchorAccepted {
        /// Anchored frame.
        frame: FrameKey,
        /// Fusion admission label of the anchor.
        eligibility: crate::FusionEligibility,
    },
    /// A map anchor did not enter the trajectory.
    AnchorRejected {
        /// Frame of the anchor.
        frame: FrameKey,
        /// Reason.
        reason: AnchorRejection,
    },
    /// A ground-plane observation constrains a keyframe tilt.
    GroundPlaneAccepted {
        /// Frame of the observation.
        frame: FrameKey,
    },
    /// A ground-plane observation left the graph after optimization.
    GroundPlaneRetracted {
        /// Frame of the observation.
        frame: FrameKey,
    },
    /// A ground-plane observation disagrees with attitude from map anchors.
    GroundPlaneRejected {
        /// Frame of the observation.
        frame: FrameKey,
    },
    /// An accepted anchor left the trajectory after a relocation.
    AnchorRetracted {
        /// Frame of the anchor.
        frame: FrameKey,
    },
    /// A revisit closure entered the trajectory.
    ClosureAccepted {
        /// Earlier keyframe.
        earlier: FrameKey,
        /// Later frame.
        later: FrameKey,
    },
    /// A revisit closure left the trajectory after optimization.
    ClosureRetracted {
        /// Earlier keyframe.
        earlier: FrameKey,
        /// Later frame.
        later: FrameKey,
        /// Normalized residual after optimization.
        residual_sigma: f64,
    },
    /// Past map poses changed.
    TrajectoryRevised(TrajectoryRevision),
    /// The memory bound removed closures that used a keyframe.
    ClosureEvicted {
        /// Frame of the removed keyframe.
        keyframe: FrameKey,
    },
    /// The memory bound removed the oldest closure.
    ClosureLimit {
        /// Earlier keyframe of the removed closure.
        earlier: FrameKey,
        /// Later frame of the removed closure.
        later: FrameKey,
    },
    /// The memory bound removed a keyframe.
    KeyframeEvicted {
        /// Frame of the keyframe.
        frame: FrameKey,
    },
}

/// Events taken from the queue.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EventBatch {
    /// Events in order.
    pub events: Vec<SessionEvent>,
    /// Events removed because the queue was full.
    pub dropped: u64,
    /// True when a dropped event revised poses. Re-project all frames.
    pub full_revision_required: bool,
}

#[derive(Debug, Default)]
pub(crate) struct EventQueue {
    events: VecDeque<SessionEvent>,
    dropped: u64,
    full_revision: bool,
}

impl EventQueue {
    pub fn push(&mut self, event: SessionEvent, capacity: usize) {
        self.events.push_back(event);
        while self.events.len() > capacity {
            if let Some(old) = self.events.pop_front() {
                self.dropped = self.dropped.wrapping_add(1);
                if matches!(old, SessionEvent::TrajectoryRevised(_)) {
                    self.full_revision = true;
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn drain(&mut self) -> EventBatch {
        let batch = EventBatch {
            events: self.events.drain(..).collect(),
            dropped: self.dropped,
            full_revision_required: self.full_revision,
        };
        self.dropped = 0;
        self.full_revision = false;
        batch
    }
}
