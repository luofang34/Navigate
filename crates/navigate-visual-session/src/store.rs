//! Bounded frame records, odometry segments, and keyframes.

use crate::{
    FrameKey, FrameRecord, ScaleSource, SegmentId, SessionError,
    config::{DriftModel, KeyframePolicy},
    pose::Pose,
};
use std::collections::{BTreeMap, VecDeque};

/// Accumulated odometry of one frame inside its segment.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Odometry {
    pub segment: SegmentId,
    /// Pose in the segment odometry frame. The first frame is the origin.
    pub pose: Pose,
    /// Camera travel since the segment start, in metres.
    pub path_m: f64,
    /// Camera rotation since the segment start, in radians.
    pub turn_rad: f64,
    /// Motion steps since the segment start.
    pub steps: u64,
    /// Scale source of the step into this frame.
    pub scale: Option<ScaleSource>,
}

impl Odometry {
    /// Declared drift between two frames of one segment.
    pub fn drift_to(&self, later: &Self, model: &DriftModel) -> (f64, f64) {
        let path = (later.path_m - self.path_m).abs();
        let turn = (later.turn_rad - self.turn_rad).abs();
        let steps = later.steps.abs_diff(self.steps) as f64;
        (
            model.translation_fraction * path + model.step_floor_m * steps.max(1.0),
            model.rotation_fraction * turn
                + model.rotation_per_m_rad * path
                + model.step_floor_rad * steps.max(1.0),
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FrameState {
    pub record: FrameRecord,
    pub odometry: Option<Odometry>,
}

/// Monotonic keyframe identity. Values are never reused in one session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct KeyframeId(pub u64);

#[derive(Clone, Debug)]
pub(crate) struct Keyframe {
    pub frame: FrameKey,
    pub capture_ns: u64,
    pub observation_sha256: String,
    pub odometry: Odometry,
    /// Current pose estimate in the session map frame.
    pub estimate: Pose,
    pub descriptor: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub(crate) struct Segment {
    /// Map reference kept when the memory bound removed every keyframe.
    pub carry: Option<Carry>,
    pub first: FrameKey,
    /// Newest frame with odometry.
    pub head: FrameKey,
    pub keyframes: Vec<KeyframeId>,
}

/// The map correction of a segment whose keyframes the memory bound removed.
#[derive(Clone, Debug)]
pub(crate) struct Carry {
    pub correction: Pose,
    pub sigma_m: f64,
    /// Attitude bound of the removed keyframe, in radians.
    pub sigma_rad: f64,
    /// Odometry of the removed keyframe, for the drift to the next one.
    pub odometry: Odometry,
}

/// Frame ring with a key index. Keys map to absolute ring positions.
#[derive(Debug, Default)]
pub(crate) struct Frames {
    ring: VecDeque<FrameState>,
    base: u64,
    index: BTreeMap<FrameKey, u64>,
    /// Newest kept frame and capture time of each epoch.
    latest: BTreeMap<(crate::StreamId, crate::ContinuityEpoch), (FrameKey, u64)>,
}

impl Frames {
    pub fn get(&self, key: &FrameKey) -> Option<&FrameState> {
        let at = self.index.get(key)?.checked_sub(self.base)?;
        self.ring.get(usize::try_from(at).ok()?)
    }

    pub fn get_mut(&mut self, key: &FrameKey) -> Option<&mut FrameState> {
        let at = self.index.get(key)?.checked_sub(self.base)?;
        self.ring.get_mut(usize::try_from(at).ok()?)
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// True while the ring keeps a frame of the epoch.
    pub fn keeps(&self, track: &(crate::StreamId, crate::ContinuityEpoch)) -> bool {
        self.latest.contains_key(track)
    }

    /// Append a frame. Index and capture time must increase inside one epoch.
    pub fn push(
        &mut self,
        record: FrameRecord,
        max_frames: usize,
    ) -> Result<Option<FrameKey>, SessionError> {
        let track = record.key.track();
        if let Some((previous, at_ns)) = self.latest.get(&track)
            && (record.key.index <= previous.index || record.capture.at_ns <= *at_ns)
        {
            return Err(SessionError::FrameOrder {
                previous: *previous,
                received: record.key,
            });
        }
        if self.index.contains_key(&record.key) {
            return Err(SessionError::RepeatedEvidence(record.observation_sha256));
        }
        self.latest
            .insert(track, (record.key, record.capture.at_ns));
        let position = self.base + self.ring.len() as u64;
        self.index.insert(record.key, position);
        self.ring.push_back(FrameState {
            record,
            odometry: None,
        });
        let mut evicted = None;
        while self.ring.len() > max_frames {
            if let Some(old) = self.ring.pop_front() {
                self.index.remove(&old.record.key);
                // An epoch with no kept frame needs no order record.
                let track = old.record.key.track();
                if self
                    .latest
                    .get(&track)
                    .is_some_and(|(key, _)| *key == old.record.key)
                {
                    self.latest.remove(&track);
                }
                self.base += 1;
                evicted = Some(old.record.key);
            }
        }
        Ok(evicted)
    }
}

/// True when the frame is far enough from the last keyframe to become one.
pub(crate) fn wants_keyframe(
    last: &Keyframe,
    odometry: &Odometry,
    policy: &KeyframePolicy,
) -> bool {
    let relative = last.odometry.pose.inv_mul(&odometry.pose);
    relative.translation.vector.norm() >= policy.translation_m
        || relative.rotation.angle() >= policy.rotation_rad
}
