//! Per-frame odometry and map poses.

use super::VisualSession;
use crate::{
    CaptureTime, FrameKey, MediaTime, ScaleSource, SegmentId, SessionError,
    pose::{Pose, blend},
    store::{KeyframeId, Odometry},
};
use navigate_visual::LocalFrame;
use std::collections::BTreeMap;

/// Cached connectivity and bounds. Every graph change refreshes it.
#[derive(Debug, Default)]
pub(super) struct Topology {
    pub bounds: BTreeMap<KeyframeId, Option<f64>>,
    pub components: BTreeMap<KeyframeId, KeyframeId>,
    pub anchors: BTreeMap<KeyframeId, usize>,
}

/// Continuous odometry of one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct OdometryPose {
    /// Segment of the odometry frame.
    pub segment: SegmentId,
    /// Camera pose in the segment odometry frame. It never changes.
    pub pose: Pose,
    /// Camera travel since the segment start, in metres.
    pub path_m: f64,
    /// Scale source of the step into this frame. `None` at a segment start.
    pub scale: Option<ScaleSource>,
}

/// How many independent anchors support a located frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confirmation {
    /// One anchor, with no agreement from another frame.
    Unconfirmed,
    /// Two or more anchors or kept priors in the connected trajectory.
    Confirmed {
        /// Number of anchors and kept priors.
        anchors: usize,
    },
}

/// Why a frame has no map pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unlocated {
    /// Tracking did not reach this frame.
    NoOdometry,
    /// No anchor or revisit joins this frame to the map.
    NoAnchor,
    /// The position bound is larger than the configured limit.
    BoundExceeded {
        /// Position bound, in metres.
        bound_m: f64,
    },
}

/// The map pose of one frame, or the reason that it has none.
#[derive(Clone, Debug, PartialEq)]
pub enum MapPose {
    /// The frame is joined to map anchors.
    Located {
        /// Camera pose in the session local frame.
        pose: Pose,
        /// Map-from-odometry transform used for this frame.
        map_from_odom: Pose,
        /// Conservative position bound, in metres. It is not a covariance.
        position_bound_m: f64,
        /// Anchor support.
        confirmation: Confirmation,
        /// Local frame of `pose`.
        local_frame: LocalFrame,
    },
    /// The frame has no map pose.
    Unlocated(Unlocated),
}

/// Odometry and map pose of one frame at one revision.
#[derive(Clone, Debug, PartialEq)]
pub struct FramePose {
    /// Frame identity.
    pub frame: FrameKey,
    /// Media presentation time of the frame.
    pub media: Option<MediaTime>,
    /// Capture time of the frame.
    pub capture: CaptureTime,
    /// Odometry, when tracking reached the frame.
    pub odometry: Option<OdometryPose>,
    /// Map pose.
    pub map: MapPose,
    /// Session revision of the map pose.
    pub revision: u64,
}

impl VisualSession {
    pub(super) fn refresh(&mut self) {
        self.topology = Topology {
            bounds: self.bounds(),
            components: self.components(),
            anchors: self.anchor_counts(),
        };
    }

    /// Correction and bound for a frame with odometry.
    fn locate(
        &self,
        frame: &FrameKey,
        odometry: &Odometry,
    ) -> Option<(Pose, Option<f64>, KeyframeId)> {
        let (before, after) = self.neighbours(odometry.segment, frame.index);
        let side = |id: Option<KeyframeId>| {
            let id = id?;
            let kf = self.keyframes.get(&id)?;
            let correction = self.correction(id)?;
            let bound = self
                .topology
                .bounds
                .get(&id)
                .copied()
                .flatten()
                .map(|b| b + kf.odometry.drift_to(odometry, &self.config.drift).0);
            Some((correction, bound, id, kf.odometry.path_m))
        };
        match (side(before), side(after)) {
            (Some(a), Some(b)) => {
                let span = b.3 - a.3;
                let t = if span > 0.0 {
                    ((odometry.path_m - a.3) / span).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let bound = match (a.1, b.1) {
                    (Some(x), Some(y)) => Some(x.min(y)),
                    (x, y) => x.or(y),
                };
                Some((blend(&a.0, &b.0, t), bound, a.2))
            }
            (Some(a), None) | (None, Some(a)) => Some((a.0, a.1, a.2)),
            (None, None) => None,
        }
    }

    /// Predicted map pose and position bound of a located frame.
    pub(super) fn predict(&self, frame: &FrameKey) -> Option<(Pose, f64)> {
        let odometry = self.frames.get(frame)?.odometry.clone()?;
        let (correction, bound, _) = self.locate(frame, &odometry)?;
        Some((correction * odometry.pose, bound?))
    }

    /// Odometry and map pose of a frame at the current revision.
    ///
    /// Frames between keyframes blend the corrections of both keyframes along
    /// the travelled path. Consumers bind each displayed frame to this result.
    ///
    /// # Errors
    /// Returns `UnknownFrame` for a frame that is not in the session or that
    /// the memory bound removed.
    pub fn pose_at(&self, frame: &FrameKey) -> Result<FramePose, SessionError> {
        let state = self
            .frames
            .get(frame)
            .ok_or(SessionError::UnknownFrame(*frame))?;
        let odometry = state.odometry.as_ref();
        let map = match odometry {
            None => MapPose::Unlocated(Unlocated::NoOdometry),
            Some(o) => self.map_pose(frame, o),
        };
        Ok(FramePose {
            frame: *frame,
            media: state.record.media,
            capture: state.record.capture,
            odometry: odometry.map(|o| OdometryPose {
                segment: o.segment,
                pose: o.pose,
                path_m: o.path_m,
                scale: o.scale.clone(),
            }),
            map,
            revision: self.revision,
        })
    }

    fn map_pose(&self, frame: &FrameKey, odometry: &Odometry) -> MapPose {
        let (Some((correction, Some(bound), keyframe)), Some(local_frame)) =
            (self.locate(frame, odometry), self.local_frame)
        else {
            return MapPose::Unlocated(Unlocated::NoAnchor);
        };
        // A bound that is not a number is not a usable bound.
        if bound.is_nan() || bound > self.config.max_located_bound_m {
            return MapPose::Unlocated(Unlocated::BoundExceeded { bound_m: bound });
        }
        let anchors = self
            .topology
            .components
            .get(&keyframe)
            .and_then(|root| self.topology.anchors.get(root))
            .copied()
            .unwrap_or(0);
        let confirmation = if anchors >= 2 {
            Confirmation::Confirmed { anchors }
        } else {
            Confirmation::Unconfirmed
        };
        MapPose::Located {
            pose: correction * odometry.pose,
            map_from_odom: correction,
            position_bound_m: bound,
            confirmation,
            local_frame,
        }
    }

    /// Map-from-odometry transform at the head of a segment, when it is located.
    pub fn map_from_odom(&self, segment: SegmentId) -> Option<Pose> {
        let head = self.segments.get(&segment)?.head;
        match self.pose_at(&head).ok()?.map {
            MapPose::Located { map_from_odom, .. } => Some(map_from_odom),
            MapPose::Unlocated(_) => None,
        }
    }
}
