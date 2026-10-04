//! Per-frame odometry and map poses.

use super::VisualSession;
use crate::{
    CaptureTime, FrameKey, MediaTime, ScaleSource, SegmentId, SessionError,
    pose::{Pose, blend, off_nadir},
    store::{KeyframeId, Odometry},
};
use navigate_visual::LocalFrame;
use std::collections::BTreeMap;

/// Cached connectivity and bounds. Every graph change refreshes it.
#[derive(Debug, Default)]
pub(super) struct Topology {
    pub bounds: BTreeMap<KeyframeId, Option<f64>>,
    /// Attitude bound from map anchors and kept priors.
    pub attitude: BTreeMap<KeyframeId, Option<f64>>,
    /// Tilt bound from map evidence and ground planes.
    pub tilt: BTreeMap<KeyframeId, Option<f64>>,
    /// Anchor capture times of each component.
    pub anchor_times: BTreeMap<KeyframeId, Vec<u64>>,
    pub components: BTreeMap<KeyframeId, KeyframeId>,
}

/// Bounds of one frame. Each value is a conservative sum, not a covariance.
#[derive(Clone, Copy, Debug)]
pub(super) struct Bounds {
    pub position_m: Option<f64>,
    pub tilt_rad: Option<f64>,
    pub attitude_rad: Option<f64>,
}

impl Bounds {
    fn tighter(&self, other: &Self) -> Self {
        let min = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        };
        Self {
            position_m: min(self.position_m, other.position_m),
            tilt_rad: min(self.tilt_rad, other.tilt_rad),
            attitude_rad: min(self.attitude_rad, other.attitude_rad),
        }
    }
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

/// How many recent anchors support a located frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confirmation {
    /// Fewer than two anchors within the recent window of this frame.
    Unconfirmed,
    /// Two or more anchors of the connected trajectory within the recent window.
    Confirmed {
        /// Number of anchors within the window.
        anchors: usize,
    },
}

/// What a located frame can be used for. Each use needs its own evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usability {
    /// An anchor of the connected trajectory is within the recent window.
    pub recent_map_support: bool,
    /// The position bound permits use as a navigation position.
    pub navigation: bool,
    /// Position, tilt, and map-attitude bounds permit projection of the image
    /// onto the ground.
    pub ground_projection: bool,
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
        /// Conservative tilt bound from map anchors and ground planes, in
        /// radians, or `None` when nothing observes tilt.
        tilt_bound_rad: Option<f64>,
        /// Conservative attitude bound from map anchors alone, in radians,
        /// including heading, or `None` when no anchor observes attitude.
        attitude_bound_rad: Option<f64>,
        /// Capture-time distance to the nearest anchor of the connected
        /// trajectory, in nanoseconds.
        nearest_anchor_ns: Option<u64>,
        /// Anchor support.
        confirmation: Confirmation,
        /// Permitted uses of this pose.
        usability: Usability,
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
            attitude: self.attitude_bounds(),
            tilt: self.tilt_bounds(),
            anchor_times: self.anchor_times(),
            components: self.components(),
        };
    }

    /// Correction, bounds, and the nearest earlier keyframe for a frame with odometry.
    pub(super) fn locate(
        &self,
        frame: &FrameKey,
        odometry: &Odometry,
    ) -> Option<(Pose, Bounds, KeyframeId)> {
        let (before, after) = self.neighbours(odometry.segment, frame.index);
        let side = |id: Option<KeyframeId>| {
            let id = id?;
            let kf = self.keyframes.get(&id)?;
            let (m, rad) = kf.odometry.drift_to(odometry, &self.config.drift);
            let grow = |map: &BTreeMap<KeyframeId, Option<f64>>, add: f64| {
                map.get(&id).copied().flatten().map(|b| b + add)
            };
            let attitude = self.topology.attitude.get(&id).copied().flatten();
            let travel = (odometry.path_m - kf.odometry.path_m).abs();
            let turned = travel * attitude.unwrap_or(1.0).min(1.0).sin();
            let bounds = Bounds {
                position_m: grow(&self.topology.bounds, m + turned),
                tilt_rad: grow(&self.topology.tilt, rad),
                attitude_rad: grow(&self.topology.attitude, rad),
            };
            Some((self.correction(id)?, bounds, id, kf.odometry.path_m))
        };
        match (side(before), side(after)) {
            (Some(a), Some(b)) => {
                let span = b.3 - a.3;
                let t = if span > 0.0 {
                    ((odometry.path_m - a.3) / span).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                Some((blend(&a.0, &b.0, t), a.1.tighter(&b.1), a.2))
            }
            (Some(a), None) | (None, Some(a)) => Some((a.0, a.1, a.2)),
            (None, None) => None,
        }
    }

    /// Predicted map pose and position bound of a located frame.
    pub(super) fn predict(&self, frame: &FrameKey) -> Option<(Pose, f64)> {
        let odometry = self.frames.get(frame)?.odometry.clone()?;
        let (correction, bounds, _) = self.locate(frame, &odometry)?;
        Some((
            crate::pose::compose(&correction, &odometry.pose),
            bounds.position_m?,
        ))
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
        let located = self.locate(frame, odometry);
        let (Some((correction, bounds, keyframe)), Some(local_frame)) = (located, self.local_frame)
        else {
            return MapPose::Unlocated(Unlocated::NoAnchor);
        };
        let Some(bound) = bounds.position_m else {
            return MapPose::Unlocated(Unlocated::NoAnchor);
        };
        // A bound that is not a number is not a usable bound.
        if bound.is_nan() || bound > self.config.max_located_bound_m {
            return MapPose::Unlocated(Unlocated::BoundExceeded { bound_m: bound });
        }
        let capture = self.frames.get(frame).map_or(0, |s| s.record.capture.at_ns);
        let times = self
            .topology
            .components
            .get(&keyframe)
            .and_then(|root| self.topology.anchor_times.get(root))
            .map(Vec::as_slice)
            .unwrap_or_default();
        let window = self.config.output.recent_anchor_ns;
        let recent = times
            .iter()
            .filter(|t| t.abs_diff(capture) <= window)
            .count();
        let nearest_anchor_ns = times.iter().map(|t| t.abs_diff(capture)).min();
        let pose = crate::pose::compose(&correction, &odometry.pose);
        let output = &self.config.output;
        let heading_ok = bounds
            .attitude_rad
            .is_some_and(|a| a <= output.projection_attitude_bound_rad);
        let projectable = heading_ok
            && bounds.tilt_rad.is_some_and(|tilt| {
                tilt <= output.projection_tilt_bound_rad
                    && off_nadir(&pose) + tilt <= output.projection_off_nadir_rad
            });
        MapPose::Located {
            pose,
            map_from_odom: correction,
            position_bound_m: bound,
            tilt_bound_rad: bounds.tilt_rad,
            attitude_bound_rad: bounds.attitude_rad,
            nearest_anchor_ns,
            confirmation: if recent >= 2 {
                Confirmation::Confirmed { anchors: recent }
            } else {
                Confirmation::Unconfirmed
            },
            usability: Usability {
                recent_map_support: recent >= 1,
                navigation: bound <= output.navigation_bound_m,
                ground_projection: projectable && bound <= output.projection_bound_m,
            },
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
