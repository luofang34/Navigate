//! Session odometry uses verified motion only; map reacquisition starts a segment.
use crate::Error;
use navigate_contract::ClockDomainId;
use navigate_recording::Pose;
use navigate_visual::{CameraModel, CameraPose, Frame, OrthographicEstimate};
use navigate_visual_session::{
    Calibration, CameraMount, CaptureTime, ContinuityEpoch, FrameKey, FrameRecord, LensModel,
    RelativeMotion, ScaleSource, SessionConfig, StreamId, VerticalReference, VisualSession,
    from_camera, to_camera,
};
use serde_json::{Value, json};

struct Previous {
    key: FrameKey,
    pose: CameraPose,
    observation: String,
}
pub(super) struct Continuity {
    session: VisualSession,
    previous: Option<Previous>,
    epoch: ContinuityEpoch,
    observed: bool,
}
impl Continuity {
    pub fn new(camera: CameraModel) -> Result<Self, Error> {
        Ok(Self {
            session: VisualSession::new(
                SessionConfig::standard(),
                Calibration {
                    intrinsics: camera,
                    lens: LensModel::Unknown,
                    mount: CameraMount::Unknown,
                    map_vertical: VerticalReference::Unknown,
                },
            )?,
            previous: None,
            epoch: ContinuityEpoch(0),
            observed: false,
        })
    }
    pub fn step(
        &mut self,
        frame: &Frame,
        observation: &str,
        selected: Option<&OrthographicEstimate>,
        refreshed: bool,
    ) -> Result<Value, Error> {
        let connected = selected.is_some() && !refreshed;
        if self.observed && !connected {
            self.epoch.0 = self.epoch.0.wrapping_add(1);
        }
        let key = FrameKey {
            stream: StreamId(0),
            continuity: self.epoch,
            index: frame.stamp.sequence,
        };
        self.session.observe_frame(FrameRecord {
            key,
            media: None,
            capture: CaptureTime {
                clock: ClockDomainId::new(0),
                at_ns: frame.stamp.capture_time_ns,
                error_bound_ns: 0,
            },
            observation_sha256: observation.into(),
        })?;
        if let Some(estimate) = selected
            && connected
        {
            let previous = self.previous.as_ref().ok_or_else(|| {
                Error::Input("surface motion has no preceding accepted pose".into())
            })?;
            self.session.apply_motion(RelativeMotion {
                from: previous.key,
                to: key,
                from_to: from_camera(&previous.pose).inv_mul(&from_camera(&estimate.pose)),
                scale: ScaleSource::SeededMapPoints {
                    map: estimate.map.clone(),
                },
                from_observation_sha256: previous.observation.clone(),
                to_observation_sha256: observation.into(),
            })?;
        }
        let pose = self.session.pose_at(&key)?;
        let events = self.session.drain_events();
        let result = json!({
            "epoch":self.epoch.0,"motion_applied":connected,
            "odometry":selected.and(pose.odometry.as_ref()).map(|o| json!({
                "segment":o.segment.0,"pose":Pose::from(to_camera(&o.pose)),"path_m":o.path_m,
                "scale":o.scale.as_ref().map(|s|format!("{s:?}")),
            })),
            "transition":if selected.is_none() { "unavailable" } else if connected { "tracked_motion" } else { "segment_start_no_motion_bridge" },
            "map_anchors_submitted":self.session.usage().anchors,"dropped_events":events.dropped,
            "scope":"Relative session odometry only; timestamps are used as declared for replay. Separate segments have no measured connecting motion. Map fits remain local-grid proposals outside this session; no geodetic anchor or navigation usability is asserted."
        });
        self.previous = selected.map(|estimate| Previous {
            key,
            pose: estimate.pose,
            observation: observation.into(),
        });
        self.observed = true;
        Ok(result)
    }
}
#[cfg(test)]
mod tests;
