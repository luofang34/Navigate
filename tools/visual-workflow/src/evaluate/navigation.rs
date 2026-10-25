//! Separate map anchors from motion that depends on seeded map points.
mod report;

use crate::Error;
use navigate_contract::ClockDomainId;
use navigate_visual::{CameraModel, Estimate, Frame};
use navigate_visual_session::{
    AnchorObservation, Calibration, CameraMount, CaptureTime, ContinuityEpoch, FrameKey,
    FrameRecord, LensModel, MapReliability, RegionChange, RelativeMotion, ScaleSource,
    SessionConfig, StreamId, SurfaceModel, VerticalReference, VisualSession, from_camera,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Settings {
    pub map_horizontal_sigma_m: f64,
    pub map_vertical_sigma_m: f64,
    pub max_object_height_m: f64,
    pub capture_time_error_bound_ns: u64,
    pub max_located_bound_m: f64,
}
impl Settings {
    pub fn valid(self) -> bool {
        [
            self.map_horizontal_sigma_m,
            self.map_vertical_sigma_m,
            self.max_located_bound_m,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
            && self.max_object_height_m.is_finite()
            && self.max_object_height_m >= 0.0
    }
    fn reliability(self) -> MapReliability {
        MapReliability {
            horizontal_m: self.map_horizontal_sigma_m,
            vertical_m: self.map_vertical_sigma_m,
            imagery_age_years: None,
            age_growth_m_per_year: 0.0,
            unknown_age_years: 0.0,
            surface: SurfaceModel::BareEarth {
                max_object_height_m: self.max_object_height_m,
            },
            change: RegionChange::Unknown,
        }
    }
}

#[derive(Default)]
pub(super) struct Evidence {
    pub tracked: Option<Estimate>,
    pub matched: Option<Estimate>,
}
pub(super) struct Output {
    pub record: Value,
    pub position: Option<[f64; 3]>,
}
struct Previous {
    key: FrameKey,
    estimate: Estimate,
}
pub(super) struct Navigation {
    session: VisualSession,
    settings: Settings,
    previous: Option<Previous>,
    continuity: ContinuityEpoch,
}
impl Navigation {
    pub fn new(camera: CameraModel, settings: Settings) -> Result<Self, Error> {
        if !settings.valid() {
            return Err(Error::Input("invalid navigation error declarations".into()));
        }
        let config = SessionConfig {
            max_located_bound_m: settings.max_located_bound_m,
            ..SessionConfig::standard()
        };
        Ok(Self {
            session: VisualSession::new(
                config,
                Calibration {
                    intrinsics: camera,
                    lens: LensModel::Unknown,
                    mount: CameraMount::Unknown,
                    map_vertical: VerticalReference::Unknown,
                },
            )?,
            settings,
            previous: None,
            continuity: ContinuityEpoch(0),
        })
    }

    pub fn step(
        &mut self,
        frame: &Frame,
        evidence: &Evidence,
        selected: Option<&Estimate>,
    ) -> Result<Output, Error> {
        if frame.stamp.sequence > 0 && evidence.tracked.is_none() {
            self.continuity.0 = self.continuity.0.wrapping_add(1);
        }
        let key = FrameKey {
            stream: StreamId(0),
            continuity: self.continuity,
            index: frame.stamp.sequence,
        };
        self.session.observe_frame(FrameRecord::from_frame(
            key,
            None,
            CaptureTime {
                clock: ClockDomainId::new(0),
                at_ns: frame.stamp.capture_time_ns,
                error_bound_ns: self.settings.capture_time_error_bound_ns,
            },
            frame,
        ))?;
        if let Some(tracked) = &evidence.tracked {
            self.motion(key, tracked)?;
        }
        let decision = evidence
            .matched
            .as_ref()
            .map(|estimate| {
                self.session.submit_anchor(AnchorObservation::from_estimate(
                    key,
                    estimate,
                    self.settings.reliability(),
                ))
            })
            .transpose()?;
        let pose = self.session.pose_at(&key)?;
        let events = self.session.drain_events();
        let output = report::output(&pose, decision.as_ref(), &events, self.session.usage());
        self.previous = selected.map(|estimate| Previous {
            key,
            estimate: estimate.clone(),
        });
        Ok(output)
    }

    fn motion(&mut self, to: FrameKey, tracked: &Estimate) -> Result<(), Error> {
        let previous = self
            .previous
            .as_ref()
            .ok_or_else(|| Error::Input("tracked motion has no preceding selected pose".into()))?;
        // The new map fit is not in this transform. Its correction enters only as an anchor.
        let from_to = from_camera(&previous.estimate.pose).inv_mul(&from_camera(&tracked.pose));
        self.session.apply_motion(RelativeMotion {
            from: previous.key,
            to,
            from_to,
            scale: ScaleSource::SeededMapPoints {
                map: tracked.map.clone(),
            },
            from_observation_sha256: previous.estimate.observation_sha256.clone(),
            to_observation_sha256: tracked.observation_sha256.clone(),
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
