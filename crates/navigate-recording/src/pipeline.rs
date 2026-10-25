//! A blocking host adapter; Navigate owns geometry, drift and anchor admission.

mod acquisition;
mod assets;
mod ground;
mod result;
mod revisit;

pub use revisit::RevisitInput;

use crate::{Configuration, Error, Observation, ResultFrame};
use navigate_contract::ClockDomainId;
use navigate_visual::{Frame, FrameStamp, LocalFrame, MapRevision, PosePrior, PoseVerifier};
use navigate_visual_session::{
    AnchorObservation, Calibration, CameraMount, CaptureTime, ContinuityEpoch, FrameKey,
    FrameRecord, LensModel, MapReliability, MediaTime, RegionChange, RelativeMotion, SessionConfig,
    StreamId, SurfaceModel, VerticalReference, VisualSession,
};
use std::{collections::BTreeMap, path::Path};

/// Decoded evidence from a live camera or a replay adapter.
pub struct VisualObservation {
    /// Exact image pixels, intrinsics and acquisition stamp.
    pub frame: Frame,
    /// Source-media presentation timestamp, in nanoseconds.
    pub media_ns: u64,
    /// Terrain elevation below the camera in the selected map datum.
    pub terrain_m: f64,
    /// Independent map image and surface for geographic anchoring.
    pub map_reference: Option<navigate_visual::ReferenceView>,
    /// Map-reference to current-image correspondences.
    pub map_matches: Vec<navigate_visual::PixelMatch>,
    /// Map depth rendered at the preceding accepted camera pose.
    pub tracking_reference: Option<navigate_visual::ReferenceView>,
    /// Previous-image to current-image correspondences.
    pub tracking_matches: Vec<navigate_visual::PixelMatch>,
    /// Revisit evidence for the preceding frame, checked before this frame.
    pub revisits: Vec<RevisitInput>,
}

/// Executes recorded evidence through a deterministic Navigate session.
/// Media decode and feature matching are host responsibilities.
pub struct Processor {
    config: Configuration,
    session: VisualSession,
    verifier: PoseVerifier,
    previous: Option<Frame>,
    /// Kept for the frames the session still retains, and dropped with them.
    records: BTreeMap<u64, Saved>,
    /// Frames the session has observed, oldest first, as many as it retains.
    observed: std::collections::VecDeque<u64>,
    retained: usize,
}

struct Saved {
    media_ns: u64,
    digest: String,
    terrain_m: f64,
    diagnostics: Vec<String>,
    map_observation_accepted: bool,
}

impl Processor {
    /// Create a processor for one camera source.
    ///
    /// # Errors
    /// Rejects invalid intrinsics, map identity, prior or uncertainty declarations.
    pub fn new(config: Configuration) -> Result<Self, Error> {
        Self::with_session(config, SessionConfig::standard())
    }

    fn with_session(config: Configuration, session: SessionConfig) -> Result<Self, Error> {
        config.prior.camera()?;
        LocalFrame::anchor_mercator(config.anchor[0], config.anchor[1])?;
        if config.map_sha256.len() != 64
            || !config.map_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || config.map_release.is_empty()
            || config.matcher.is_empty()
            || !config.position_radius_m.is_finite()
            || config.position_radius_m <= 0.0
            || !config.map_sigma_m.iter().all(|v| v.is_finite() && *v > 0.0)
            || !config.object_height_m.is_finite()
            || config.object_height_m < 0.0
        {
            return Err(Error::Invalid(
                "invalid map, matcher, prior or error budget".into(),
            ));
        }
        let calibration = Calibration {
            intrinsics: config.camera.model(),
            lens: LensModel::Unknown,
            mount: CameraMount::Unknown,
            map_vertical: VerticalReference::Unknown,
        };
        Ok(Self {
            config,
            session: VisualSession::new(session, calibration)?,
            verifier: PoseVerifier::new(Default::default())?,
            previous: None,
            records: BTreeMap::new(),
            observed: std::collections::VecDeque::new(),
            retained: session.limits.max_frames,
        })
    }

    /// Process one timestamped observation. Rejected visual matches remain gaps.
    ///
    /// # Errors
    /// Returns asset, ordering or session-contract errors. Geometry rejection is recorded.
    pub fn process_blocking(
        &mut self,
        root: &Path,
        observation: Observation,
    ) -> Result<ResultFrame, Error> {
        let frame = Frame {
            camera: self.config.camera.model(),
            stamp: FrameStamp {
                sequence: observation.sequence,
                capture_time_ns: observation.capture_ns,
            },
            image: assets::image_blocking(root, &observation.image)?,
        };
        let read = |source: &crate::Reference| {
            assets::reference_blocking(root, source, &self.config, self.map())
        };
        let input = VisualObservation {
            frame,
            media_ns: observation.media_ns,
            terrain_m: observation.terrain_m,
            map_reference: observation.map_reference.as_ref().map(read).transpose()?,
            map_matches: assets::matches(&observation.map_matches),
            tracking_reference: observation
                .tracking_reference
                .as_ref()
                .map(read)
                .transpose()?,
            tracking_matches: assets::matches(&observation.tracking_matches),
            revisits: observation
                .revisits
                .iter()
                .map(|r| revisit::decode_blocking(root, r, &self.config, self.map()))
                .collect::<Result<_, _>>()?,
        };
        self.process(input)
    }

    /// Ingest decoded live or recorded evidence through the same state machine.
    /// This method performs no file, network or clock IO.
    ///
    /// # Errors
    /// Returns invalid calibration, ordering or session-contract errors.
    pub fn process(&mut self, input: VisualObservation) -> Result<ResultFrame, Error> {
        let frame = &input.frame;
        if !input.terrain_m.is_finite() || input.media_ns > i64::MAX as u64 {
            return Err(Error::Invalid("invalid terrain or media time".into()));
        }
        if frame.camera != self.config.camera.model()
            || frame.image.dimensions() != (frame.camera.width, frame.camera.height)
        {
            return Err(Error::Invalid(
                "observation dimensions disagree with calibration".into(),
            ));
        }
        let sequence = frame.stamp.sequence;
        let record = FrameRecord::from_frame(
            key(sequence),
            Some(MediaTime {
                pts: input.media_ns as i64,
                timebase_num: 1,
                timebase_den: 1_000_000_000,
            }),
            CaptureTime {
                clock: ClockDomainId::new(1),
                at_ns: frame.stamp.capture_time_ns,
                error_bound_ns: self.config.capture_error_ns,
            },
            frame,
        );
        let digest = record.observation_sha256.clone();
        let mut diagnostics = Vec::new();
        // Revisits refer to the preceding frame: check them in the session
        // state that the host queried, before this frame changes it.
        self.revisit(&input.revisits, &mut diagnostics)?;
        self.session.observe_frame(record)?;
        // The session drops its oldest frame record first, even one whose processing failed
        // after it was observed; the records here follow it.
        self.observed.push_back(sequence);
        while self.observed.len() > self.retained {
            if let Some(dropped) = self.observed.pop_front() {
                self.records.remove(&dropped);
            }
        }
        self.track(&input, &mut diagnostics)?;
        self.ground(&input, &mut diagnostics)?;
        let map_observation_accepted = self.anchor(&input, &mut diagnostics)?;
        let events = self.session.drain_events();
        if events.full_revision_required {
            return Err(Error::Invalid("Navigate event queue overflowed".into()));
        }
        self.records.insert(
            sequence,
            Saved {
                media_ns: input.media_ns,
                digest,
                terrain_m: input.terrain_m,
                diagnostics,
                map_observation_accepted,
            },
        );
        self.previous = Some(input.frame);
        self.result(sequence)
    }

    fn prior(&self) -> Result<PosePrior, Error> {
        Ok(PosePrior {
            pose: self.config.prior.camera()?,
            position_radius_m: self.config.position_radius_m,
            attitude_radius_rad: std::f64::consts::PI,
        })
    }

    fn map(&self) -> MapRevision {
        MapRevision {
            release_id: self.config.map_release.clone(),
            manifest_sha256: self.config.map_sha256.clone(),
        }
    }

    fn track(
        &mut self,
        input: &VisualObservation,
        diagnostics: &mut Vec<String>,
    ) -> Result<(), Error> {
        let (Some(previous), Some(reference)) = (&self.previous, &input.tracking_reference) else {
            return Ok(());
        };
        let surface = reference;
        let frame = &input.frame;
        let matches = &input.tracking_matches;
        let proposal = self.verifier.track(
            frame,
            navigate_visual::TrackingReference {
                observation: previous,
                surface,
            },
            &self.prior()?,
            matches,
            &self.config.matcher,
        );
        match proposal {
            Ok(proposal) => {
                let motion = RelativeMotion::from_tracking(
                    key(previous.stamp.sequence),
                    key(input.frame.stamp.sequence),
                    &surface.pose,
                    &proposal,
                );
                self.session.apply_motion(motion)?;
                diagnostics.push(format!(
                    "Relative tracking: {} inliers",
                    proposal.quality.inliers
                ));
            }
            Err(error) => diagnostics.push(format!("Tracking rejected: {error}")),
        }
        Ok(())
    }

    fn anchor(
        &mut self,
        input: &VisualObservation,
        diagnostics: &mut Vec<String>,
    ) -> Result<bool, Error> {
        let Some(reference) = &input.map_reference else {
            return Ok(false);
        };
        let frame = &input.frame;
        let matches = &input.map_matches;
        match self.verifier.verify(
            frame,
            reference,
            &self.prior()?,
            matches,
            &self.config.matcher,
        ) {
            Ok(estimate) => {
                let reliability = MapReliability {
                    horizontal_m: self.config.map_sigma_m[0],
                    vertical_m: self.config.map_sigma_m[1],
                    imagery_age_years: None,
                    unknown_age_years: 5.0,
                    age_growth_m_per_year: 1.0,
                    surface: SurfaceModel::BareEarth {
                        max_object_height_m: self.config.object_height_m,
                    },
                    change: RegionChange::Unknown,
                };
                let decision = self
                    .session
                    .submit_anchor(AnchorObservation::from_estimate(
                        key(input.frame.stamp.sequence),
                        &estimate,
                        reliability,
                    ))?;
                diagnostics.push(format!(
                    "Map anchor: {decision:?}; {} inliers",
                    estimate.quality.inliers
                ));
                return Ok(matches!(
                    decision,
                    navigate_visual_session::AnchorDecision::Accepted { .. }
                        | navigate_visual_session::AnchorDecision::Relocated { .. }
                ));
            }
            Err(error) => diagnostics.push(format!("Map match rejected: {error}")),
        }
        Ok(false)
    }
}

fn key(sequence: u64) -> FrameKey {
    FrameKey {
        stream: StreamId(0),
        continuity: ContinuityEpoch(0),
        index: sequence,
    }
}

#[cfg(test)]
mod tests;
