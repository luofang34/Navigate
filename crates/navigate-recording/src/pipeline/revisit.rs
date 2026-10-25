//! Revisit closures: an earlier keyframe seen again by a later frame.
//!
//! The host renders depth at the session estimate of the earlier keyframe
//! and matches the two images. Navigate verifies the geometry and decides
//! whether the closure enters the trajectory.

use super::{Processor, assets, key};
use crate::{Configuration, Error, Revisit, RevisitRequest};
use navigate_visual::{Frame, FrameStamp, MapRevision, PixelMatch, ReferenceView};
use navigate_visual_session::{RevisitEvidence, SessionError};
use std::path::Path;

/// Decoded revisit evidence for the preceding frame.
pub struct RevisitInput {
    /// Later frame index; it must be the preceding frame of the stream.
    pub later: u64,
    /// Earlier keyframe pixels and capture stamp.
    pub earlier: Frame,
    /// Map depth rendered at the session estimate of the earlier keyframe.
    pub earlier_surface: ReferenceView,
    /// Earlier (reference) to later (query) correspondences.
    pub matches: Vec<PixelMatch>,
}

pub(super) fn decode_blocking(
    root: &Path,
    record: &Revisit,
    config: &Configuration,
    map: MapRevision,
) -> Result<RevisitInput, Error> {
    Ok(RevisitInput {
        later: record.later,
        earlier: Frame {
            camera: config.camera.model(),
            stamp: FrameStamp {
                sequence: record.earlier,
                capture_time_ns: record.earlier_capture_ns,
            },
            image: assets::image_blocking(root, &record.earlier_image)?,
        },
        earlier_surface: assets::reference_blocking(root, &record.earlier_reference, config, map)?,
        matches: assets::matches(&record.matches),
    })
}

impl Processor {
    /// Earlier keyframes that the preceding frame may see again, best first.
    /// The query does not change the session.
    ///
    /// # Errors
    /// Returns session errors for an inconsistent frame identity.
    pub fn revisit_candidates(&self) -> Result<Vec<RevisitRequest>, Error> {
        let Some(previous) = &self.previous else {
            return Ok(Vec::new());
        };
        let later = key(previous.stamp.sequence);
        if self.session.pose_at(&later)?.odometry.is_none() {
            return Ok(Vec::new());
        }
        Ok(self
            .session
            .revisit_candidates(&later, None)?
            .into_iter()
            .map(|c| RevisitRequest {
                later: c.later.index,
                earlier: c.earlier.index,
                earlier_pose: c.earlier_pose.into(),
                later_pose: c.later_prior.pose.into(),
            })
            .collect())
    }

    /// Verify and submit revisit evidence for the preceding frame.
    pub(super) fn revisit(
        &mut self,
        inputs: &[RevisitInput],
        diagnostics: &mut Vec<String>,
    ) -> Result<(), Error> {
        let Some(previous) = &self.previous else {
            if !inputs.is_empty() {
                diagnostics.push("Revisit ignored: no preceding frame".into());
            }
            return Ok(());
        };
        for input in inputs {
            let label = format!(
                "Revisit of {} with {}",
                input.later, input.earlier.stamp.sequence
            );
            if input.later != previous.stamp.sequence {
                diagnostics.push(format!("{label} ignored: not the preceding frame"));
                continue;
            }
            let earlier = key(input.earlier.stamp.sequence);
            let candidates = self.session.revisit_candidates(&key(input.later), None)?;
            let Some(candidate) = candidates.iter().find(|c| c.earlier == earlier) else {
                diagnostics.push(format!("{label} ignored: not a candidate"));
                continue;
            };
            let evidence = RevisitEvidence {
                later: previous,
                earlier: &input.earlier,
                earlier_surface: &input.earlier_surface,
                matches: &input.matches,
                matcher_identity: &self.config.matcher,
            };
            let constraint = match self.session.verify_revisit(candidate, evidence) {
                Ok(constraint) => constraint,
                Err(SessionError::Visual { source, .. }) => {
                    diagnostics.push(format!("{label} rejected: {source}"));
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            let inliers = constraint.quality.inliers;
            let decision = self.session.submit_revisit(constraint)?;
            diagnostics.push(format!("{label}: {decision:?}; {inliers} inliers"));
        }
        Ok(())
    }
}
