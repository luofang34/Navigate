//! Bounded map retrieval proposals do not enter the trajectory as anchors.

use super::{Processor, assets};
use crate::{Error, Observation, Pose};
use navigate_visual::{GroundCorrespondence, planar_proposal};
use std::path::Path;

impl Processor {
    /// Bounded reference-render candidates with evidence-supported attitude.
    /// No candidate is admitted as a navigation observation by this query.
    ///
    /// # Errors
    /// Returns an error for an inconsistent session frame identity.
    pub fn relocation_candidates(&self) -> Result<Vec<Pose>, Error> {
        let Some(previous) = &self.previous else {
            return Ok(vec![self.config.prior]);
        };
        let candidates = self
            .session
            .relocation_candidates(&super::key(previous.stamp.sequence), None)?;
        Ok(candidates
            .into_iter()
            .map(|candidate| candidate.pose.into())
            .collect())
    }

    /// Verify a candidate before committing this frame to a stream.
    ///
    /// # Errors
    /// Returns malformed-asset errors; unsupported geometry returns no pose.
    pub fn assess_blocking(&self, root: &Path, input: &Observation) -> Result<Option<Pose>, Error> {
        let Some(source) = &input.map_reference else {
            return Ok(None);
        };
        let reference = assets::reference_blocking(root, source, &self.config, self.map())?;
        let frame = navigate_visual::Frame {
            camera: self.config.camera.model(),
            stamp: navigate_visual::FrameStamp {
                sequence: input.sequence,
                capture_time_ns: input.capture_ns,
            },
            image: assets::image_blocking(root, &input.image)?,
        };
        Ok(self
            .verifier
            .verify(
                &frame,
                &reference,
                &self.prior()?,
                &assets::matches(&input.map_matches),
                &self.config.matcher,
            )
            .ok()
            .map(|estimate| estimate.pose.into()))
    }

    /// Propose a better reference-render pose from coarse map correspondences.
    /// This does not observe a frame or change the session.
    ///
    /// # Errors
    /// Returns invalid map-reference or asset errors.
    pub fn propose_blocking(
        &self,
        root: &Path,
        input: &Observation,
    ) -> Result<Option<Pose>, Error> {
        let Some(source) = &input.map_reference else {
            return Ok(None);
        };
        let reference = assets::reference_blocking(root, source, &self.config, self.map())?;
        let camera = self.config.camera.model();
        let mut pairs = Vec::new();
        for pair in &input.map_matches {
            let [x, y] = pair[0];
            if !x.is_finite()
                || !y.is_finite()
                || x < 0.0
                || y < 0.0
                || x >= f64::from(camera.width)
                || y >= f64::from(camera.height)
            {
                continue;
            }
            let index = (y as usize) * camera.width as usize + x as usize;
            let Some(&depth) = reference.depth_m.get(index) else {
                continue;
            };
            if !depth.is_finite() || depth <= 0.0 {
                continue;
            }
            pairs.push(GroundCorrespondence {
                world: camera.unproject(&reference.pose, pair[0].into(), f64::from(depth)),
                query: pair[1].into(),
            });
        }
        Ok(planar_proposal(&camera, &pairs)?.map(|proposal| proposal.pose.into()))
    }
}
