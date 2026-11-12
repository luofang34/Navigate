//! One match of a frame: the reference view at one belief and the pose
//! that the frame gives against it.

use std::sync::Arc;

use image::GrayImage;
use navigate_visual::{Estimate, Frame, FrameStamp, PosePrior, VisualError};

use super::super::acquire::Search;
use super::super::belief::Belief;
use super::super::map::TerrainMap;
use super::super::names;
use super::super::view::{NoView, Reference, Sight};
use super::{TerrainCamera, offset_bound_px, resampled};

/// The match of a frame at one belief.
pub(in super::super) struct Attempt {
    pub reference: Reference,
    /// The frame as the match took it, at the size of the working camera.
    pub frame: Frame,
    pub estimate: Result<Estimate, VisualError>,
}

impl TerrainCamera {
    /// The match of `image` against the reference view at `belief`, with
    /// the bounds of `search`.
    pub(in super::super) fn attempt(
        &mut self,
        map: &TerrainMap,
        search: &Search,
        belief: Belief,
        (image, stamp, lag_s): (&GrayImage, FrameStamp, f64),
    ) -> Result<Attempt, (NoView, Option<&'static str>)> {
        let sight = Sight {
            declared: &self.declared,
            mount: &self.mount,
            belief: &belief,
            radius_m: search.position_radius_m,
            ground_height_m: self.ground_height_m,
        };
        let reference = self.views.reference(map, &sight).map_err(|e| (e, None))?;
        let path = reference.path;
        let Ok(pose) = belief.camera_in(reference.view.frame, &self.mount) else {
            return Err((NoView::Unused(names::BELIEF_OUT_OF_RANGE), Some(path)));
        };
        let prior = PosePrior {
            pose,
            position_radius_m: search.position_radius_m,
            attitude_radius_rad: search.attitude_radius_rad,
        };
        self.working = Some(reference.working);
        let bound_px = offset_bound_px(&belief, (&map.config, lag_s), &prior, &reference.working);
        self.localizer.matcher_mut().set_offset_bound_px(bound_px);
        let frame = resampled(stamp, image.clone(), &reference.working.intrinsics);
        let estimate = self.matched(&frame, &reference, &prior);
        Ok(Attempt {
            reference,
            frame,
            estimate,
        })
    }

    /// The pose that `frame` gives against the view of `reference`.
    fn matched(
        &mut self,
        frame: &Frame,
        reference: &Reference,
        prior: &PosePrior,
    ) -> Result<Estimate, VisualError> {
        let view = &*reference.view;
        if !self.inspect {
            return self.localizer.estimate_blocking(frame, view, prior);
        }
        // Both calls run the same match; this one also returns its pairs.
        let (estimate, evidence) = self
            .localizer
            .estimate_with_evidence_blocking(frame, view, prior);
        let working = reference.working;
        let shown = Arc::clone(&reference.view);
        self.pending
            .compared(shown, frame.image.clone(), working, &evidence);
        self.pending.gate(prior, &estimate);
        estimate
    }
}
