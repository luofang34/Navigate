//! One query owns all candidate refinements and their common prior.
use crate::{
    BenchError, matches::VerifiedMatches, read_blocking, session::estimate_report,
    stream::write_record_blocking, trial::writer_blocking,
};
use navigate_visual::{
    CameraModel, CandidateDecision, CandidateId, CandidateResults, Frame, FrameStamp, ImageMatcher,
    LocalizerConfig, PosePrior, PoseVerifier, ReferenceView,
};
use std::{collections::BTreeMap, path::Path, time::Instant};
pub(crate) struct Observation {
    pub frame: Frame,
    prior: PosePrior,
    results: CandidateResults,
    reports: BTreeMap<u64, serde_json::Value>,
}
impl Observation {
    pub fn new_blocking(
        query: &Path,
        stamp: FrameStamp,
        camera: CameraModel,
        prior: PosePrior,
    ) -> Result<Self, BenchError> {
        let image = image::load_from_memory(&read_blocking(query)?)
            .map_err(|source| BenchError::Image {
                path: query.into(),
                source,
            })?
            .to_luma8();
        Self::from_frame(
            Frame {
                stamp,
                camera,
                image,
            },
            prior,
        )
    }
    pub fn from_frame(frame: Frame, prior: PosePrior) -> Result<Self, BenchError> {
        frame.camera.validate()?;
        prior.validate()?;
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height) {
            return Err(BenchError::Record {
                reason: "query dimensions do not match calibration".into(),
            });
        }
        let results = CandidateResults::new(&frame);
        Ok(Self {
            frame,
            prior,
            results,
            reports: BTreeMap::new(),
        })
    }
    pub fn invalidate(&mut self, id: u64) -> Result<(), BenchError> {
        self.results.record(
            CandidateId(id),
            Err(navigate_visual::VisualError::Invalid {
                field: "candidate refinement incomplete",
            }),
        )?;
        self.reports.insert(id, serde_json::json!({"candidate_id":id,"accepted":false,
            "acceptance_stage":"reference_or_matching", "reason":"Candidate refinement is incomplete.",
            "observation_sha256":self.frame.evidence_sha256()}));
        Ok(())
    }
    pub(super) fn refine_blocking(
        &mut self,
        input: Refinement<'_>,
    ) -> Result<serde_json::Value, BenchError> {
        self.invalidate(input.id)?;
        let mut matcher = VerifiedMatches::open_blocking(input.matches)?;
        matcher.validate_depth(input.reference)?;
        let (report, _) = self.match_candidate_blocking(
            input.id,
            input.reference,
            &mut matcher,
            input.map_context,
        )?;
        let mut writer = writer_blocking(input.output)?;
        write_record_blocking(&mut writer, input.output, &report)?;
        Ok(report)
    }
    pub fn match_candidate_blocking(
        &mut self,
        id: u64,
        reference: &ReferenceView,
        matcher: &mut dyn ImageMatcher,
        map_context: &serde_json::Value,
    ) -> Result<(serde_json::Value, Option<navigate_visual::RefinementSeed>), BenchError> {
        self.invalidate(id)?;
        let started = Instant::now();
        let pairs = matcher.match_images_blocking(&reference.image, &self.frame.image)?;
        let matching_ms = started.elapsed().as_secs_f64() * 1000.0;
        let geometry = Instant::now();
        let evaluation = PoseVerifier::new(LocalizerConfig::default())?.evaluate(
            &self.frame,
            reference,
            &self.prior,
            &pairs,
            matcher.identity(),
        );
        let geometry_ms = geometry.elapsed().as_secs_f64() * 1000.0;
        let result = evaluation.acceptance;
        let mut report = estimate_report(
            map_context,
            reference.frame,
            &self.frame,
            &result,
            started.elapsed().as_secs_f64() * 1000.0,
        )?;
        if let Some(seed) = &evaluation.refinement {
            let pose = seed.pose;
            report["refinement_pose"] = serde_json::json!({
                "position_enu_m": [pose.position.x, pose.position.y, pose.position.z],
                "eye_to_enu_xyzw": pose.orientation.coords.as_slice(),
                "inliers": seed.inliers, "spatial_support": seed.spatial_support,
                "query_cells": seed.query_cells, "reference_cells": seed.reference_cells,
                "status": "render seed only; not an accepted measurement"
            });
        }
        self.results.record(CandidateId(id), result)?;
        report["candidate_id"] = id.into();
        report["matching_ms"] = matching_ms.into();
        report["geometry_ms"] = geometry_ms.into();
        report["reference_image_sha256"] = crate::package::digest(reference.image.as_raw()).into();
        report["reference_depth_sha256"] = crate::matches::depth_digest(&reference.depth_m).into();
        let pose = reference.pose;
        report["reference_pose"] = serde_json::json!({"position_enu_m":pose.position.as_slice(),"eye_to_enu_xyzw":pose.orientation.coords.as_slice()});
        self.reports.insert(id, report.clone());
        Ok((report, evaluation.refinement))
    }
    pub fn select(&self) -> serde_json::Value {
        let mut result = match self.results.decision() {
            CandidateDecision::Unique(id) => self.reports.get(&id.0).cloned().unwrap_or_else(
                || serde_json::json!({"accepted":false,"reason":"candidate report is absent"}),
            ),
            CandidateDecision::Unresolved(ids) => {
                serde_json::json!({"accepted":false,"decision":"unresolved","reason":"Multiple geographic hypotheses passed geometry. No single visual fix is selected.","unresolved_candidate_ids":ids.iter().map(|id|id.0).collect::<Vec<_>>()})
            }
            CandidateDecision::Rejected => {
                serde_json::json!({"accepted":false,"decision":"rejected","reason":"No evaluated candidate passed geometry and prior bounds."})
            }
        };
        if result["accepted"].as_bool() == Some(true) {
            result["decision"] = "unique_among_evaluated".into();
        }
        result["sequence"] = self.frame.stamp.sequence.into();
        result["capture_time_ns"] = self.frame.stamp.capture_time_ns.into();
        result["observation_sha256"] = self.frame.evidence_sha256().into();
        result["acceptance_stage"] = "candidate_selection".into();
        result["geographic_accuracy"] = "not_independently_measured".into();
        result["evidence_correlation"] =
            "unknown; refinements share the query and reference data".into();
        result["candidate_hypotheses"] = self.reports.values().cloned().collect::<Vec<_>>().into();
        result
    }
}
pub(super) struct Refinement<'a> {
    pub id: u64,
    pub reference: &'a ReferenceView,
    pub matches: &'a Path,
    pub output: &'a Path,
    pub map_context: &'a serde_json::Value,
}

#[cfg(test)]
mod tests;
