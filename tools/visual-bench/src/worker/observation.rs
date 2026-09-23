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
pub(super) struct Observation {
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
        camera.validate()?;
        if image.dimensions() != (camera.width, camera.height) {
            return Err(BenchError::Record {
                reason: "query dimensions do not match calibration".into(),
            });
        }
        let frame = Frame {
            stamp,
            camera,
            image,
        };
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
    pub fn refine_blocking(
        &mut self,
        input: Refinement<'_>,
    ) -> Result<serde_json::Value, BenchError> {
        self.invalidate(input.id)?;
        let started = Instant::now();
        let mut matcher = VerifiedMatches::open_blocking(input.matches)?;
        matcher.validate_depth(input.reference)?;
        let pairs = matcher.match_images_blocking(&input.reference.image, &self.frame.image)?;
        let result = PoseVerifier::new(LocalizerConfig::default())?.verify(
            &self.frame,
            input.reference,
            &self.prior,
            &pairs,
            matcher.identity(),
        );
        let mut report = estimate_report(
            input.map_context,
            input.map_frame,
            &self.frame,
            &result,
            started.elapsed().as_secs_f64() * 1000.0,
        )?;
        self.results.record(CandidateId(input.id), result)?;
        report["candidate_id"] = input.id.into();
        report["reference_image_sha256"] =
            crate::package::digest(input.reference.image.as_raw()).into();
        report["reference_depth_sha256"] =
            crate::matches::depth_digest(&input.reference.depth_m).into();
        let pose = input.reference.pose;
        let q = pose.orientation.quaternion();
        report["reference_pose"] = serde_json::json!({"position_enu_m":[pose.position.x,pose.position.y,pose.position.z],"eye_to_enu_xyzw":[q.i,q.j,q.k,q.w]});
        self.reports.insert(input.id, report.clone());
        let mut writer = writer_blocking(input.output)?;
        write_record_blocking(&mut writer, input.output, &report)?;
        Ok(report)
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
    pub map_frame: navigate_visual::LocalFrame,
}

#[cfg(test)]
mod tests;
