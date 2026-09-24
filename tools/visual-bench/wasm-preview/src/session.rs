use crate::{
    error::PreviewError,
    model::{Camera, Pose},
    preview::Preview,
};
use image::GrayImage;
use nalgebra::Vector2;
use navigate_visual::{
    CandidateDecision, CandidateId, CandidateResults, Frame, FrameStamp, LocalizerConfig,
    PixelMatch, PosePrior, PoseVerifier, ReferenceView, VisualError,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use wasm_bindgen::prelude::*;
#[derive(Deserialize)]
struct Prior {
    pose: Pose,
    position_radius_m: f64,
    attitude_radius_rad: f64,
}
#[derive(Deserialize)]
struct Pair {
    reference: [f64; 2],
    query: [f64; 2],
}
pub(crate) struct Session {
    frame: Frame,
    prior: PosePrior,
    results: CandidateResults,
    reports: BTreeMap<u32, Value>,
    active: Option<u32>,
}
impl Session {
    fn new(
        camera: Camera,
        pixels: Vec<u8>,
        prior: Prior,
        sequence: u32,
        capture_time_ns: f64,
    ) -> Result<Self, PreviewError> {
        if !capture_time_ns.is_finite()
            || !(0.0..=9_007_199_254_740_991.0).contains(&capture_time_ns)
        {
            return Err(PreviewError::Input {
                reason: "invalid capture time".into(),
            });
        }
        let frame = Frame {
            camera: camera.model(),
            stamp: FrameStamp {
                sequence: u64::from(sequence),
                capture_time_ns: capture_time_ns as u64,
            },
            image: GrayImage::from_raw(camera.width, camera.height, pixels).ok_or_else(|| {
                PreviewError::Input {
                    reason: "query dimensions do not match calibration".into(),
                }
            })?,
        };
        let prior = PosePrior {
            pose: prior.pose.model()?,
            position_radius_m: prior.position_radius_m,
            attitude_radius_rad: prior.attitude_radius_rad,
        };
        prior.validate()?;
        let results = CandidateResults::new(&frame);
        Ok(Self {
            frame,
            prior,
            results,
            reports: BTreeMap::new(),
            active: None,
        })
    }
    pub fn invalidate(&mut self, id: u32) -> Result<(), PreviewError> {
        self.active = Some(id);
        self.results.record(
            CandidateId(u64::from(id)),
            Err(VisualError::Invalid {
                field: "candidate refinement incomplete",
            }),
        )?;
        self.reports.insert(id,json!({"candidate_id":id,"accepted":false,"reason":"Candidate refinement is incomplete."}));
        Ok(())
    }
    fn refine(
        &mut self,
        id: u32,
        reference: &ReferenceView,
        pairs: &[PixelMatch],
        backend: &str,
    ) -> Result<Value, PreviewError> {
        if self.active != Some(id) {
            return Err(PreviewError::Input {
                reason: "reference candidate identity does not match".into(),
            });
        }
        self.invalidate(id)?;
        let evaluation = PoseVerifier::new(LocalizerConfig::default())?.evaluate(
            &self.frame,
            reference,
            &self.prior,
            pairs,
            backend,
        );
        let result = evaluation.acceptance;
        let mut report = match &result {
            Ok(e) => {
                let p = e.pose.position;
                let q = e.pose.orientation.quaternion();
                let [latitude, longitude, _] = e.frame.geodetic(p);
                json!({"candidate_id":id,"accepted":true,"acceptance_stage":"geometry_and_prior","position_enu_m":[p.x,p.y,p.z],"eye_to_enu_xyzw":[q.i,q.j,q.k,q.w],"latitude_deg":latitude,"longitude_deg":longitude,"altitude_m":p.z,"inliers":e.quality.inliers,"depth_matches":e.quality.depth_matches,"reprojection_rms_px":e.quality.reprojection_rms_px,"occupied_cells":e.quality.occupied_cells,"condition_number":e.quality.condition_number,"geometry_covariance":(0..6).map(|row|(0..6).map(|col|e.geometry_covariance[(row,col)]).collect::<Vec<_>>()).collect::<Vec<_>>(),"geometry_covariance_axes":["east_m","north_m","up_m","camera_rx_rad","camera_ry_rad","camera_rz_rad"],"covariance_scope":"local image geometry only; excludes map, calibration and association errors","backend":e.backend,"observation_sha256":e.observation_sha256,"map_release_id":e.map.release_id,"map_manifest_sha256":e.map.manifest_sha256})
            }
            Err(e) => {
                json!({"candidate_id":id,"accepted":false,"acceptance_stage":"geometry_and_prior","reason":e.to_string()})
            }
        };
        if result.is_err()
            && let Some(pose) = evaluation.refinement
        {
            let p = pose.position;
            let q = pose.orientation.quaternion();
            report["refinement_proposal"] = json!({"position_enu_m":[p.x,p.y,p.z],"eye_to_enu_xyzw":[q.i,q.j,q.k,q.w],"stage":"render_initialization_only","accepted":false});
        }
        reference_provenance(&mut report, reference);
        self.results.record(CandidateId(u64::from(id)), result)?;
        self.reports.insert(id, report.clone());
        Ok(report)
    }
    fn select(&self) -> Value {
        let mut value = match self.results.decision() {
            CandidateDecision::Unique(id) => self.reports[&(id.0 as u32)].clone(),
            CandidateDecision::Unresolved(ids) => {
                json!({"accepted":false,"decision":"unresolved","unresolved_candidate_ids":ids.iter().map(|id|id.0).collect::<Vec<_>>(),"reason":"Multiple geographic hypotheses passed geometry."})
            }
            CandidateDecision::Rejected => {
                json!({"accepted":false,"decision":"rejected","reason":"No evaluated candidate passed geometry and prior bounds."})
            }
        };
        if value["accepted"] == true {
            value["decision"] = "unique_among_evaluated".into();
        }
        value["candidate_hypotheses"] = self.reports.values().cloned().collect::<Vec<_>>().into();
        value["observation_sha256"] = self.frame.evidence_sha256().into();
        value["sequence"] = self.frame.stamp.sequence.into();
        value["capture_time_ns"] = self.frame.stamp.capture_time_ns.into();
        value["geographic_accuracy"] = "not_independently_measured".into();
        value["evidence_correlation"] = "unknown; refinements share image and map evidence".into();
        value
    }
}
fn reference_provenance(report: &mut Value, reference: &ReferenceView) {
    use sha2::{Digest, Sha256};
    report["reference_image_sha256"] =
        format!("{:x}", Sha256::digest(reference.image.as_raw())).into();
    let mut digest = Sha256::new();
    for depth in &reference.depth_m {
        digest.update(depth.to_le_bytes());
    }
    report["reference_depth_sha256"] = format!("{:x}", digest.finalize()).into();
    let p = reference.pose.position;
    let q = reference.pose.orientation.quaternion();
    report["reference_pose"] =
        json!({"position_enu_m":[p.x,p.y,p.z],"eye_to_enu_xyzw":[q.i,q.j,q.k,q.w]});
    report["surface_geometry"] = "rendered terrain; not independently verified".into();
}
mod tracking;
#[wasm_bindgen]
impl Preview {
    /// Bind one calibrated observation and its navigation prior to candidate results.
    pub fn begin(
        &mut self,
        pixels: Vec<u8>,
        prior_json: String,
        sequence: u32,
        capture_time_ns: f64,
    ) -> Result<(), JsValue> {
        self.reference = None;
        self.camera_session = None;
        self.camera_session = Some(Session::new(
            self.camera,
            pixels,
            serde_json::from_str(&prior_json).map_err(PreviewError::from)?,
            sequence,
            capture_time_ns,
        )?);
        Ok(())
    }
    /// Verify backend-independent pixel correspondences against rendered surface depth.
    pub fn refine(
        &mut self,
        id: u32,
        pairs_json: String,
        backend: String,
    ) -> Result<String, JsValue> {
        let session = self
            .camera_session
            .as_mut()
            .ok_or_else(|| PreviewError::Input {
                reason: "no active observation".into(),
            })?;
        if session.active != Some(id) {
            return Err(PreviewError::Input {
                reason: "reference candidate identity does not match".into(),
            }
            .into());
        }
        session.invalidate(id)?;
        let pairs: Vec<Pair> = serde_json::from_str(&pairs_json).map_err(PreviewError::from)?;
        if pairs.len() > 4096 {
            return Err(PreviewError::Input {
                reason: "too many pixel correspondences".into(),
            }
            .into());
        }
        let pairs: Vec<_> = pairs
            .into_iter()
            .map(|p| PixelMatch {
                reference: Vector2::from(p.reference),
                query: Vector2::from(p.query),
            })
            .collect();
        let reference = self.reference.as_ref().ok_or_else(|| PreviewError::Input {
            reason: "no rendered reference".into(),
        })?;
        Ok(session.refine(id, reference, &pairs, &backend)?.to_string())
    }
    /// Preserve all evaluated alternatives without adding confidence across refinements.
    pub fn select(&self) -> Result<String, JsValue> {
        Ok(self
            .camera_session
            .as_ref()
            .ok_or_else(|| PreviewError::Input {
                reason: "no active observation".into(),
            })?
            .select()
            .to_string())
    }
}

#[cfg(test)]
mod tests;
