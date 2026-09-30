//! Conditional frame tracking and separate map checks share one geometry verifier.
mod map_check;
mod relative;
use relative::{Check, relative};
mod surface;
use super::record::{PoseRecord, pose_report};
use crate::BenchError;
use map_check::map_check;
use navigate_visual::ReferenceRenderer;
use navigate_visual::{
    CameraPose, Frame, ImageMatcher, LocalizerConfig, PosePrior, PoseVerifier, ReferenceView,
    TrackingReference,
};
use serde_json::{Value, json};
use std::time::Instant;

struct Branch {
    id: usize,
    pose: CameraPose,
    previous: Option<Frame>,
    reference_pose: CameraPose,
    surface_tracks: Option<navigate_visual::SurfaceTracks>,
    anchor: Option<String>,
    last_map_ns: u64,
}
pub(super) struct Policy {
    pub dense_only: bool,
    pub map_interval_ns: u64,
    pub reanchor: bool,
    pub fixed_tilt: bool,
    pub surface_tracks: bool,
    pub keyframe_interval_ns: u64,
}
pub(super) struct Tracker {
    renderer: Box<dyn ReferenceRenderer<Error = BenchError>>,
    dense: Box<dyn ImageMatcher>,
    fast: Box<dyn navigate_visual::PointTracker>,
    verifier: PoseVerifier,
    prior: PosePrior,
    branches: Vec<Branch>,
    policy: Policy,
}
#[derive(Default)]
pub(super) struct Timing {
    pub render_ms: f64,
    pub matching_ms: f64,
    pub geometry_ms: f64,
    pub dense_runs: u32,
    pub fast_runs: u32,
}
impl Tracker {
    pub fn new(
        renderer: Box<dyn ReferenceRenderer<Error = BenchError>>,
        dense: Box<dyn ImageMatcher>,
        fast: Box<dyn navigate_visual::PointTracker>,
        prior: PosePrior,
        poses: Vec<CameraPose>,
        policy: Policy,
    ) -> Result<Self, BenchError> {
        if poses.is_empty() || poses.len() > 16 {
            return Err(BenchError::Record {
                reason: "supply one through sixteen candidates".into(),
            });
        }
        Ok(Self {
            renderer,
            dense,
            fast,
            verifier: PoseVerifier::new(LocalizerConfig::default())?,
            prior,
            branches: poses
                .into_iter()
                .enumerate()
                .map(|(id, pose)| Branch {
                    id,
                    pose,
                    previous: None,
                    reference_pose: pose,
                    surface_tracks: None,
                    anchor: None,
                    last_map_ns: 0,
                })
                .collect(),
            policy,
        })
    }
    pub fn observe_blocking(&mut self, frame: &Frame) -> Result<(Value, Timing), BenchError> {
        let mut timing = Timing::default();
        let mut reports = Vec::new();
        for branch in &mut self.branches {
            let mut reference = render(self.renderer.as_mut(), branch.reference_pose, &mut timing)?;
            let mut relative_result = (json!({"accepted":false,"tracking_supported":false}), None);
            if let Some(previous) = &branch.previous {
                let context = TrackingReference {
                    observation: previous,
                    surface: &reference,
                };
                relative_result = if self.policy.surface_tracks {
                    surface::relative(
                        &mut branch.surface_tracks,
                        self.fast.as_mut(),
                        &self.verifier,
                        frame,
                        context,
                        &self.prior,
                        &mut timing,
                    )?
                } else {
                    relative(
                        self.fast.as_mut(),
                        self.dense.as_mut(),
                        frame,
                        context,
                        Check {
                            verifier: &self.verifier,
                            prior: &self.prior,
                            dense_only: self.policy.dense_only,
                            fixed_tilt: self.policy.fixed_tilt,
                        },
                        &mut timing,
                    )?
                };
            }
            let (mut report, pose) = relative_result;
            if let Some(pose) = pose {
                branch.pose = pose;
                if self.policy.map_interval_ns > 0
                    && frame
                        .stamp
                        .capture_time_ns
                        .saturating_sub(branch.last_map_ns)
                        >= self.policy.map_interval_ns
                {
                    reference = render(self.renderer.as_mut(), branch.pose, &mut timing)?;
                    let (checked, pose) = map_check(
                        self.renderer.as_mut(),
                        self.dense.as_mut(),
                        &self.verifier,
                        frame,
                        reference,
                        &self.prior,
                        &mut timing,
                    )?;
                    branch.record_check(&mut report, checked, pose, frame, self.policy.reanchor);
                }
            } else {
                if branch.previous.is_some() {
                    reference = render(self.renderer.as_mut(), branch.pose, &mut timing)?;
                }
                let (checked, pose) = map_check(
                    self.renderer.as_mut(),
                    self.dense.as_mut(),
                    &self.verifier,
                    frame,
                    reference,
                    &self.prior,
                    &mut timing,
                )?;
                branch.recover(&mut report, checked, pose, frame);
            }
            branch.finish(&mut report, frame, self.policy.keyframe_interval_ns);
            reports.push(report);
        }
        Ok((
            json!({"candidate_hypotheses":reports,"geographic_accuracy":"not independently measured",
            "uncertainty":"unknown; pose, calibration, registration and surface errors are correlated",
            "map_checks":if self.policy.reanchor {"map-supported restarts; relative alternatives retained; no statistical fusion"} else {"diagnostic only; they do not overwrite a supported relative track"}}),
            timing,
        ))
    }
}
impl Branch {
    fn recover(
        &mut self,
        report: &mut Value,
        checked: Value,
        pose: Option<CameraPose>,
        frame: &Frame,
    ) {
        if let Some(pose) = pose {
            *report = checked;
            report["continuity_break"] = true.into();
            self.pose = pose;
            self.anchor = Some(frame.evidence_sha256());
            self.last_map_ns = frame.stamp.capture_time_ns;
        } else {
            report["recovery"] = checked;
        }
    }
    fn record_check(
        &mut self,
        report: &mut Value,
        checked: Value,
        pose: Option<CameraPose>,
        frame: &Frame,
        reanchor: bool,
    ) {
        self.last_map_ns = frame.stamp.capture_time_ns;
        if let Some(pose) = pose.filter(|_| reanchor) {
            let mut alternative = report.take();
            alternative["anchor_observation_sha256"] = json!(self.anchor);
            *report = checked;
            report["relative_alternative"] = alternative;
            report["continuity_break"] = true.into();
            report["anchor_policy"] =
                "restart at map hypothesis; no fusion or independence claim".into();
            self.pose = pose;
            self.anchor = Some(frame.evidence_sha256());
        } else {
            report["map_check"] = checked;
        }
    }

    fn finish(&mut self, report: &mut Value, frame: &Frame, interval_ns: u64) {
        if report["accepted"] == true {
            self.surface_tracks = None;
        }
        let refresh = self.previous.as_ref().is_none_or(|reference| {
            frame
                .stamp
                .capture_time_ns
                .saturating_sub(reference.stamp.capture_time_ns)
                >= interval_ns
        });
        let turned = self
            .pose
            .orientation
            .angle_to(&self.reference_pose.orientation)
            > 0.08;
        if report["accepted"] == true
            || (report["tracking_supported"] == true && (refresh || turned))
        {
            self.reference_pose = self.pose;
            self.previous = Some(Frame {
                camera: frame.camera,
                stamp: frame.stamp,
                image: frame.image.clone(),
            });
        }
        report["candidate_id"] = self.id.into();
        report["anchor_observation_sha256"] = json!(self.anchor);
        report["observation_sha256"] = frame.evidence_sha256().into();
    }
}

fn render(
    renderer: &mut dyn ReferenceRenderer<Error = BenchError>,
    pose: CameraPose,
    timing: &mut Timing,
) -> Result<ReferenceView, BenchError> {
    let start = Instant::now();
    let reference = renderer.render_blocking(pose)?;
    timing.render_ms += start.elapsed().as_secs_f64() * 1000.0;
    Ok(reference)
}

fn error_report(error: &navigate_visual::VisualError) -> Value {
    let mut causes = Vec::new();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        causes.push(cause.to_string());
        source = cause.source();
    }
    json!({"message":error.to_string(),"causes":causes})
}
fn provenance(report: &mut Value, reference: &ReferenceView) {
    report["reference_valid_depth_pixels"] = reference
        .depth_m
        .iter()
        .filter(|d| d.is_finite() && **d > 0.0)
        .count()
        .into();
    report["reference_pose"] = json!(PoseRecord::from(reference.pose));
    report["reference_image_sha256"] = crate::package::digest(reference.image.as_raw()).into();
    report["reference_depth_sha256"] = crate::matches::depth_digest(&reference.depth_m).into();
    report["map_manifest_sha256"] = reference.map.manifest_sha256.clone().into();
}

#[cfg(test)]
mod tests;
