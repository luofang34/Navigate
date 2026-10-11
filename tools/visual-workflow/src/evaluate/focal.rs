//! Conditional focal sensitivity from one fixed set of image matches.
use super::{Runner, metrics, position::PositionCheck};
use crate::Error;
use navigate_recording::Pose;
use navigate_visual::{CameraModel, Frame, PixelMatch, PosePrior, PoseVerifier, ReferenceView};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub(super) fn valid_scales(scales: &[f64]) -> bool {
    !scales.is_empty()
        && scales.len() <= 21
        && scales
            .iter()
            .all(|s| s.is_finite() && (0.8..=1.2).contains(s))
}

struct Probe<'a> {
    verifier: &'a PoseVerifier,
    reference: (&'a ReferenceView, &'a CameraModel),
    prior: PosePrior,
    identity: &'a str,
}
impl Probe<'_> {
    fn run(
        &self,
        frame: &Frame,
        pairs: &[PixelMatch],
        controls: &[PixelMatch],
        scales: &[f64],
        check: Option<&PositionCheck>,
    ) -> Value {
        let start = Instant::now();
        let mut query = Frame {
            camera: frame.camera,
            image: frame.image.clone(),
            stamp: frame.stamp,
        };
        let hypotheses: Vec<_> = scales
            .iter()
            .map(|scale| {
                query.camera = metrics::query_camera(frame.camera, *scale);
                let start = Instant::now();
                let result = self.verifier.verify_calibrated_reference(
                    &query,
                    self.reference.0,
                    self.reference.1,
                    &self.prior,
                    pairs,
                    self.identity,
                );
                let geometry_ms = start.elapsed().as_secs_f64() * 1000.0;
                let mut value =
                    json!({"scale":scale,"geometry_ms":geometry_ms,"accepted":result.is_ok()});
                if let Some(check) = check {
                    value["position_check"] =
                        check.compare(frame.stamp.capture_time_ns, result.as_ref().ok());
                }
                match result {
                    Ok(estimate) => {
                        value["pose"] = json!(Pose::from(estimate.pose));
                        value["geodetic"] = json!(estimate.frame.geodetic(estimate.pose.position));
                        value["spatial_support"] = json!(estimate.quality.spatial_support);
                        value["training_rms_px"] = json!(estimate.quality.reprojection_rms_px);
                        value["heldout"] =
                            metrics::score(controls, &query.camera, self.reference, &estimate.pose);
                    }
                    Err(error) => value["rejection"] = json!(error.to_string()),
                }
                value
            })
            .collect();
        json!({"hypotheses":hypotheses,"training_matches":pairs.len(),"training_sha256":digest(pairs),
            "heldout_matches":controls.len(),"heldout_sha256":digest(controls),
            "elapsed_ms":start.elapsed().as_secs_f64()*1000.0,
            "scope":"One fixed correspondence set; query focal lengths vary relative to the effective frame camera. Reference geometry and prior stay fixed. No hypothesis changes tracking, acceptance, or session state. Shared matcher, reference, and calibration errors remain. Probe time is validation time, excluded from pipeline_ms. No accuracy or confidence is inferred from the spread."})
    }
}

fn digest(pairs: &[PixelMatch]) -> String {
    let mut digest = Sha256::new();
    for pair in pairs {
        for value in [
            pair.reference.x,
            pair.reference.y,
            pair.query.x,
            pair.query.y,
        ] {
            digest.update(value.to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

impl Runner {
    pub(super) fn validation_blocking(
        &mut self,
        index: usize,
        frame: &Frame,
        matches: Option<Vec<PixelMatch>>,
    ) -> Result<(Vec<PixelMatch>, Option<Value>), Error> {
        let training = matches
            .as_ref()
            .filter(|_| self.experiment.focal_probe_scales.is_some())
            .map(|m| self.training(m));
        let withheld = self.controls_blocking(index, frame, matches)?;
        let probe = match (&self.experiment.focal_probe_scales, training) {
            (Some(scales), Some(pairs)) => Some(
                Probe {
                    verifier: &self.verifier,
                    reference: (&self.reference, &self.reference_camera),
                    prior: self.job.prior()?,
                    identity: self.matcher.identity(),
                }
                .run(
                    frame,
                    &pairs,
                    &withheld,
                    scales,
                    self.experiment.frames[index].position_check.as_ref(),
                ),
            ),
            _ => None,
        };
        Ok((withheld, probe))
    }
}

#[cfg(test)]
mod tests;
