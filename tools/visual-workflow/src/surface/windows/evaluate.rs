//! Every representation must pass the same orthographic geometry policy.
use super::{cache::Cache, geometry};
use crate::Error;
use navigate_visual::{Frame, OrthographicReference, PixelMatch, PosePrior, PoseVerifier};
use navigate_visual_onnx::{XFeatFeatures, XFeatStages};
use navigate_visual_workflow::FeatureBackend;
use serde_json::{Value, json};
use std::time::Instant;

pub(super) struct Context<'a> {
    pub frame: &'a Frame,
    pub reference: &'a OrthographicReference,
    pub prior: &'a PosePrior,
    pub verifier: &'a PoseVerifier,
    pub record_matches: bool,
}
pub(super) fn arm_blocking(
    backend: &mut XFeatStages,
    query: &XFeatFeatures,
    caches: &Cache,
    context: &Context<'_>,
    arm: usize,
) -> Result<Value, Error> {
    let started = Instant::now();
    let mut records = Vec::new();
    let mut matching_ms = 0.0;
    let mut matcher_calls = 0usize;
    for block in &caches.blocks {
        let mut whole = Vec::new();
        if arm == 2 {
            let start = Instant::now();
            whole = backend.match_features_blocking(&block.features, query)?;
            for pair in &mut whole {
                super::super::matching::restore(
                    pair,
                    block.rotation,
                    context.reference.image.dimensions(),
                );
            }
            matching_ms += start.elapsed().as_secs_f64() * 1000.0;
            matcher_calls = matcher_calls.wrapping_add(1);
        }
        for window in &block.windows {
            let start = Instant::now();
            let mut pairs = match arm {
                0 => backend.match_features_blocking(&window.crop, query)?,
                1 => backend.match_features_blocking(&window.shared, query)?,
                _ => whole
                    .iter()
                    .filter(|p| geometry::contains(p, window.area))
                    .map(|p| PixelMatch {
                        reference: p.reference,
                        query: p.query,
                    })
                    .collect(),
            };
            if arm < 2 {
                for pair in &mut pairs {
                    if arm == 0 {
                        geometry::restore_crop(pair, window.area, block.rotation);
                    } else {
                        super::super::matching::restore(
                            pair,
                            block.rotation,
                            context.reference.image.dimensions(),
                        );
                    }
                }
                matcher_calls = matcher_calls.wrapping_add(1);
            }
            let candidate_ms = start.elapsed().as_secs_f64() * 1000.0;
            matching_ms += candidate_ms;
            let count = if arm == 0 {
                window.crop.len()
            } else {
                window.shared.len()
            };
            records.push(context.record(
                block.rotation,
                pairs,
                candidate_ms,
                backend.identity(),
                window.area,
                count,
            ));
        }
    }
    Ok(
        json!({"representation":(["independent_crops","shared_block_subsets","shared_block_match_once"][arm]),
        "matching_and_filtering_ms":matching_ms,"matcher_calls":matcher_calls,
        "total_ms":started.elapsed().as_secs_f64()*1000.0,"candidates":records}),
    )
}

impl Context<'_> {
    fn record(
        &self,
        rotation: u8,
        pairs: Vec<PixelMatch>,
        matching_ms: f64,
        identity: &str,
        window: [u32; 4],
        count: usize,
    ) -> Value {
        let mut report = super::super::report::verify(
            self.verifier,
            self.frame,
            self.reference,
            self.prior,
            super::super::matching::Matches {
                rotation,
                pairs,
                matching_ms,
            },
            identity,
            self.record_matches,
        );
        report["window"] = json!(window);
        report["reference_features_in_window"] = json!(count);
        report
    }
}
