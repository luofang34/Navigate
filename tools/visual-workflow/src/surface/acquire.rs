//! Coarse proposals and local crop search retain all examined geometry decisions.
mod cache;
pub(super) mod engine;
mod selection;
use super::{
    job::{SurfaceJob, window},
    report,
};
use crate::{
    Error, assets,
    config::{Config, Graph},
};
use engine::Engine;
use navigate_visual::{
    Frame, LocalizerConfig, OrthographicEstimate, OrthographicReference, PixelMatch, PosePrior,
    PoseVerifier,
};
use navigate_visual_workflow::FeatureBackend;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

struct Trial<'a> {
    frame: &'a Frame,
    reference: &'a OrthographicReference,
    prior: &'a PosePrior,
    verifier: PoseVerifier,
    record_matches: bool,
    config: &'a Config,
}
pub(super) struct Acquired {
    pub estimate: OrthographicEstimate,
    pub pairs: Vec<PixelMatch>,
    pub window: Option<[u32; 4]>,
}
pub(super) struct SearchResult {
    pub record: Value,
    pub selected: Option<Acquired>,
}
pub(super) fn validate(config: &Config, job: &SurfaceJob) -> Result<(), Error> {
    if job.reference.window.is_some()
        || job.reference_quarter_turns != [0, 1, 2, 3]
        || !matches!(config.graph, Graph::Sparse | Graph::DenseImage)
        || config.limits.candidates < 76
    {
        return Err(Error::Input("acquisition needs a full map, four ordered quarter turns, a candidate budget of at least 76, and sparse or dense-image features".into()));
    }
    Ok(())
}
pub(super) fn search_blocking(
    frame: &Frame,
    reference: &OrthographicReference,
    prior: &PosePrior,
    engine: &mut Engine,
    config: &Config,
    record_matches: bool,
) -> Result<SearchResult, Error> {
    selection::seed(&[], reference.image.dimensions())?;
    Trial {
        frame,
        reference,
        prior,
        verifier: PoseVerifier::new(LocalizerConfig::default())?,
        record_matches,
        config,
    }
    .run_blocking(engine)
}
pub fn run_blocking(
    config: &Config,
    root: &Path,
    input: &Path,
    output: &Path,
) -> Result<(), Error> {
    let total = Instant::now();
    let job: SurfaceJob = assets::read_json_blocking(input)?;
    validate(config, &job)?;
    let started = Instant::now();
    let (frame, reference, prior) =
        job.load_blocking(input.parent().unwrap_or(Path::new(".")), config)?;
    selection::seed(&[], reference.image.dimensions())?;
    let loading_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let backend = config.load_blocking(root)?;
    let model_load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let mut engine = Engine::new(backend)?;
    engine.query_blocking(&frame.image, config)?;
    let query_extraction_ms = started.elapsed().as_secs_f64() * 1000.0;
    let trial = Trial {
        frame: &frame,
        reference: &reference,
        prior: &prior,
        verifier: PoseVerifier::new(LocalizerConfig::default())?,
        record_matches: job.record_matches,
        config,
    };
    let preparation_ms = total.elapsed().as_secs_f64() * 1000.0;
    let mut result = trial.run_blocking(&mut engine)?.record;
    result["preparation_ms"] = json!(preparation_ms);
    result["elapsed_to_first_pass_ms"] = json!(
        result["search_to_first_pass_ms"]
            .as_f64()
            .map(|search| preparation_ms + search)
    );
    result["backend"] = json!(engine.backend.identity());
    result["job"] = json!(job);
    result["observation_sha256"] = json!(frame.evidence_sha256());
    result["map_input_sha256"] = json!(format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&job.reference).map_err(|source| Error::Json {
                path: input.into(),
                source
            })?
        )
    ));
    result["loading_ms"] = json!(loading_ms);
    result["model_load_ms"] = json!(model_load_ms);
    result["query_extraction_ms"] = json!(query_extraction_ms);
    result["query_encodings"] = json!(1);
    result["reference_encodings"] = json!(engine.reference_encodings);
    result["candidate_limit"] = json!(config.limits.candidates);
    result["search_candidate_bound"] = json!(76);
    result["peak_reference_accounted_bytes"] = json!(engine.peak_reference_bytes);
    result["query_feature_bytes"] = json!(engine.query_bytes);
    result["elapsed_before_output_ms"] = json!(total.elapsed().as_secs_f64() * 1000.0);
    result["scope"] = json!(
        "Bounded development search: coarse best match count, median crop, then at most nine adjacent 2048px crops. If all fail, examine nine 1024px crops inside the parent with most raw matches. Examine all quarter turns and all proposed crops. Select the first geometry pass by stage, crop order, then rotation; do not rank with truth. A pass is not a unique or independently accurate fix. One query encoding and model load; reference features are released per rotation. Full source raster stays resident. Accounted memory excludes models, runtime tensors, driver buffers, and file decoding peaks. Map input digest binds image, heights, geometry, release, and paths; it is not a portable cache key."
    );
    assets::write_json_blocking(output, &result)?;
    tracing::info!(path=%output.display(),"Saved bounded acquisition results");
    Ok(())
}
impl Trial<'_> {
    fn run_blocking(&self, engine: &mut Engine) -> Result<SearchResult, Error> {
        let started = Instant::now();
        let bytes = self.reference.image.as_raw().len() * 5;
        let mut selected = None;
        let (coarse, best) =
            self.stage_blocking(engine, self.reference, bytes, &mut selected, None)?;
        let seed = selection::seed(&best, self.reference.image.dimensions())?;
        let windows = seed
            .map(|s| selection::neighbors(s, self.reference.image.dimensions()))
            .unwrap_or_default();
        let coarse_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut first =
            first_pass(&coarse).map(|rotation| json!({"stage":"coarse","rotation":rotation}));
        let local = self.windows_blocking(engine, &windows, started, &mut selected)?;
        let mut first_fix_ms = first.as_ref().map(|_| coarse_ms);
        if first.is_none()
            && let Some((decision, time)) = window_pass(&local, "local")
        {
            first = Some(decision);
            first_fix_ms = Some(time);
        }
        let mut fine = Vec::new();
        let mut parent = None;
        if first.is_none() {
            let mut count = 0;
            for (area, row) in windows.iter().zip(&local) {
                let maximum = row["candidates"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|r| r["matches"].as_u64())
                    .max()
                    .unwrap_or(0);
                if maximum > count {
                    count = maximum;
                    parent = Some(*area);
                }
            }
            if let Some(area) = parent {
                fine = self.windows_blocking(
                    engine,
                    &selection::children(area),
                    started,
                    &mut selected,
                )?;
                if let Some((decision, time)) = window_pass(&fine, "fine") {
                    first = Some(decision);
                    first_fix_ms = Some(time);
                }
            }
        }
        Ok(SearchResult {
            selected,
            record: json!({"coarse":coarse,"coarse_ms":coarse_ms,"seed":seed,"windows":local,
            "fine_parent":parent,"fine_windows":fine,"first_pass":first,
            "accepted":first.is_some(),"search_to_first_pass_ms":first_fix_ms,"search_total_ms":started.elapsed().as_secs_f64()*1000.0,
            "first_pass_time_scope":"Search only, after file loading, model loading and query encoding. Includes all four directions through the first accepted window. Exhaustive diagnostic work continues after selection."}),
        })
    }
    fn windows_blocking(
        &self,
        engine: &mut Engine,
        windows: &[[u32; 4]],
        started: Instant,
        selected: &mut Option<Acquired>,
    ) -> Result<Vec<Value>, Error> {
        let mut crops = Vec::new();
        for &area in windows {
            let window_start = Instant::now();
            engine.prepare_raster(
                self.reference.image.as_raw().len() * 5 + area[2] as usize * area[3] as usize * 5,
                self.config,
            )?;
            let reference =
                window::crop_borrowed(self.reference, area, self.config.limits.reference_bytes)?;
            let crop_ms = window_start.elapsed().as_secs_f64() * 1000.0;
            let bytes = (self.reference.image.as_raw().len() + reference.image.as_raw().len()) * 5;
            let (rows, _) = self.stage_blocking(engine, &reference, bytes, selected, Some(area))?;
            crops.push(json!({"window":area,"crop_ms":crop_ms,"total_ms":window_start.elapsed().as_secs_f64()*1000.0,
                "search_elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"candidates":rows}));
        }
        Ok(crops)
    }
    fn stage_blocking(
        &self,
        engine: &mut Engine,
        reference: &OrthographicReference,
        bytes: usize,
        selected: &mut Option<Acquired>,
        window: Option<[u32; 4]>,
    ) -> Result<(Vec<Value>, Vec<PixelMatch>), Error> {
        let mut rows = Vec::new();
        let mut best = Vec::new();
        let key_start = Instant::now();
        let reference_input = engine.reference_input(&reference.image, window);
        let source_key_ms = key_start.elapsed().as_secs_f64() * 1000.0;
        for rotation in 0..4 {
            let prepared = engine.match_blocking(&reference_input, rotation, bytes, self.config)?;
            let candidate = prepared.candidate;
            if candidate.pairs.len() > best.len() {
                best = candidate.pairs.clone();
            }
            let (mut row, decision) = report::verify_with_estimate(
                &self.verifier,
                self.frame,
                reference,
                self.prior,
                &candidate,
                engine.backend.identity(),
                self.record_matches,
            );
            if selected.is_none()
                && let Ok(estimate) = decision
            {
                *selected = Some(Acquired {
                    estimate,
                    pairs: candidate.pairs,
                    window,
                });
            }
            row["reference_extraction_ms"] = json!(prepared.extraction_ms);
            row["reference_preparation_ms"] = json!(prepared.preparation_ms);
            row["reference_source_key_ms"] = json!(if rotation == 0 { source_key_ms } else { 0.0 });
            row["reference_cache_hit"] = json!(prepared.cache_hit);
            rows.push(row);
        }
        Ok((rows, best))
    }
}
fn first_pass(rows: &[Value]) -> Option<u64> {
    rows.iter()
        .find(|row| row["accepted"] == true)
        .and_then(|row| row["reference_quarter_turns"].as_u64())
}

fn window_pass(windows: &[Value], stage: &str) -> Option<(Value, f64)> {
    windows.iter().enumerate().find_map(|(rank, row)| {
        let rotation = first_pass(row["candidates"].as_array()?)?;
        Some((
            json!({"stage":stage,"window_index":rank,"rotation":rotation,"window":row["window"]}),
            row["search_elapsed_ms"].as_f64()?,
        ))
    })
}

#[cfg(test)]
mod tests;
