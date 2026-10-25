//! Sparse evidence orders bounded dense fallback work without changing geometry.
use super::{job::SurfaceJob, matching::Matcher, report};
use crate::{
    Error, assets,
    config::{Config, Graph, Runtime},
};
use navigate_visual::{Frame, LocalizerConfig, OrthographicReference, PosePrior, PoseVerifier};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

struct Trial<'a> {
    frame: &'a Frame,
    reference: &'a OrthographicReference,
    prior: &'a PosePrior,
    settings: &'a SurfaceJob,
    verifier: PoseVerifier,
}
struct Engines<'a> {
    runtime: Runtime,
    primary: Matcher,
    fallback: Option<Matcher>,
    primary_config: &'a Config,
    fallback_config: &'a Config,
    fallback_root: &'a Path,
    raster_bytes: usize,
}
pub fn run_blocking(
    primary_path: &Path,
    fallback_path: &Path,
    input: &Path,
    output: &Path,
    iterations: usize,
) -> Result<(), Error> {
    let total = Instant::now();
    if !(1..=1000).contains(&iterations) {
        return Err(Error::Input("iterations must be 1 through 1000".into()));
    }
    let config: Config = assets::read_json_blocking(primary_path)?;
    let fallback_config: Config = assets::read_json_blocking(fallback_path)?;
    if matches!(config.graph, Graph::Loftr | Graph::LoftrSplit)
        || !matches!(fallback_config.graph, Graph::Loftr | Graph::LoftrSplit)
    {
        return Err(Error::Input(
            "cascade requires a feature primary and pair fallback".into(),
        ));
    }
    let root = primary_path.parent().unwrap_or(Path::new("."));
    let fallback_root = fallback_path.parent().unwrap_or(Path::new("."));
    let settings: SurfaceJob = assets::read_json_blocking(input)?;
    let (frame, reference, prior) =
        settings.load_blocking(input.parent().unwrap_or(Path::new(".")), &config)?;
    validate_pixels(&frame, &reference, fallback_config.limits.image_pixels)?;
    let raster_bytes = reference.image.as_raw().len() + reference.heights_m.len() * 4;
    let start = Instant::now();
    let runtime = Runtime::new_blocking(&config, root)?;
    let backend = runtime.load_blocking(&config, root)?;
    let primary = Matcher::from_backend_blocking(
        &config,
        &reference.image,
        &settings.reference_quarter_turns,
        raster_bytes,
        backend,
    )?;
    let setup_ms = start.elapsed().as_secs_f64() * 1000.0;
    let trial = Trial {
        frame: &frame,
        reference: &reference,
        prior: &prior,
        settings: &settings,
        verifier: PoseVerifier::new(LocalizerConfig::default())?,
    };
    let mut engines = Engines {
        runtime,
        primary,
        fallback: None,
        primary_config: &config,
        fallback_config: &fallback_config,
        fallback_root,
        raster_bytes,
    };
    let rows = (0..iterations)
        .map(|_| trial.run_blocking(&mut engines))
        .collect::<Result<Vec<_>, _>>()?;
    let elapsed = total.elapsed().as_secs_f64() * 1000.0;
    assets::write_json_blocking(
        output,
        &json!({"primary_backend":engines.primary.identity(),"fallback_backend":engines.fallback.as_ref().map(|m|m.identity()),"primary_setup_ms":setup_ms,"elapsed_before_output_ms":elapsed,"iterations":rows,"observation_sha256":frame.evidence_sha256(),"scope":"Experimental stopping policy. Primary verifies all directions. On rejection, match count orders fallback; evaluate the first two directions, then all remaining only if neither passes. Unexamined directions do not establish uniqueness or equal reliability to exhaustive search. Load times and model residency are included separately."}),
    )?;
    tracing::info!(path=%output.display(),"Saved routed surface trial");
    Ok(())
}
impl Trial<'_> {
    fn run_blocking(&self, engines: &mut Engines<'_>) -> Result<Value, Error> {
        let started = Instant::now();
        let (extraction_ms, matches) = engines
            .primary
            .matches_blocking(&self.frame.image, engines.primary_config)?;
        let mut primary = Vec::new();
        for candidate in matches {
            primary.push(report::verify(
                &self.verifier,
                self.frame,
                self.reference,
                self.prior,
                candidate,
                engines.primary.identity(),
                self.settings.record_matches,
            ));
        }
        let order = rank(&primary);
        let primary_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut fallback = Vec::new();
        let mut load_ms = 0.0;
        if !accepted(&primary) {
            load_ms = engines.load_fallback_blocking(self)?;
            let matcher = engines
                .fallback
                .as_mut()
                .ok_or_else(|| Error::Input("fallback initialization failed".into()))?;
            fallback = routed_blocking(&order, |rotation| {
                let candidate = matcher.match_pair_blocking(rotation, &self.frame.image)?;
                Ok(report::verify(
                    &self.verifier,
                    self.frame,
                    self.reference,
                    self.prior,
                    candidate,
                    matcher.identity(),
                    self.settings.record_matches,
                ))
            })?;
        }
        let skipped: Vec<_> = order
            .iter()
            .copied()
            .filter(|r| !fallback.iter().any(|v| v["reference_quarter_turns"] == *r))
            .collect();
        Ok(
            json!({"accepted":accepted(&primary)||accepted(&fallback),"primary":primary,"fallback":fallback,"fallback_order":order,"unexamined_fallback_directions":skipped,"query_extraction_ms":extraction_ms,"primary_ms":primary_ms,"fallback_load_ms":load_ms,"total_ms":started.elapsed().as_secs_f64()*1000.0,"reference_budget_accounted_bytes":engines.primary.resident_bytes+engines.fallback.as_ref().map_or(0,|m|m.resident_bytes)}),
        )
    }
}
impl Engines<'_> {
    fn load_fallback_blocking(&mut self, trial: &Trial<'_>) -> Result<f64, Error> {
        if self.fallback.is_some() {
            return Ok(0.0);
        }
        let started = Instant::now();
        let remaining = self
            .primary_config
            .limits
            .reference_bytes
            .saturating_sub(self.primary.resident_bytes);
        let required = self.raster_bytes.saturating_add(
            trial
                .reference
                .image
                .as_raw()
                .len()
                .saturating_mul(trial.settings.reference_quarter_turns.len()),
        );
        if required > remaining {
            return Err(Error::Input("combined cascade reference budget".into()));
        }
        let backend = self
            .runtime
            .load_blocking(self.fallback_config, self.fallback_root)?;
        self.fallback = Some(Matcher::from_backend_blocking(
            self.fallback_config,
            &trial.reference.image,
            &trial.settings.reference_quarter_turns,
            self.raster_bytes,
            backend,
        )?);
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }
}
fn validate_pixels(
    frame: &Frame,
    reference: &OrthographicReference,
    limit: usize,
) -> Result<(), Error> {
    if frame.image.as_raw().len() > limit || reference.image.as_raw().len() > limit {
        return Err(Error::Input("fallback image pixel budget".into()));
    }
    Ok(())
}
fn accepted(rows: &[Value]) -> bool {
    rows.iter().any(|r| r["accepted"] == true)
}
fn rank(rows: &[Value]) -> Vec<u8> {
    let mut items: Vec<_> = rows
        .iter()
        .filter_map(|r| {
            Some((
                r["reference_quarter_turns"].as_u64()? as u8,
                r["matches"].as_u64()?,
            ))
        })
        .collect();
    items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    items.into_iter().map(|v| v.0).collect()
}

fn routed_blocking(
    order: &[u8],
    mut evaluate: impl FnMut(u8) -> Result<Value, Error>,
) -> Result<Vec<Value>, Error> {
    let mut rows = Vec::new();
    for (index, rotation) in order.iter().enumerate() {
        if index == 2 && accepted(&rows) {
            break;
        }
        rows.push(evaluate(*rotation)?);
    }
    Ok(rows)
}
#[cfg(test)]
mod tests;
