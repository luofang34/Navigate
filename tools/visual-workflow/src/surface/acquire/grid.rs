//! Fixed map coverage separates window search failures from geometry failures.
use super::*;

pub(super) fn windows(
    size: (u32, u32),
    side: u32,
    candidate_limit: usize,
) -> Result<Vec<[u32; 4]>, Error> {
    if !(32..=8192).contains(&side)
        || !side.is_multiple_of(2)
        || size.0 > 8192
        || size.1 > 8192
        || side > size.0
        || side > size.1
    {
        return Err(Error::Input(
            "grid side must be even, 32..=8192, and fit a map no larger than 8192px".into(),
        ));
    }
    let axis = |length: u32| {
        let end = length - side;
        let mut positions: Vec<_> = (0..=end).step_by((side / 2) as usize).collect();
        if positions.last() != Some(&end) {
            positions.push(end);
        }
        positions
    };
    let xs = axis(size.0);
    let ys = axis(size.1);
    let candidates = xs.len().saturating_mul(ys.len()).saturating_mul(4);
    if candidates > candidate_limit || candidates > 4096 {
        return Err(Error::Input(format!(
            "grid needs {candidates} candidates; limit is {}",
            candidate_limit.min(4096)
        )));
    }
    Ok(ys
        .into_iter()
        .flat_map(|y| xs.iter().map(move |&x| [x, y, side, side]))
        .collect())
}

pub fn run_blocking(
    config: &Config,
    root: &Path,
    input: &Path,
    output: &Path,
    side: u32,
    rectification_path: Option<&Path>,
) -> Result<(), Error> {
    run_search_blocking(
        config,
        root,
        input,
        output,
        Search::Grid(side),
        rectification_path,
    )
}

pub(super) enum Search {
    Grid(u32),
    Pyramid(super::pyramid::Options),
}

pub(super) fn run_search_blocking(
    config: &Config,
    root: &Path,
    input: &Path,
    output: &Path,
    search: Search,
    rectification_path: Option<&Path>,
) -> Result<(), Error> {
    let total = Instant::now();
    let (job, query, reference, prior) = load_blocking(config, input, rectification_path)?;
    let (windows, bound) = search.plan(reference.image.dimensions(), config.limits.candidates)?;
    let Query {
        frame,
        rectification,
        report: rectification_report,
    } = query;
    let loading_ms = total.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    let mut engine = Engine::new(config.load_blocking(root)?)?;
    let model_load_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    engine.query_blocking(&frame.image, config)?;
    let query_extraction_ms = start.elapsed().as_secs_f64() * 1000.;
    let trial = Trial {
        frame: &frame,
        reference: &reference,
        prior: &prior,
        verifier: PoseVerifier::new(LocalizerConfig::default())?,
        rectification: rectification.as_ref(),
        record_matches: job.record_matches,
        hypothesis_limit: 1,
        probe_fine_after_pass: false,
        config,
    };
    let preparation_ms = total.elapsed().as_secs_f64() * 1000.;
    let mut result = search.run_blocking(&trial, &mut engine, windows)?;
    result["search_candidate_bound"] = json!(bound);
    bind_result(
        &mut result,
        &job,
        &frame,
        &engine,
        input,
        rectification_report,
    )?;
    result["loading_ms"] = json!(loading_ms);
    result["model_load_ms"] = json!(model_load_ms);
    result["query_extraction_ms"] = json!(query_extraction_ms);
    result["preparation_ms"] = json!(preparation_ms);
    result["elapsed_to_first_pass_ms"] = json!(
        result["search_to_first_pass_ms"]
            .as_f64()
            .map(|v| v + preparation_ms)
    );
    result["query_encodings"] = json!(1);
    result["reference_encodings"] = json!(engine.reference_encodings);
    result["query_feature_bytes"] = json!(engine.query_bytes);
    result["peak_reference_accounted_bytes"] = json!(engine.peak_reference_bytes);
    result["candidate_limit"] = json!(config.limits.candidates);
    result["elapsed_before_output_ms"] = json!(total.elapsed().as_secs_f64() * 1000.);
    assets::write_json_blocking(output, &result)?;
    tracing::info!(path=%output.display(), "Saved window search results");
    Ok(())
}

fn search_blocking(
    trial: &Trial<'_>,
    engine: &mut Engine,
    windows: &[[u32; 4]],
) -> Result<Value, Error> {
    let start = Instant::now();
    let rows = trial.windows_blocking(engine, windows, start, &mut Vec::new())?;
    let first = window_pass(&rows, "grid");
    Ok(json!({"windows":rows,"accepted":first.is_some(),
        "first_pass":first.as_ref().map(|(decision, _)| decision),
        "search_to_first_pass_ms":first.map(|(_, time)| time),
        "search_total_ms":start.elapsed().as_secs_f64()*1000.,
        "search_candidate_bound":windows.len()*4,
        "scope":"Exhaustive diagnostic on fixed half-overlapping square map windows. Include clamped map edges, ordered by y then x then quarter turn. No image scores or query pose truth select the windows. One query encoding; original geometry gates and optional raw pixel gate. Search continues after the first pass. A pass is not independent accuracy or a unique fix. Full map remains resident. Accounted reference bytes exclude models, runtime, output records and decoding peaks."}))
}

#[cfg(test)]
mod tests;

fn load_blocking(
    config: &Config,
    input: &Path,
    rectification_path: Option<&Path>,
) -> Result<(SurfaceJob, Query, OrthographicReference, PosePrior), Error> {
    let job: SurfaceJob = assets::read_json_blocking(input)?;
    if job.reference.window.is_some()
        || job.reference_quarter_turns != [0, 1, 2, 3]
        || !matches!(config.graph, Graph::Sparse | Graph::DenseImage)
    {
        return Err(Error::Input(
            "grid probe needs a full map, ordered quarter turns and independent features".into(),
        ));
    }
    let (frame, reference, prior) =
        job.load_blocking(input.parent().unwrap_or(Path::new(".")), config)?;
    let query = query_blocking(
        frame,
        &job,
        input,
        rectification_path,
        config.limits.image_pixels,
    )?;
    Ok((job, query, reference, prior))
}

fn bind_result(
    result: &mut Value,
    job: &SurfaceJob,
    frame: &Frame,
    engine: &Engine,
    input: &Path,
    rectification_report: Option<Value>,
) -> Result<(), Error> {
    result["job"] = json!(job);
    result["backend"] = json!(engine.backend.identity());
    result["rectification"] = json!(rectification_report);
    result["observation_sha256"] = json!(frame.evidence_sha256());
    result["map_input_sha256"] = json!(hex::encode(Sha256::digest(
        serde_json::to_vec(&job.reference).map_err(|source| Error::Json {
            path: input.into(),
            source
        })?
    )));
    Ok(())
}

impl Search {
    fn plan(&self, size: (u32, u32), limit: usize) -> Result<(Vec<[u32; 4]>, usize), Error> {
        match self {
            Self::Grid(side) => {
                let windows = windows(size, *side, limit)?;
                let bound = windows.len() * 4;
                Ok((windows, bound))
            }
            Self::Pyramid(options) => options.plan(size, limit),
        }
    }
    fn run_blocking(
        self,
        trial: &Trial<'_>,
        engine: &mut Engine,
        windows: Vec<[u32; 4]>,
    ) -> Result<Value, Error> {
        match self {
            Self::Grid(side) => {
                let mut result = search_blocking(trial, engine, &windows)?;
                result["grid_side"] = json!(side);
                Ok(result)
            }
            Self::Pyramid(options) => {
                super::pyramid::search_blocking(trial, engine, windows, options)
            }
        }
    }
}
