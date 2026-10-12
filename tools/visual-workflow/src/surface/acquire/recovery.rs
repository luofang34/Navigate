//! A bounded retained-window probe shares query features with global recovery.
use super::*;

pub(in crate::surface) fn search_recovery_blocking(
    frame: &Frame,
    reference: &OrthographicReference,
    prior: &PosePrior,
    engine: &mut Engine,
    config: &Config,
    record_matches: bool,
    retained: Option<[u32; 4]>,
) -> Result<SearchResult, Error> {
    let Some(area) = retained else {
        return search_blocking(frame, reference, prior, engine, config, record_matches);
    };
    if config.limits.candidates < 80 {
        return Err(Error::Input(
            "retained-window recovery requires 80 candidates".into(),
        ));
    }
    let started = Instant::now();
    let trial = Trial {
        frame,
        reference,
        prior,
        verifier: PoseVerifier::new(LocalizerConfig::default())?,
        rectification: None,
        record_matches,
        hypothesis_limit: 1,
        probe_fine_after_pass: false,
        config,
    };
    let mut selected = Vec::new();
    let rows = trial.windows_blocking(engine, &[area], started, &mut selected)?;
    let probe_ms = started.elapsed().as_secs_f64() * 1000.0;
    if let Some(selected) = selected.into_iter().next() {
        let first = window_pass(&rows, "retained_window").map(|(row, _)| row);
        return Ok(SearchResult {
            record: json!({"accepted":true,"retained_window_probe":rows,
                "global_search_attempted":false,"first_pass":first,
                "retained_window_ms":probe_ms,"search_to_first_pass_ms":probe_ms,
                "search_total_ms":probe_ms,"search_candidate_bound":80}),
            selected: Some(selected),
            alternatives: Vec::new(),
        });
    }
    let mut result = search_blocking(frame, reference, prior, engine, config, record_matches)?;
    let first_ms = result.record["search_to_first_pass_ms"].as_f64();
    result.record["search_to_first_pass_ms"] = json!(first_ms.map(|ms| probe_ms + ms));
    result.record["retained_window_probe"] = json!(rows);
    result.record["retained_window_ms"] = json!(probe_ms);
    result.record["global_search_attempted"] = json!(true);
    result.record["search_candidate_bound"] = json!(80);
    result.record["search_total_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    Ok(result)
}
