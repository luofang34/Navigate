use crate::{ExecutionConfig, Provider, XFeatGraph, XFeatStages};
use navigate_visual_workflow::FeatureBackend;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Regression {
    runtime: PathBuf,
    detector: PathBuf,
    original: PathBuf,
    prepared: PathBuf,
    features: usize,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    reference: PathBuf,
    query: PathBuf,
    matches: usize,
}

#[test]
#[ignore = "requires NAVIGATE_MATCH_REGRESSION JSON and Core ML; run in its own process"]
fn prepared_matcher_preserves_every_real_correspondence() -> Result<(), Box<dyn std::error::Error>>
{
    let path =
        std::env::var_os("NAVIGATE_MATCH_REGRESSION").ok_or("set NAVIGATE_MATCH_REGRESSION")?;
    let config: Regression = serde_json::from_slice(&std::fs::read(path)?)?;
    crate::initialize_blocking(&config.runtime)?;
    let cpu = ExecutionConfig::default();
    let gpu = ExecutionConfig {
        provider: Provider::CoreMlGpu,
        allow_cpu_fallback: false,
        ..ExecutionConfig::default()
    };
    let mut original = XFeatStages::load_blocking(
        &config.detector,
        XFeatGraph::Sparse,
        Some(&config.original),
        &cpu,
        &cpu,
        config.features,
    )?;
    let mut prepared = XFeatStages::load_blocking(
        &config.detector,
        XFeatGraph::Sparse,
        Some(&config.prepared),
        &cpu,
        &gpu,
        config.features,
    )?;
    assert!(!config.cases.is_empty());
    for case in config.cases {
        let reference = image::open(&case.reference)?.to_luma8();
        let query = image::open(&case.query)?.to_luma8();
        let a = original.extract_features_blocking(&reference)?;
        let b = original.extract_features_blocking(&query)?;
        let expected = original.match_features_blocking(&a, &b)?;
        let actual = prepared.match_features_blocking(&a, &b)?;
        assert_eq!(expected.len(), case.matches, "{} fixture", case.name);
        assert_eq!(actual.len(), expected.len(), "{} count", case.name);
        for (left, right) in expected.iter().zip(&actual) {
            assert_eq!(
                left.reference, right.reference,
                "{} reference pixel",
                case.name
            );
            assert_eq!(left.query, right.query, "{} query pixel", case.name);
        }
    }
    Ok(())
}
