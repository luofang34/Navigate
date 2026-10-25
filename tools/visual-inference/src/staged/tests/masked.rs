use super::*;
use crate::{ExecutionConfig, Provider};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Regression {
    runtime: PathBuf,
    detector: PathBuf,
    original: PathBuf,
    masked: PathBuf,
    reference: PathBuf,
    query: PathBuf,
}

#[test]
#[ignore = "requires NAVIGATE_MASKED_REGRESSION and Core ML; run in its own process"]
fn masked_static_graph_preserves_dynamic_matches_with_unequal_feature_counts()
-> Result<(), Box<dyn std::error::Error>> {
    let path =
        std::env::var_os("NAVIGATE_MASKED_REGRESSION").ok_or("set NAVIGATE_MASKED_REGRESSION")?;
    let config: Regression = serde_json::from_slice(&std::fs::read(path)?)?;
    crate::initialize_blocking(&config.runtime)?;
    let cpu = ExecutionConfig::default();
    let gpu = ExecutionConfig {
        provider: Provider::CoreMlGpu,
        allow_cpu_fallback: false,
        ..cpu.clone()
    };
    let mut encoder =
        XFeatStages::load_blocking(&config.detector, XFeatGraph::Sparse, None, &cpu, &cpu, 2048)?;
    let a = encoder.extract_features_blocking(&image::open(config.reference)?.to_luma8())?;
    let b = encoder.extract_features_blocking(&image::open(config.query)?.to_luma8())?;
    assert_eq!(a.len(), 2048);
    assert_eq!(b.len(), 2048);
    let mut original = native_runtime::session_blocking(&config.original, &cpu)?;
    let mut masked_cpu = native_runtime::session_blocking(&config.masked, &cpu)?;
    let mut masked_gpu = native_runtime::session_blocking(&config.masked, &gpu)?;
    for (na, nb) in [(2048, 2048), (2048, 1662), (317, 129), (1, 3), (0, 50)] {
        let first = prefix(&a.features, na);
        let second = prefix(&b.features, nb);
        let expected = lighterglue::match_features(&mut original, &first, &second)?;
        if na >= 1662 && nb >= 1662 {
            assert!(expected.len() >= 20);
        }
        for (label, session) in [("cpu", &mut masked_cpu), ("coreml", &mut masked_gpu)] {
            let actual = lighterglue::match_features(session, &first, &second)?;
            let pixels = |pairs: &[PixelMatch]| {
                pairs
                    .iter()
                    .map(|p| [p.reference.x, p.reference.y, p.query.x, p.query.y])
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                pixels(&expected),
                pixels(&actual),
                "{label} counts {na}/{nb}"
            );
            tracing::info!(
                label,
                na,
                nb,
                matches = actual.len(),
                "Masked matcher agrees with dynamic graph"
            );
        }
    }
    Ok(())
}

fn prefix(f: &Features, count: usize) -> Features {
    Features {
        pixels: f.pixels[..count].to_vec(),
        model_pixels: f.model_pixels[..count * 2].to_vec(),
        scores: f.scores[..count].to_vec(),
        descriptors: f.descriptors[..count * 64].to_vec(),
        dimensions: 64,
    }
}
