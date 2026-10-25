use super::*;
use crate::LoFtrMatcher;
#[cfg(target_os = "macos")]
use crate::Provider;
use navigate_visual::ImageMatcher;
use std::{collections::BTreeMap, path::PathBuf};

#[test]
#[ignore = "requires NAVIGATE_LOFTR_SPLIT manifest with runtime, model, coarse, fine, reference and query; run alone"]
fn split_batches_preserve_real_image_correspondences() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = std::env::var_os("NAVIGATE_LOFTR_SPLIT").ok_or("set NAVIGATE_LOFTR_SPLIT")?;
    let paths: BTreeMap<String, PathBuf> = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let path = |key: &str| paths.get(key).ok_or_else(|| format!("missing {key}"));
    crate::initialize_blocking(path("runtime")?)?;
    let reference = image::open(path("reference")?)?.to_luma8();
    let query = image::open(path("query")?)?.to_luma8();
    let cpu = ExecutionConfig::default();
    let mut original = LoFtrMatcher::load_blocking(path("model")?, &cpu)?;
    let expected = original.match_images_blocking(&reference, &query)?;
    let mut split = LoFtrMatcher::load_split_blocking(path("coarse")?, path("fine")?, &cpu, &cpu)?;
    let actual = split.match_images_blocking(&reference, &query)?;
    assert!(LoFtrMatcher::load_split_blocking(path("model")?, path("fine")?, &cpu, &cpu).is_err());
    assert!(expected.len() > 256 && expected.len() % 256 != 0);
    assert_eq!(expected.len(), actual.len());
    let mut maximum = 0.0_f64;
    for (a, b) in expected.iter().zip(&actual) {
        maximum = maximum
            .max((a.reference - b.reference).norm())
            .max((a.query - b.query).norm());
    }
    assert!(maximum < 1e-3, "split source-pixel deviation {maximum}");
    tracing::info!(
        matches = actual.len(),
        maximum,
        "Verified split LoFTR correspondence parity"
    );
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires NAVIGATE_LOFTR_SPLIT, real fine model and Core ML; run alone"]
fn strict_coreml_fine_batch_keeps_other_patch_pairs_independent()
-> Result<(), Box<dyn std::error::Error>> {
    let manifest = std::env::var_os("NAVIGATE_LOFTR_SPLIT").ok_or("set NAVIGATE_LOFTR_SPLIT")?;
    let paths: BTreeMap<String, PathBuf> = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let path = |key: &str| paths.get(key).ok_or_else(|| format!("missing {key}"));
    crate::initialize_blocking(path("runtime")?)?;
    let execution = ExecutionConfig {
        provider: Provider::CoreMlGpu,
        allow_cpu_fallback: false,
        profile_prefix: Some(
            std::env::temp_dir().join(format!("navigate-fine-{}", std::process::id())),
        ),
        ..ExecutionConfig::default()
    };
    let mut fine = FineStage::load_blocking(path("fine")?, &execution)?;
    assert!(fine.batch > 19);
    let mut inputs: [Vec<f32>; 2] = std::array::from_fn(|j| {
        (0..fine.batch * PATCH)
            .map(|i| ((i + j * 113) as f32 * 0.011).sin())
            .collect()
    });
    let first = run(&mut fine, &inputs)?;
    for input in &mut inputs {
        input[19 * PATCH..].fill(7.0);
    }
    let second = run(&mut fine, &inputs)?;
    for (a, b) in first[..38].iter().zip(&second[..38]) {
        assert!((a - b).abs() < 1e-5);
    }
    assert!(
        first[38..]
            .iter()
            .zip(&second[38..])
            .any(|(a, b)| (a - b).abs() > 1e-3)
    );
    let profile = fine.session.end_profiling()?;
    let events: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&profile)?)?;
    let providers: Vec<_> = events
        .iter()
        .filter_map(|e| e["args"]["provider"].as_str())
        .collect();
    assert!(!providers.is_empty());
    assert!(
        providers.iter().all(|p| *p == "CoreMLExecutionProvider"),
        "{providers:?}"
    );
    std::fs::remove_file(profile)?;
    Ok(())
}
#[cfg(target_os = "macos")]
fn run(fine: &mut FineStage, inputs: &[Vec<f32>; 2]) -> Result<Vec<f32>, InferenceError> {
    let tensor = |i: usize| {
        TensorRef::from_array_view(([fine.batch, 25, 128], inputs[i].as_slice()))
            .map_err(|e| native_runtime::runtime("test fine tensor", e))
    };
    let output = fine
        .session
        .run(ort::inputs!["patches0"=>tensor(0)?,"patches1"=>tensor(1)?])
        .map_err(|e| native_runtime::runtime("test fine batch", e))?;
    Ok(read(&output, "offsets", &[fine.batch as i64, 2])?.to_vec())
}
