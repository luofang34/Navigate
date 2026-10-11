use super::*;
use ort::value::Tensor;

#[test]
#[ignore = "requires NAVIGATE_TEST_ORT and Core ML on macOS; run in its own process"]
#[cfg(target_os = "macos")]
fn cpu_placement_and_strict_fallback_execute_real_graphs() -> Result<(), Box<dyn std::error::Error>>
{
    let library = std::env::var_os("NAVIGATE_TEST_ORT").ok_or("set NAVIGATE_TEST_ORT")?;
    initialize_blocking(Path::new(&library))?;
    let prefix = std::env::temp_dir().join(format!("navigate-placement-{}", std::process::id()));
    let config = ExecutionConfig {
        profile_prefix: Some(prefix),
        ..ExecutionConfig::default()
    };
    let model = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/add.onnx"
    ));
    let mut session = session_blocking(model, &config)?;
    let input = Tensor::from_array(([1_usize, 2], vec![0.25_f32, -2.0]))?;
    let output = session.run(ort::inputs!["input" => input])?;
    let (_, data) = output
        .get("output")
        .ok_or("missing output")?
        .try_extract_tensor::<f32>()?;
    assert_eq!(data, &[0.5, -4.0]);
    drop(output);
    let profile_path = session.end_profiling()?;
    let events: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&profile_path)?)?;
    let providers: Vec<_> = events
        .iter()
        .filter_map(|event| event["args"]["provider"].as_str())
        .collect();
    assert!(!providers.is_empty());
    assert!(
        providers.iter().all(|p| *p == "CPUExecutionProvider"),
        "{providers:?}"
    );
    std::fs::remove_file(profile_path)?;

    let model = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/nonzero.onnx"
    ));
    let strict = ExecutionConfig {
        provider: Provider::CoreMlGpu,
        allow_cpu_fallback: false,
        ..ExecutionConfig::default()
    };
    let failure = session_blocking(model, &strict)
        .err()
        .ok_or("strict graph unexpectedly loaded")?;
    assert!(failure.to_string().contains("CPU EP"), "{failure}");
    let mut fallback = session_blocking(
        model,
        &ExecutionConfig {
            allow_cpu_fallback: true,
            ..strict
        },
    )?;
    let input = Tensor::from_array(([1_usize, 2], vec![0.0_f32, 7.0]))?;
    let output = fallback.run(ort::inputs!["input" => input])?;
    let (shape, values) = output
        .get("output")
        .ok_or("missing NonZero output")?
        .try_extract_tensor::<i64>()?;
    assert_eq!(shape.as_ref(), &[2, 1]);
    assert_eq!(values, &[0, 1]);
    Ok(())
}
