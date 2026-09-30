use super::*;

#[test]
fn model_runtime_device_and_shape_changes_cannot_reuse_compiled_weights() {
    let config = ExecutionConfig {
        provider: Provider::CoreMlGpu,
        ..Default::default()
    };
    let dimensions = [("features".into(), 1024)];
    let base = cache_key(b"weights-a", &config, &dimensions, "runtime-a");
    assert_eq!(
        base,
        cache_key(b"weights-a", &config, &dimensions, "runtime-a")
    );
    assert_ne!(
        base,
        cache_key(b"weights-b", &config, &dimensions, "runtime-a")
    );
    assert_ne!(
        base,
        cache_key(b"weights-a", &config, &dimensions, "runtime-b")
    );
    assert_ne!(
        base,
        cache_key(
            b"weights-a",
            &config,
            &[("features".into(), 512)],
            "runtime-a"
        )
    );
    for changed in [
        ExecutionConfig {
            provider: Provider::CoreMlAne,
            ..config.clone()
        },
        ExecutionConfig {
            threads: 2,
            ..config.clone()
        },
        ExecutionConfig {
            cpu_spinning: true,
            ..config.clone()
        },
    ] {
        assert_ne!(
            base,
            cache_key(b"weights-a", &changed, &dimensions, "runtime-a")
        );
    }
}

#[test]
fn disabled_or_other_provider_does_not_touch_model_or_load_runtime() -> Result<(), InferenceError> {
    for config in [
        ExecutionConfig {
            provider: Provider::CoreMlGpu,
            ..Default::default()
        },
        ExecutionConfig {
            coreml_cache_directory: Some("unused-cache".into()),
            ..Default::default()
        },
    ] {
        assert!(prepare_blocking(Path::new("missing.onnx"), &config, &[])?.is_none());
    }
    Ok(())
}

#[test]
#[ignore = "requires NAVIGATE_TEST_ORT with Core ML on macOS"]
fn cached_sessions_use_replaced_weights_and_reject_external_weights()
-> Result<(), Box<dyn std::error::Error>> {
    use crate::native_runtime::{initialize_blocking, session_blocking};
    let library = std::env::var_os("NAVIGATE_TEST_ORT").ok_or("set NAVIGATE_TEST_ORT")?;
    initialize_blocking(Path::new(&library))?;
    let root = std::env::temp_dir().join(format!("navigate-coreml-cache-{}", std::process::id()));
    std::fs::create_dir(&root)?;
    let result = check_cached_weights_blocking(&root);
    std::fs::remove_dir_all(&root)?;
    result?;
    fn check_cached_weights_blocking(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let assets = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets"));
        let path = root.join("model.onnx");
        let config = ExecutionConfig {
            provider: Provider::CoreMlGpu,
            coreml_cache_directory: Some(root.join("cache")),
            ..Default::default()
        };
        for factor in [2, 2, 3] {
            std::fs::copy(assets.join(format!("cache-times-{factor}.onnx")), &path)?;
            let mut session = session_blocking(&path, &config)?;
            let input = ort::value::Tensor::from_array(([1_usize, 2], vec![1.5_f32, -2.0]))?;
            let output = session.run(ort::inputs!["input" => input])?;
            let (_, data) = output
                .get("output")
                .ok_or("missing output")?
                .try_extract_tensor::<f32>()?;
            assert_eq!(data, &[1.5 * factor as f32, 0.0]);
        }
        assert_eq!(std::fs::read_dir(root.join("cache"))?.count(), 2);
        std::fs::copy(assets.join("cache-external.onnx"), &path)?;
        std::fs::copy(
            assets.join("cache-weights.bin"),
            root.join("cache-weights.bin"),
        )?;
        let uncached = ExecutionConfig {
            coreml_cache_directory: None,
            ..config.clone()
        };
        assert!(session_blocking(&path, &uncached).is_ok());
        assert!(matches!(
            session_blocking(&path, &config),
            Err(InferenceError::Runtime { .. })
        ));
        Ok(())
    }
    Ok(())
}
