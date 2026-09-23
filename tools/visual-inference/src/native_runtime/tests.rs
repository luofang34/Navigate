use super::*;

#[test]
fn invalid_cpu_threads_fail_before_runtime_load() {
    for threads in [0, 65] {
        let config = ExecutionConfig {
            threads,
            ..ExecutionConfig::default()
        };
        assert!(matches!(
            validate_config(&config),
            Err(InferenceError::Invalid(_))
        ));
    }
}

#[test]
fn negative_nvidia_device_is_rejected() {
    for provider in [Provider::Cuda, Provider::TensorRt] {
        let config = ExecutionConfig {
            provider,
            nvidia_device_id: -1,
            ..ExecutionConfig::default()
        };
        assert!(matches!(
            validate_config(&config),
            Err(InferenceError::Invalid(_))
        ));
    }
}

#[test]
fn empty_tensorrt_workspace_is_rejected() {
    let config = ExecutionConfig {
        provider: Provider::TensorRt,
        tensor_rt_workspace_bytes: 0,
        ..ExecutionConfig::default()
    };
    assert!(matches!(
        validate_config(&config),
        Err(InferenceError::Invalid(_))
    ));
}

#[test]
#[ignore = "requires NAVIGATE_TEST_ORT pointing to a runtime without NVIDIA providers"]
fn unavailable_accelerators_cannot_become_cpu_success() -> Result<(), Box<dyn std::error::Error>> {
    use ort::{ep::ExecutionProvider, value::Tensor};
    let library = std::env::var_os("NAVIGATE_TEST_ORT").ok_or("set NAVIGATE_TEST_ORT")?;
    initialize_blocking(Path::new(&library))?;
    assert!(!ep::CUDA::default().is_available()?);
    assert!(!ep::TensorRT::default().is_available()?);
    let model = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/add.onnx"
    ));
    let mut cpu = session_blocking(model, &ExecutionConfig::default())?;
    let input = Tensor::from_array(([1_usize, 2], vec![0.25_f32, -2.0]))?;
    let output = cpu.run(ort::inputs!["input" => input])?;
    let value = output.get("output").ok_or("missing CPU output")?;
    let (_, data) = value.try_extract_tensor::<f32>()?;
    assert_eq!(data, &[0.5, -4.0]);
    for provider in [Provider::Cuda, Provider::TensorRt] {
        let config = ExecutionConfig {
            provider,
            ..ExecutionConfig::default()
        };
        assert!(
            matches!(session_blocking(model, &config), Err(InferenceError::Runtime { operation, .. }) if operation.contains("configure requested"))
        );
    }
    Ok(())
}

#[test]
fn cache_creation_failure_keeps_path_and_source() {
    let model = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/add.onnx"
    ));
    let config = ExecutionConfig {
        provider: Provider::TensorRt,
        engine_cache_directory: Some(model.to_owned()),
        ..ExecutionConfig::default()
    };
    assert!(
        matches!(providers_blocking(model, &config), Err(InferenceError::Cache { path, source }) if path == model && source.kind() == std::io::ErrorKind::AlreadyExists)
    );
}
