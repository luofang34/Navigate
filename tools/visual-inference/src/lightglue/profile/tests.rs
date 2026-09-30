use super::*;
use ort::value::{Shape, SymbolicDimensions, Tensor};

fn tensor(symbol: &str, width: i64) -> ValueType {
    ValueType::Tensor {
        ty: TensorElementType::Float32,
        shape: Shape::new([1, -1, width]),
        dimension_symbols: SymbolicDimensions::new([String::new(), symbol.into(), String::new()]),
    }
}
fn metadata() -> Vec<(&'static str, ValueType)> {
    vec![
        ("keypoints0", tensor("reference_count", 2)),
        ("descriptors0", tensor("reference_count", 256)),
        ("keypoints1", tensor("query_count", 2)),
        ("descriptors1", tensor("query_count", 256)),
    ]
}
fn names(inputs: &[(&str, ValueType)]) -> Result<Vec<String>, InferenceError> {
    dimensions(
        &inputs
            .iter()
            .map(|(name, dtype)| (*name, dtype))
            .collect::<Vec<_>>(),
    )
}
#[test]
fn feature_profiles_use_export_dimension_names_and_reject_incompatible_inputs() {
    let inputs = metadata();
    assert_eq!(
        names(&inputs).expect("dynamic model"),
        ["query_count", "reference_count"]
    );
    assert!(names(&inputs[..3]).is_err());
    for invalid in [
        tensor("", 2),
        tensor("reference_count", 3),
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape: Shape::new([1, 1024, 2]),
            dimension_symbols: SymbolicDimensions::empty(3),
        },
    ] {
        let mut inputs = metadata();
        inputs[0].1 = invalid;
        assert!(names(&inputs).is_err());
    }
    let mut inputs = metadata();
    inputs[3].0 = "unknown";
    assert!(names(&inputs).is_err());
}

#[test]
#[ignore = "requires NAVIGATE_TEST_ORT pointing to an ONNX Runtime library"]
fn unequal_feature_counts_execute_dynamic_without_padding_or_truncation()
-> Result<(), Box<dyn std::error::Error>> {
    let library = std::env::var_os("NAVIGATE_TEST_ORT").ok_or("set NAVIGATE_TEST_ORT")?;
    crate::initialize_blocking(Path::new(&library))?;
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/feature_counts.onnx"
    ));
    let execution = ExecutionConfig::default();
    let mut dynamic = native_runtime::session_blocking(path, &execution)?;
    let mut profile = KeypointProfile::load_blocking(path, &dynamic, 32, &execution)?;
    for (reference, query) in [(32, 32), (31, 32), (32, 31), (33, 34)] {
        let specialized = profile.session_for(reference, query);
        assert_eq!(specialized.is_some(), reference == 32 && query == 32);
        let session = specialized.unwrap_or(&mut dynamic);
        let mut feeds = Vec::new();
        for (i, count) in [reference, query].into_iter().enumerate() {
            for (name, width) in [("keypoints", 2), ("descriptors", 256)] {
                let tensor =
                    Tensor::from_array(([1, count, width], vec![count as f32; count * width]))?;
                feeds.push((format!("{name}{i}"), tensor.into_dyn()));
            }
        }
        let output = session.run(feeds)?;
        for (i, count) in [reference, query].into_iter().enumerate() {
            for (name, width) in [("keypoints", 2), ("descriptors", 256)] {
                let value = output
                    .get(format!("{name}{i}_out"))
                    .ok_or("missing fixture output")?;
                let (shape, values) = value.try_extract_tensor::<f32>()?;
                assert_eq!(shape.as_ref(), [1, count as i64, width as i64]);
                assert_eq!(values, vec![count as f32; count * width]);
            }
        }
    }
    Ok(())
}
