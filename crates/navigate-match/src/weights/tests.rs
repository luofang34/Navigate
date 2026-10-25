use super::*;

fn sample() -> Weights {
    let mut weights = Weights::default();
    weights.insert(
        "a.weight",
        WeightTensor {
            shape: vec![2, 3],
            values: vec![0.5, -1.0, 2.0, 0.25, 1e-3, -7.5],
        },
    );
    weights.insert(
        "a.bias",
        WeightTensor {
            shape: vec![2],
            values: vec![1.0, -2.0],
        },
    );
    weights
}

#[test]
fn f32_files_round_trip_exactly() {
    let weights = sample();
    let bytes = weights.to_safetensors(false).unwrap();
    assert_eq!(Weights::from_safetensors(&bytes).unwrap(), weights);
}

#[test]
fn f16_files_round_trip_within_half_precision() {
    let weights = sample();
    let read = Weights::from_safetensors(&weights.to_safetensors(true).unwrap()).unwrap();
    for (name, tensor) in &weights.tensors {
        let other = &read.tensors[name];
        assert_eq!(other.shape, tensor.shape);
        for (a, b) in tensor.values.iter().zip(&other.values) {
            assert!(
                (a - b).abs() <= a.abs() * 1e-3 + 1e-6,
                "{name}: {a} against {b}"
            );
        }
    }
}

#[test]
fn a_wrong_shape_names_the_tensor_and_both_shapes() {
    let mut weights = sample();
    let error = weights.take("a.weight", &[3, 2]).unwrap_err();
    assert!(matches!(
        error,
        MatchError::WeightShape { ref name, ref actual, ref expected }
            if name == "a.weight" && actual == &[2, 3] && expected == &[3, 2]
    ));
}

#[test]
fn a_tensor_left_after_loading_is_an_error() {
    let mut weights = sample();
    weights.take("a.weight", &[2, 3]).unwrap();
    let error = weights.finish().unwrap_err();
    assert!(
        matches!(error, MatchError::UnusedWeights { count: 1, ref example } if example == "a.bias")
    );
}

#[test]
fn a_missing_tensor_is_an_error() {
    let error = sample().take("b.weight", &[1]).unwrap_err();
    assert!(matches!(error, MatchError::MissingWeight { ref name } if name == "b.weight"));
}

#[test]
fn malformed_files_are_rejected() {
    assert!(matches!(
        Weights::from_safetensors(b"not a file"),
        Err(MatchError::WeightFile(_))
    ));
}
