use super::*;

fn tensor(shape: &[usize], values: Vec<f32>) -> WeightTensor {
    WeightTensor {
        shape: shape.to_vec(),
        values,
    }
}

#[test]
fn folding_scales_each_output_channel_and_makes_the_bias() {
    let weight = tensor(&[2, 1, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let (gamma, beta, mean, var) = ([2.0, 0.5], [0.1, -0.2], [1.0, -4.0], [3.0, 0.25]);
    let (folded, bias) = fold(weight, &gamma, &beta, &mean, &var);
    let scale = [
        2.0 / (3.0_f32 + 1e-5).sqrt(),
        0.5 / (0.25_f32 + 1e-5).sqrt(),
    ];
    let expected = [
        1.0 * scale[0],
        2.0 * scale[0],
        3.0 * scale[1],
        4.0 * scale[1],
    ];
    for (a, b) in folded.values.iter().zip(expected) {
        assert!((a - b).abs() < 1e-6);
    }
    assert_eq!(bias.shape, vec![2]);
    assert!((bias.values[0] - (0.1 - 1.0 * scale[0])).abs() < 1e-6);
    assert!((bias.values[1] - (-0.2 + 4.0 * scale[1])).abs() < 1e-6);
}

#[test]
fn the_backbone_has_seventeen_folded_convolutions() {
    let pairs = folded_pairs();
    assert_eq!(pairs.len(), 17);
    assert!(pairs.contains(&(
        "backbone.layer3.0.downsample.0".into(),
        "backbone.layer3.0.downsample.1".into()
    )));
    assert!(
        !pairs
            .iter()
            .any(|(conv, _)| conv.starts_with("backbone.layer1.0.downsample"))
    );
}

#[test]
fn a_checkpoint_without_a_normalization_is_rejected() {
    let mut checkpoint = Weights::default();
    checkpoint.insert("backbone.conv1.weight", tensor(&[1, 1, 1, 1], vec![1.0]));
    let error = loftr_from_checkpoint(checkpoint).unwrap_err();
    assert!(
        matches!(error, MatchError::MissingWeight { ref name } if name == "backbone.bn1.weight")
    );
}
