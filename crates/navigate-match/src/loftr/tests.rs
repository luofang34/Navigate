use super::*;
use crate::{WeightTensor, loftr_from_checkpoint};

fn device() -> Device {
    Device::flex()
}

/// Deterministic values in −scale through scale.
fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

struct Checkpoint {
    weights: Weights,
    seed: u64,
}

impl Checkpoint {
    fn add(&mut self, name: &str, shape: &[usize]) {
        self.seed += 1;
        let count = shape.iter().product();
        self.weights.insert(
            name,
            WeightTensor {
                shape: shape.to_vec(),
                values: values(count, self.seed, 0.05),
            },
        );
    }

    fn norm(&mut self, name: &str, channels: usize) {
        self.add(&format!("{name}.weight"), &[channels]);
        self.add(&format!("{name}.bias"), &[channels]);
        self.add(&format!("{name}.running_mean"), &[channels]);
        let variance = values(channels, self.seed, 0.5)
            .iter()
            .map(|v| 1.0 + v)
            .collect();
        self.weights.insert(
            format!("{name}.running_var"),
            WeightTensor {
                shape: vec![channels],
                values: variance,
            },
        );
    }

    fn encoder(&mut self, name: &str, width: usize) {
        for part in ["q_proj", "k_proj", "v_proj", "merge"] {
            self.add(&format!("{name}.{part}.weight"), &[width, width]);
        }
        self.add(&format!("{name}.mlp.0.weight"), &[2 * width, 2 * width]);
        self.add(&format!("{name}.mlp.2.weight"), &[width, 2 * width]);
        for norm in ["norm1", "norm2"] {
            self.add(&format!("{name}.{norm}.weight"), &[width]);
            self.add(&format!("{name}.{norm}.bias"), &[width]);
        }
    }
}

/// Every tensor of `loftr_outdoor.ckpt`, with the checkpoint names and shapes.
fn synthetic_checkpoint() -> Weights {
    let mut c = Checkpoint {
        weights: Weights::default(),
        seed: 0,
    };
    c.add("backbone.conv1.weight", &[128, 1, 7, 7]);
    c.norm("backbone.bn1", 128);
    let mut input = 128;
    for (layer, output) in [(1, 128), (2, 196), (3, 256)] {
        for block in 0..2 {
            let name = format!("backbone.layer{layer}.{block}");
            let from = if block == 0 { input } else { output };
            c.add(&format!("{name}.conv1.weight"), &[output, from, 3, 3]);
            c.norm(&format!("{name}.bn1"), output);
            c.add(&format!("{name}.conv2.weight"), &[output, output, 3, 3]);
            c.norm(&format!("{name}.bn2"), output);
            if layer > 1 && block == 0 {
                c.add(
                    &format!("{name}.downsample.0.weight"),
                    &[output, input, 1, 1],
                );
                c.norm(&format!("{name}.downsample.1"), output);
            }
        }
        input = output;
    }
    c.add("backbone.layer3_outconv.weight", &[256, 256, 1, 1]);
    c.add("backbone.layer2_outconv.weight", &[256, 196, 1, 1]);
    c.add("backbone.layer2_outconv2.0.weight", &[256, 256, 3, 3]);
    c.norm("backbone.layer2_outconv2.1", 256);
    c.add("backbone.layer2_outconv2.3.weight", &[196, 256, 3, 3]);
    c.add("backbone.layer1_outconv.weight", &[196, 128, 1, 1]);
    c.add("backbone.layer1_outconv2.0.weight", &[196, 196, 3, 3]);
    c.norm("backbone.layer1_outconv2.1", 196);
    c.add("backbone.layer1_outconv2.3.weight", &[128, 196, 3, 3]);
    for layer in 0..8 {
        c.encoder(&format!("loftr_coarse.layers.{layer}"), 256);
    }
    for layer in 0..2 {
        c.encoder(&format!("loftr_fine.layers.{layer}"), 128);
    }
    for part in ["down_proj", "merge_feat"] {
        c.add(&format!("fine_preprocess.{part}.weight"), &[128, 256]);
        c.add(&format!("fine_preprocess.{part}.bias"), &[128]);
    }
    c.weights
}

#[test]
fn the_converter_output_loads_and_the_model_runs() {
    let checkpoint = synthetic_checkpoint();
    assert_eq!(
        checkpoint.len(),
        194,
        "the checkpoint without num_batches_tracked has 194 tensors"
    );
    let model = Loftr::load(loftr_from_checkpoint(checkpoint).unwrap(), &device()).unwrap();
    let size = [64, 64];
    let image = values(64 * 64, 7, 0.5)
        .iter()
        .map(|v| v + 0.5)
        .collect::<Vec<_>>();
    let result = pollster::block_on(model.correspondences(image.clone(), image, size)).unwrap();
    for c in result {
        assert!(c.first.iter().chain(&c.second).all(|v| v.is_finite()));
        assert!(c.confidence > coarse::THRESHOLD);
    }
}

#[test]
fn sizes_that_are_not_multiples_of_eight_are_rejected() {
    let model = Loftr::load(
        loftr_from_checkpoint(synthetic_checkpoint()).unwrap(),
        &device(),
    )
    .unwrap();
    let error =
        pollster::block_on(model.correspondences(vec![0.0; 60 * 64], vec![0.0; 60 * 64], [60, 64]));
    assert!(matches!(error, Err(MatchError::Output(_))));
}

#[test]
fn aligned_corner_weights_keep_the_end_samples() {
    assert_eq!(
        interpolate::weights(2, 3),
        vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0]
    );
    let rows = interpolate::weights(60, 120);
    for row in rows.chunks(60) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}

#[test]
fn upsampling_matches_the_weight_matrices() {
    let x = layers::upload(vec![0.0, 1.0, 2.0, 3.0], [1, 1, 2, 2], &device());
    let y: Vec<f32> = interpolate::upsample_align_corners(x, [3, 3])
        .into_data()
        .iter::<f32>()
        .collect();
    let expected = [0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0];
    for (a, b) in y.iter().zip(expected) {
        assert!((a - b).abs() < 1e-6, "{y:?}");
    }
}

#[test]
fn the_position_encoding_uses_the_original_frequencies() {
    let pe = position::values(256, [2, 3]);
    let plane = 6;
    assert!((pe[0] - 1.0_f32.sin()).abs() < 1e-6, "x = 1, frequency 1");
    assert!(
        (pe[plane + 2] - 3.0_f32.cos()).abs() < 1e-6,
        "x = 3, frequency 1"
    );
    assert!(
        (pe[2 * plane + 3] - 2.0_f32.sin()).abs() < 1e-6,
        "y = 2, frequency 1"
    );
    let frequency = (-2.0_f32).exp();
    assert!(
        (pe[4 * plane] - frequency.sin()).abs() < 1e-6,
        "second group has frequency exp(-2)"
    );
}

#[test]
fn linear_attention_matches_the_reference_formula() {
    let [length, sources, width] = [3, 4, 16];
    let depth = width / 8;
    let q = values(length * width, 1, 1.0);
    let k = values(sources * width, 2, 1.0);
    let v = values(sources * width, 3, 1.0);
    let upload = |data: &[f32], l: usize| layers::upload(data.to_vec(), [1, l, width], &device());
    let heads = transformer::Heads::new(width, &device());
    let out: Vec<f32> = transformer::linear_attention(
        upload(&q, length),
        upload(&k, sources),
        upload(&v, sources),
        &heads,
    )
    .into_data()
    .iter::<f32>()
    .collect();
    let feature = |x: f32| if x > 0.0 { x + 1.0 } else { x.exp() };
    for l in 0..length {
        for c in 0..width {
            let head = c / depth;
            let channels = head * depth..(head + 1) * depth;
            let (mut numerator, mut denominator) = (0.0_f32, 0.0_f32);
            for s in 0..sources {
                let weight: f32 = channels
                    .clone()
                    .map(|d| feature(q[l * width + d]) * feature(k[s * width + d]))
                    .sum();
                denominator += weight;
                numerator += weight * v[s * width + c];
            }
            let expected = numerator / (denominator + 1e-6);
            let actual = out[l * width + c];
            assert!(
                (actual - expected).abs() < 1e-4,
                "row {l} channel {c}: {actual} against {expected}"
            );
        }
    }
}

#[test]
fn coarse_selection_needs_threshold_border_and_mutual_maximum() {
    let grid = [6, 6];
    let mut row_max = vec![0.0; 36];
    let mut row_argmax = vec![0; 36];
    let mut column_max = vec![0.0; 36];
    for (i, j, confidence) in [(14, 15, 0.9), (15, 21, 0.5), (8, 14, 0.15), (2, 14, 0.9)] {
        row_max[i] = confidence;
        row_argmax[i] = j as i64;
        column_max[j] = f32::max(column_max[j], confidence);
    }
    column_max[21] = 0.6;
    let matches = coarse::select(&row_max, &row_argmax, &column_max, [grid; 2]);
    assert_eq!(
        matches,
        vec![coarse::CoarseMatch {
            i: 14,
            j: 15,
            confidence: 0.9
        }]
    );
}

#[test]
fn fine_windows_start_at_the_padded_cell_corner() {
    let indices = fine::window_indices(&[81], 80, 324);
    assert_eq!(indices.len(), fine::WINDOW * fine::WINDOW);
    assert_eq!(indices[0], 4 * 324 + 4);
    assert_eq!(indices[fine::WINDOW * fine::WINDOW - 1], 8 * 324 + 8);
    let grid = fine::grid();
    assert_eq!(&grid[..4], &[-1.0, -1.0, -0.5, -1.0]);
    assert_eq!(&grid[24..26], &[0.0, 0.0]);
}
