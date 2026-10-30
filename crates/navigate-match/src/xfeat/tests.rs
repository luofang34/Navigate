use super::*;
use crate::WeightTensor;

pub(crate) fn synthetic_xfeat_weights() -> Weights {
    let specs: &[(&str, [usize; 3])] = &[
        ("block1.0", [4, 1, 3]),
        ("block1.1", [8, 4, 3]),
        ("block1.2", [8, 8, 3]),
        ("block1.3", [24, 8, 3]),
        ("skip1.1", [24, 1, 1]),
        ("block2.0", [24, 24, 3]),
        ("block2.1", [24, 24, 3]),
        ("block3.0", [64, 24, 3]),
        ("block3.1", [64, 64, 3]),
        ("block3.2", [64, 64, 1]),
        ("block4.0", [64, 64, 3]),
        ("block4.1", [64, 64, 3]),
        ("block4.2", [64, 64, 3]),
        ("block5.0", [128, 64, 3]),
        ("block5.1", [128, 128, 3]),
        ("block5.2", [128, 128, 3]),
        ("block5.3", [64, 128, 1]),
        ("block_fusion.0", [64, 64, 3]),
        ("block_fusion.1", [64, 64, 3]),
        ("block_fusion.2", [64, 64, 1]),
        ("heatmap_head.0", [64, 64, 1]),
        ("heatmap_head.1", [64, 64, 1]),
        ("heatmap_head.2", [1, 64, 1]),
        ("keypoint_head.0", [64, 64, 1]),
        ("keypoint_head.1", [64, 64, 1]),
        ("keypoint_head.2", [64, 64, 1]),
        ("keypoint_head.3", [65, 64, 1]),
    ];
    let mut weights = Weights::default();
    for &(name, [out, input, k]) in specs {
        let len = out * input * k * k;
        let values = (0..len)
            .map(|i| (((i * 17 + 3) % 29) as f32 - 14.0) * 0.01)
            .collect();
        weights.insert(
            format!("{name}.weight"),
            WeightTensor {
                shape: vec![out, input, k, k],
                values,
            },
        );
        let mut bias = vec![0.01; out];
        if name == "keypoint_head.3" {
            bias.fill(0.0);
            bias[27] = 4.0;
        }
        weights.insert(
            format!("{name}.bias"),
            WeightTensor {
                shape: vec![out],
                values: bias,
            },
        );
    }
    weights
}

fn contract_of(weights: Weights) -> [u8; 32] {
    XFeat::load(weights, &Device::flex())
        .expect("load synthetic XFeat")
        .encoder_contract()
}

#[test]
fn f32_and_f16_files_of_one_model_share_the_encoder_contract() {
    use sha2::{Digest, Sha256};
    let weights = synthetic_xfeat_weights();
    let f32_file = weights.to_safetensors(false).expect("F32 file");
    let f16_file = weights.to_safetensors(true).expect("F16 file");
    let f32_sha: [u8; 32] = Sha256::digest(&f32_file).into();
    let f16_sha: [u8; 32] = Sha256::digest(&f16_file).into();
    assert_ne!(f32_sha, f16_sha);

    let from_f32 = contract_of(Weights::from_safetensors(&f32_file).expect("read F32"));
    let from_f16 = contract_of(Weights::from_safetensors(&f16_file).expect("read F16"));
    assert_eq!(from_f32, from_f16);
    assert_eq!(from_f32, XFeat::encoder_contract_of(&weights));
    assert_ne!(from_f32, f32_sha);
    let architecture_only: [u8; 32] = Sha256::digest(XFeat::ENCODER_CONTRACT_V1_DESCRIPTION).into();
    assert_ne!(from_f32, architecture_only);
}

#[test]
fn a_changed_weight_changes_the_encoder_contract_and_is_rejected() {
    let device = Device::flex();
    let model = XFeat::load(synthetic_xfeat_weights(), &device).expect("load");
    let mut perturbed = synthetic_xfeat_weights();
    let mut tensor = perturbed.remove("block5.2.weight").expect("tensor");
    tensor.values[0] += 0.25;
    perturbed.insert("block5.2.weight", tensor);
    let other = XFeat::load(perturbed, &device).expect("load perturbed");
    assert_ne!(model.encoder_contract(), other.encoder_contract());

    let file = [7_u8; 32];
    assert!(model.accepts_encoder_contract(&model.encoder_contract(), &file));
    assert!(!other.accepts_encoder_contract(&model.encoder_contract(), &file));
    assert!(model.accepts_encoder_contract(&file, &file));
    assert!(!model.accepts_encoder_contract(&file, &[8; 32]));
}

#[test]
fn both_zero_signs_give_one_encoder_contract() {
    let mut negative = synthetic_xfeat_weights();
    let mut tensor = negative.remove("block1.0.bias").expect("tensor");
    tensor.values.fill(0.0);
    let mut positive = synthetic_xfeat_weights();
    positive.insert("block1.0.bias", tensor.clone());
    tensor.values.fill(-0.0);
    negative.insert("block1.0.bias", tensor);
    assert_eq!(contract_of(positive), contract_of(negative));
}

fn textured_image(width: u32, height: u32) -> GrayImage {
    let mut image = GrayImage::new(width, height);
    for (x, y, p) in image.enumerate_pixels_mut() {
        p.0[0] = (((x * 7 + y * 13) ^ (x * y)) & 0xff) as u8;
    }
    image
}

#[test]
fn xfeat_loads_and_executes_dense_forward_pass() {
    let device = Device::flex();
    let model = XFeat::load(synthetic_xfeat_weights(), &device).expect("load synthetic XFeat");
    let image = textured_image(160, 120);
    let features = pollster::block_on(model.extract(&image, 128)).expect("extract features");
    assert!(features.len() <= 128);
    assert_eq!(features.descriptors.len(), features.len() * 64);
    for chunk in features.descriptors.as_chunks::<64>().0 {
        let norm: f32 = chunk.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}

#[test]
fn flat_image_returns_empty_features() {
    let device = Device::flex();
    let model = XFeat::load(synthetic_xfeat_weights(), &device).expect("load synthetic XFeat");
    let flat = GrayImage::from_pixel(64, 64, image::Luma([128]));
    let features = pollster::block_on(model.extract(&flat, 64)).expect("flat extract");
    assert!(features.is_empty());
}

#[test]
fn extract_masked_excludes_sky_region_and_retains_terrain_keypoints() {
    let device = Device::flex();
    let model = XFeat::load(synthetic_xfeat_weights(), &device).expect("load synthetic XFeat");
    let (width, height) = (160, 120);
    let half_y = f64::from(height / 2);
    let image = textured_image(width, height);

    let unmasked = pollster::block_on(model.extract(&image, 256)).expect("unmasked extract");
    let none_mask =
        pollster::block_on(model.extract_masked(&image, None, 256)).expect("none mask extract");
    assert_eq!(unmasked, none_mask);
    assert!(unmasked.pixels.iter().any(|p| p[1] < half_y));
    assert!(unmasked.pixels.iter().any(|p| p[1] >= half_y));

    let sky_mask = GrayImage::from_fn(width, height, |_, y| {
        if y < height / 2 {
            image::Luma([0])
        } else {
            image::Luma([255])
        }
    });
    let masked = pollster::block_on(model.extract_masked(&image, Some(&sky_mask), 256))
        .expect("masked extract");
    assert!(!masked.is_empty());
    assert!(
        masked.pixels.iter().all(|p| p[1] >= half_y),
        "expected zero keypoints in y < {half_y}, got {:?}",
        masked.pixels
    );

    let threshold_mask = GrayImage::from_fn(width, height, |_, y| {
        if y < height / 2 {
            image::Luma([127])
        } else {
            image::Luma([128])
        }
    });
    let threshold_features =
        pollster::block_on(model.extract_masked(&image, Some(&threshold_mask), 256))
            .expect("threshold mask extract");
    assert_eq!(threshold_features, masked);
}

#[test]
fn extract_masked_rejects_mismatched_mask_dimensions() {
    let device = Device::flex();
    let model = XFeat::load(synthetic_xfeat_weights(), &device).expect("load synthetic XFeat");
    let image = textured_image(160, 120);
    for (mw, mh) in [(80, 120), (160, 60)] {
        let bad_mask = GrayImage::from_pixel(mw, mh, image::Luma([255]));
        let err = pollster::block_on(model.extract_masked(&image, Some(&bad_mask), 64))
            .expect_err("mismatched mask must fail");
        assert!(matches!(
            err,
            MatchError::Output("valid mask dimensions do not match image")
        ));
    }
}

#[test]
fn truncate_top_keeps_highest_scoring_keypoints_deterministically() {
    let mut descriptors = Vec::new();
    for k in 0..4 {
        let mut d = [0.0_f32; 64];
        d[k] = 1.0;
        descriptors.extend_from_slice(&d);
    }
    let unsorted = SparseFeatures {
        pixels: vec![[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]],
        model_pixels: vec![
            [100.0, 110.0],
            [200.0, 210.0],
            [300.0, 310.0],
            [400.0, 410.0],
        ],
        scores: vec![0.25, 0.90, 0.40, 0.90],
        descriptors,
    };

    assert!(unsorted.truncate_top(0).is_empty());
    let top2 = unsorted.truncate_top(2);
    assert_eq!(top2.len(), 2);
    assert_eq!(top2.pixels, vec![[20.0, 21.0], [40.0, 41.0]]);
    assert_eq!(top2.model_pixels, vec![[200.0, 210.0], [400.0, 410.0]]);
    assert_eq!(top2.scores, vec![0.90, 0.90]);
    assert_eq!(top2.descriptors[1], 1.0);
    assert_eq!(top2.descriptors[64 + 3], 1.0);

    let sorted = top2.truncate_top(1);
    assert_eq!(sorted.len(), 1);
    assert_eq!(sorted.pixels, vec![[20.0, 21.0]]);
}
