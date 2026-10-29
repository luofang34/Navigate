use super::*;
use crate::WeightTensor;

fn synthetic_xfeat_weights() -> Weights {
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
        weights.insert(
            format!("{name}.bias"),
            WeightTensor {
                shape: vec![out],
                values: vec![0.01; out],
            },
        );
    }
    weights
}

#[test]
fn xfeat_loads_and_executes_dense_forward_pass() {
    let device = Device::flex();
    let model = XFeat::load(synthetic_xfeat_weights(), &device).expect("load synthetic XFeat");
    let mut image = GrayImage::new(160, 120);
    for (x, y, p) in image.enumerate_pixels_mut() {
        p.0[0] = (((x * 7 + y * 13) ^ (x * y)) & 0xff) as u8;
    }
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
