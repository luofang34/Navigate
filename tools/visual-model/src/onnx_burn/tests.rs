use super::*;
use burn::tensor::Device;
use navigate_match::LighterGlue;
use std::path::Path;

#[test]
fn converts_real_lighterglue_onnx_and_loads_in_burn() {
    let asset =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../visual-bench/model-assets/lighterglue.onnx");
    let bytes = std::fs::read(&asset).expect("read lighterglue.onnx");
    let f32_bytes = prepare_lighterglue(&bytes, false).expect("convert f32");
    let f16_bytes = prepare_lighterglue(&bytes, true).expect("convert f16");
    assert!(f16_bytes.len() < f32_bytes.len());

    let weights = Weights::from_safetensors(&f32_bytes).expect("parse safetensors");
    assert_eq!(weights.len(), 139);
    let device = Device::flex();
    let model = LighterGlue::load(weights, &device).expect("load LighterGlue");

    let kp0 = vec![-0.4, -0.3, 0.2, 0.1, 0.5, -0.2, -0.1, 0.4];
    let kp1 = vec![-0.4, -0.3, 0.2, 0.1, 0.5, -0.2, -0.1, 0.4];
    let mut d0 = vec![0.0_f32; 4 * 64];
    for i in 0..4 {
        for c in 0..64 {
            d0[i * 64 + c] = if c % 4 == i { 0.5 } else { 0.0 };
        }
    }
    let scores =
        pollster::block_on(model.log_assignment(&kp0, &kp1, &d0, &d0)).expect("forward pass");
    assert_eq!(scores.len(), 16);
    assert!(scores.iter().all(|v| v.is_finite()));
    for i in 0..4 {
        let diag = scores[i * 4 + i];
        for j in 0..4 {
            if i != j {
                assert!(
                    diag > scores[i * 4 + j],
                    "identical feature {i} should score higher on diagonal ({diag}) than {j} ({})",
                    scores[i * 4 + j]
                );
            }
        }
    }
}
