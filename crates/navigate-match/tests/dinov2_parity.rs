//! DINOv2 ViT-S/14 on burn against the recorded ONNX Runtime result of the pinned export.
//!
//! The input is a fixed pattern, made by [`pattern`] here and by the recording tool.
//! `ort-dinov2-224.json` holds the class token, three patch tokens and the patch mean
//! of `onnx-community/dinov2-small` `onnx/model.onnx` on the ONNX Runtime CPU provider.
//!
//! The weight file is not in git. Run with the `facebook/dinov2-small` safetensors file:
//!
//! ```sh
//! NAVIGATE_DINOV2_WEIGHTS=model.safetensors cargo test -p navigate-match --release -- --ignored
//! ```
use burn::tensor::Device;
use navigate_match::{DINOV2_WIDTH, Dinov2, Weights};
use std::path::Path;

/// Largest difference between a burn and a reference token value.
const MAX_ERROR: f32 = 2e-3;
/// Least cosine between a burn and a reference token.
const MIN_COSINE: f32 = 0.9999;

fn pattern(n: usize) -> Vec<f32> {
    (0..3 * n * n)
        .map(|i| {
            let (c, y, x) = (i / (n * n), (i / n) % n, i % n);
            1.5 * (0.05 * x as f32 + 0.07 * y as f32 + c as f32).sin()
                + 0.3 * ((x * y) % 7) as f32 / 7.0
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (a.iter().map(|x| x * x).sum::<f32>().sqrt()
        * b.iter().map(|x| x * x).sum::<f32>().sqrt())
}

#[test]
#[ignore = "needs the weight file; the golden-case workflow runs it"]
fn burn_tokens_agree_with_the_onnx_runtime_reference() {
    let weights = std::env::var("NAVIGATE_DINOV2_WEIGHTS").expect("set NAVIGATE_DINOV2_WEIGHTS");
    let bytes = std::fs::read(&weights).expect("read the weight file");
    let model = Dinov2::load(
        Weights::from_safetensors(&bytes).unwrap(),
        224,
        &Device::flex(),
    )
    .unwrap();
    let tokens = pollster::block_on(model.tokens(&pattern(224), 1)).unwrap();
    assert_eq!(tokens.len(), 257 * DINOV2_WIDTH);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/parity");
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("ort-dinov2-224.json")).unwrap()).unwrap();
    let floats = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    for (index, expected) in fixture["tokens"].as_object().unwrap() {
        let t: usize = index.parse().unwrap();
        let got = &tokens[t * DINOV2_WIDTH..(t + 1) * DINOV2_WIDTH];
        let expected = floats(expected);
        let worst = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            worst <= MAX_ERROR && cosine(got, &expected) >= MIN_COSINE,
            "token {t}: max difference {worst}, cosine {}",
            cosine(got, &expected)
        );
    }
    let mean: Vec<f32> = (0..DINOV2_WIDTH)
        .map(|i| (1..257).map(|t| tokens[t * DINOV2_WIDTH + i]).sum::<f32>() / 256.0)
        .collect();
    let expected = floats(&fixture["patch_mean"]);
    assert!(
        cosine(&mean, &expected) >= MIN_COSINE,
        "patch mean cosine {}",
        cosine(&mean, &expected)
    );
}
