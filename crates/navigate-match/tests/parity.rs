//! LoFTR-DS on burn against the recorded ONNX Runtime result of the pinned export.
//!
//! The fixture pairs are the Oryol rooftop photo, its map rendering, crops of both,
//! and the photo against a rotated copy of itself. `ort-loftr.json` holds the
//! correspondences of `loftr.onnx` on the ONNX Runtime CPU provider.
//!
//! The weight file is not in git. Run with the published file:
//!
//! ```sh
//! NAVIGATE_LOFTR_WEIGHTS=loftr-f16.safetensors cargo test -p navigate-match --release -- --ignored
//! ```
use burn::tensor::Device;
use navigate_match::{Loftr, LoftrMatcher, Weights};
use std::{collections::HashMap, path::Path};

/// Fraction of the reference matches that the burn result must also contain.
const MIN_SHARED: f64 = 0.99;
/// Largest query-pixel distance between shared matches.
const MAX_ERROR_PX: f64 = 0.05;

fn key(x: f64, y: f64) -> (i64, i64) {
    ((x * 1000.0).round() as i64, (y * 1000.0).round() as i64)
}

#[test]
#[ignore = "needs the weight file; the golden-case workflow runs it"]
fn burn_matches_agree_with_the_onnx_runtime_reference() {
    let weights = std::env::var("NAVIGATE_LOFTR_WEIGHTS").expect("set NAVIGATE_LOFTR_WEIGHTS");
    let bytes = std::fs::read(&weights).expect("read the weight file");
    let model = Loftr::load(Weights::from_safetensors(&bytes).unwrap(), &Device::flex()).unwrap();
    let matcher = LoftrMatcher::new(model, "parity");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/parity");
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("ort-loftr.json")).unwrap()).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let open = |side: &str| {
            image::open(root.join(format!("{name}-{side}.png")))
                .unwrap()
                .to_luma8()
        };
        let matches =
            pollster::block_on(matcher.match_images(&open("reference"), &open("query"))).unwrap();
        let expected: HashMap<_, _> = case["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let v: Vec<f64> = p
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                (key(v[0], v[1]), (v[2], v[3]))
            })
            .collect();
        let mut shared = 0;
        let mut worst = 0.0_f64;
        for m in &matches {
            if let Some(&(x, y)) = expected.get(&key(m.reference.x, m.reference.y)) {
                shared += 1;
                worst = worst.max((m.query.x - x).hypot(m.query.y - y));
            }
        }
        let fraction = shared as f64 / expected.len().max(matches.len()) as f64;
        assert!(
            fraction >= MIN_SHARED,
            "{name}: {shared} shared of {} reference and {} burn matches",
            expected.len(),
            matches.len()
        );
        assert!(
            worst <= MAX_ERROR_PX,
            "{name}: shared matches differ by up to {worst} px"
        );
    }
}
