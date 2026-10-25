//! Match `<name>-reference.png` and `<name>-query.png` pairs and write a JSON report.
//!
//! The report has the layout of `tools/visual-inference/examples/dense_pairs.rs`,
//! so the two backends can be compared pair by pair.
//!
//! ```sh
//! cargo run --release -p navigate-match --features wgpu --example pairs -- \
//!     WEIGHTS.safetensors PAIRS_DIRECTORY REPORT.json wgpu
//! ```
use burn::tensor::Device;
use navigate_match::{Loftr, LoftrMatcher, Weights};
use std::{path::Path, time::Instant};

fn device(name: &str) -> Result<Device, Box<dyn std::error::Error>> {
    match name {
        "flex" => Ok(Device::flex()),
        #[cfg(feature = "wgpu")]
        "wgpu" => Ok(Device::wgpu(Default::default())),
        other => Err(format!("unknown or disabled device {other}").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [weights, pairs, output, device_name] = <[String; 4]>::try_from(args)
        .map_err(|_| "usage: pairs WEIGHTS PAIRS_DIRECTORY REPORT DEVICE")?;
    let device = device(&device_name)?;
    let bytes = std::fs::read(&weights)?;
    let started = Instant::now();
    let model = Loftr::load(Weights::from_safetensors(&bytes)?, &device)?;
    let matcher = LoftrMatcher::new(model, "local");
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut names: Vec<_> = std::fs::read_dir(&pairs)?
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter_map(|name| name.strip_suffix("-reference.png").map(str::to_owned))
        .collect();
    names.sort();
    let mut cases = Vec::new();
    for base in names {
        let reference =
            image::open(Path::new(&pairs).join(format!("{base}-reference.png")))?.to_luma8();
        let query = image::open(Path::new(&pairs).join(format!("{base}-query.png")))?.to_luma8();
        let mut runs = Vec::new();
        let mut matches = Vec::new();
        for _ in 0..3 {
            let started = Instant::now();
            matches = pollster::block_on(matcher.match_images(&reference, &query))?;
            runs.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        tracing::info!(base, matches = matches.len(), ?runs, "burn LoFTR result");
        cases.push(serde_json::json!({
            "name": format!("{base}-reference.png"),
            "elapsed_ms": runs,
            "pairs": matches.iter().map(|p| serde_json::json!({
                "reference": [p.reference.x, p.reference.y],
                "query": [p.query.x, p.query.y],
            })).collect::<Vec<_>>(),
        }));
    }
    let report = serde_json::json!({"backend": matcher.identity(), "device": device_name, "load_ms": load_ms, "cases": cases});
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
