//! Export a dense descriptor tensor without a matcher or pose prior.
#[cfg(all(feature = "rustnn-coreml", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use navigate_visual_onnx::{ExecutionConfig, Provider, XFeatStages};
    use sha2::{Digest, Sha256};
    use std::{io::Write, path::PathBuf, time::Instant};
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [model, image_path, output] = args.as_slice() else {
        return Err("usage: dense_grid MODEL.webnn IMAGE OUTPUT.json".into());
    };
    let image_bytes = std::fs::read(image_path)?;
    let image = image::load_from_memory(&image_bytes)?.to_luma8();
    let started = Instant::now();
    let mut encoder = XFeatStages::load_rustnn_blocking(
        model,
        None,
        Provider::CoreMlGpu,
        &ExecutionConfig::default(),
        2048,
    )?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let values = encoder.extract_dense_grid_blocking(&image)?;
    let extraction_ms = started.elapsed().as_secs_f64() * 1000.0;
    let tensor_path = output.with_extension("f32le");
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&tensor_path, &bytes)?;
    let scale = (800.0 / f64::from(image.width())).min(600.0 / f64::from(image.height()));
    let offset = [
        (800.0 - f64::from(image.width()) * scale + scale - 1.0) / 2.0,
        (600.0 - f64::from(image.height()) * scale + scale - 1.0) / 2.0,
    ];
    let report = serde_json::json!({
        "shape_chw": [64, 72, 100], "dtype": "f32le", "normalization": "raw",
        "image": image_path, "image_size": [image.width(), image.height()],
        "image_sha256": format!("{:x}", Sha256::digest(&image_bytes)),
        "model_graph": model,
        "encoder_contract": encoder.contract().iter().map(|v| format!("{v:02x}")).collect::<String>(),
        "original_to_model_pixel": {"scale": scale, "offset": offset},
        "descriptor_to_original_pixel": "decoder convention: px=((gx+0.5)*799/100-offset_x)/scale; py=((gy+0.5)*575/72*600/576-offset_y)/scale; not a measured receptive-field center",
        "tensor": tensor_path, "tensor_sha256": format!("{:x}", Sha256::digest(&bytes)),
        "requested_provider": "CoreMlGpu", "load_ms": load_ms,
        "extraction_ms": extraction_ms, "tensor_statistics": encoder.tensor_statistics(),
        "preprocessing": "shared dense-image path; grayscale, letterbox to 800x600; model normalizes and resizes to 800x576",
        "mask": "not provided; correlation must declare descriptor support separately"
    });
    let mut file = std::fs::File::create(output)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    tracing::info!(?output, "Saved raw dense descriptors");
    Ok(())
}

#[cfg(not(all(feature = "rustnn-coreml", target_os = "macos")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("dense_grid requires macOS and the rustnn-coreml feature".into())
}
