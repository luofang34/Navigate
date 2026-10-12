//! Prepare a source-bound pinhole image without changing geometric acceptance.
use crate::{Error, assets, job::Image};
use image::{DynamicImage, GrayImage, ImageFormat};
use navigate_recording::Camera;
use navigate_visual::RectificationMap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Cursor, path::Path, time::Instant};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Specification {
    image: Image,
    source_camera: Camera,
    target_camera: Camera,
    distortion: [f64; 5],
    max_map_bytes: usize,
}

pub fn run_blocking(spec_path: &Path, output: &Path, iterations: usize) -> Result<(), Error> {
    let total = Instant::now();
    if !(1..=20).contains(&iterations) {
        return Err(Error::Input(
            "rectification iterations must be in 1..=20".into(),
        ));
    }
    let bytes = assets::read_blocking(spec_path, 1024 * 1024)?;
    let mut spec: Specification = serde_json::from_slice(&bytes).map_err(|source| Error::Json {
        path: spec_path.into(),
        source,
    })?;
    if spec.max_map_bytes > 64 * 1024 * 1024 {
        return Err(Error::Input(
            "rectification map limit must not exceed 64 MiB".into(),
        ));
    }
    let root = spec_path.parent().unwrap_or(Path::new("."));
    let source_path = root.join(&spec.image.path);
    spec.image.path = source_path.canonicalize().map_err(|source| Error::Io {
        operation: "resolve raw image",
        path: source_path,
        source,
    })?;
    let image = assets::image_verified_blocking(
        &spec.image.path,
        16 * 1024 * 1024,
        Some(&spec.image.sha256),
    )?;
    let loading_ms = total.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let map = RectificationMap::new(
        spec.source_camera.model(),
        spec.target_camera.model(),
        spec.distortion,
        spec.max_map_bytes,
    )?;
    let map_build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (rectified, warps_ms) = repeat(&map, &image, iterations)?;
    let writing = Instant::now();
    std::fs::create_dir(output).map_err(|source| Error::Io {
        operation: "create rectification output directory",
        path: output.into(),
        source,
    })?;
    let png = save_blocking(output, rectified)?;
    let image_write_ms = writing.elapsed().as_secs_f64() * 1000.0;
    assets::write_json_blocking(
        &output.join("rectified.json"),
        &json!({
            "image":png,"camera":spec.target_camera,"source":spec,
            "specification_sha256":hex::encode(Sha256::digest(bytes)),
            "pixel_space":"Rectified pinhole. Integer pixel centres; unchanged optical axes. Use this image and camera together in matching, tracking and geometry.",
            "sampling":"Brown-Conrady k1,k2,p1,p2,k3. Bilinear grayscale; nearest-integer output. All target pixel centres map inside the raw source. No invalid border filling.",
            "scope":"Image preparation only. Rectified pixel residuals are not raw pixel residuals. This output does not establish raw-camera navigation acceptance.",
            "map_bytes":map.map_bytes(),"input_loading_ms":loading_ms,"map_build_ms":map_build_ms,
            "warp_ms":warps_ms,"image_write_ms":image_write_ms,"pre_report_total_ms":total.elapsed().as_secs_f64()*1000.0,
        }),
    )?;
    tracing::info!(path=%output.display(),"Saved calibrated pinhole image");
    Ok(())
}
fn repeat(
    map: &RectificationMap,
    source: &GrayImage,
    iterations: usize,
) -> Result<(GrayImage, Vec<f64>), Error> {
    let mut last = None;
    let mut times = Vec::with_capacity(iterations);
    for _iteration in 0..iterations {
        let start = Instant::now();
        let pixels = map.rectify(source)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        if last.as_ref().is_some_and(|previous| *previous != pixels) {
            return Err(Error::Input(
                "rectification changed across repeated calls".into(),
            ));
        }
        last = Some(pixels);
    }
    Ok((
        last.ok_or_else(|| Error::Input("rectification requires an iteration".into()))?,
        times,
    ))
}
fn save_blocking(output: &Path, image: GrayImage) -> Result<Image, Error> {
    let path = output.join("rectified.png");
    let mut png = Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(image)
        .write_to(&mut png, ImageFormat::Png)
        .map_err(|source| Error::Image {
            path: path.clone(),
            source,
        })?;
    let bytes = png.into_inner();
    assets::write_blocking(&path, &bytes)?;
    Ok(Image {
        path: "rectified.png".into(),
        sha256: hex::encode(Sha256::digest(bytes)),
    })
}

#[cfg(test)]
mod tests;
