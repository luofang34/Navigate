//! Keep each experimental representation bound to its actual encoder input.
use super::geometry;
use crate::{Error, config::Config};
use image::{GrayImage, imageops};
use navigate_visual_onnx::{XFeatFeatures, XFeatStages};
use navigate_visual_workflow::FeatureBackend;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub(super) struct Window {
    pub area: [u32; 4],
    pub crop: XFeatFeatures,
    pub shared: XFeatFeatures,
}
pub(super) struct Block {
    pub rotation: u8,
    pub features: XFeatFeatures,
    pub windows: Vec<Window>,
}
pub(super) struct Cache {
    pub blocks: Vec<Block>,
    pub report: Value,
}

pub(super) fn prepare_blocking(
    backend: &mut XFeatStages,
    image: &GrayImage,
    windows: &[[u32; 4]],
    rotations: &[u8],
    config: &Config,
    raster_bytes: usize,
) -> Result<Cache, Error> {
    let mut blocks = Vec::new();
    let mut reports = Vec::new();
    let mut bytes = raster_bytes;
    for &rotation in rotations {
        let rotated = super::super::matching::rotate(image, rotation);
        let start = Instant::now();
        let features = backend.extract_features_blocking(&rotated)?;
        let block_extraction_ms = start.elapsed().as_secs_f64() * 1000.0;
        check(backend, &features, &rotated, config)?;
        bytes = bytes.saturating_add(features.resident_bytes());
        let start = Instant::now();
        let packet = features.encode();
        let restored = XFeatFeatures::decode(&packet, config.keypoints)?;
        check(backend, &restored, &rotated, config)?;
        if !backend.equivalent_features(&features, &restored) {
            return Err(Error::Input(
                "map packet changes the shared representation".into(),
            ));
        }
        let packet_roundtrip_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut entries = Vec::new();
        let mut window_reports = Vec::new();
        for &area in windows {
            let (entry, report) = prepare_window_blocking(
                backend, image, &rotated, &restored, area, rotation, config,
            )?;
            bytes = bytes
                .saturating_add(entry.crop.resident_bytes())
                .saturating_add(entry.shared.resident_bytes());
            if bytes > config.limits.reference_bytes {
                return Err(Error::Input(
                    "window feature caches exceed reference budget".into(),
                ));
            }
            entries.push(entry);
            window_reports.push(report);
        }
        reports.push(json!({"rotation":rotation,"block_features":features.len(),"block_feature_bytes":features.resident_bytes(),
            "source_to_model":geometry::model_transform(rotated.dimensions()),"block_extraction_ms":block_extraction_ms,"packet_bytes":packet.len(),"packet_roundtrip_ms":packet_roundtrip_ms,
            "packet_sha256":hex::encode(Sha256::digest(&packet)),"windows":window_reports}));
        blocks.push(Block {
            rotation,
            features,
            windows: entries,
        });
    }
    Ok(Cache {
        blocks,
        report: json!({"rotations":reports,"all_arms_resident_bytes_including_raster":bytes,
        "independent_encoder_calls":windows.len()*rotations.len(),"shared_encoder_calls":rotations.len(),
        "shared_subset_copy_calls":windows.len()*rotations.len(),"packet_memory_roundtrip_exact_equivalence":true}),
    })
}
fn prepare_window_blocking(
    backend: &mut XFeatStages,
    image: &GrayImage,
    rotated: &GrayImage,
    parent: &XFeatFeatures,
    area: [u32; 4],
    rotation: u8,
    config: &Config,
) -> Result<(Window, Value), Error> {
    let start = Instant::now();
    let selected = parent.select_pixels(|pixel| {
        geometry::contains_pixel(
            super::super::matching::restore_point(pixel.into(), rotation, image.dimensions()),
            area,
        )
    });
    let select_ms = start.elapsed().as_secs_f64() * 1000.0;
    check(backend, &selected, rotated, config)?;
    let cropped = imageops::crop_imm(image, area[0], area[1], area[2], area[3]).to_image();
    let cropped = super::super::matching::rotate(&cropped, rotation);
    let start = Instant::now();
    let features = backend.extract_features_blocking(&cropped)?;
    let features = features.select_pixels(|pixel| {
        let p = super::super::matching::restore_point(pixel.into(), rotation, (area[2], area[3]));
        geometry::contains_pixel(
            p + nalgebra::Vector2::new(f64::from(area[0]), f64::from(area[1])),
            area,
        )
    });
    let extraction_ms = start.elapsed().as_secs_f64() * 1000.0;
    check(backend, &features, &cropped, config)?;
    let report = json!({"window":area,"crop_features":features.len(),"shared_features":selected.len(),
        "crop_feature_bytes":features.resident_bytes(),"shared_feature_bytes":selected.resident_bytes(),
        "source_to_model":geometry::model_transform(cropped.dimensions()),"valid_window":geometry::valid_area(area),"crop_extraction_ms":extraction_ms,"shared_selection_ms":select_ms});
    Ok((
        Window {
            area,
            crop: features,
            shared: selected,
        },
        report,
    ))
}
fn check(
    backend: &XFeatStages,
    features: &XFeatFeatures,
    image: &GrayImage,
    config: &Config,
) -> Result<(), Error> {
    let metadata = backend.validate_features(features, image)?;
    if metadata.count > config.limits.features_per_image
        || metadata.resident_bytes > config.limits.reference_bytes
    {
        return Err(Error::Input(
            "map features exceed window probe budget".into(),
        ));
    }
    Ok(())
}
