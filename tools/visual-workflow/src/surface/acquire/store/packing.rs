//! Package verification preserves exact host feature values.
use super::{Entry, hex};
use crate::{Error, assets, surface::matching::rotate};
use image::GrayImage;
use navigate_visual_onnx::{XFeatFeatures, XFeatStages};
use navigate_visual_workflow::FeatureBackend;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};
pub(super) struct Request<'a> {
    pub image: &'a GrayImage,
    pub rotation: u8,
    pub key: [u8; 32],
    pub base_bytes: usize,
    pub byte_limit: usize,
    pub feature_limit: usize,
    pub feature_budget: usize,
}
pub(super) fn write_blocking(
    backend: &mut XFeatStages,
    request: Request<'_>,
    directory: &Path,
) -> Result<(Entry, Value), Error> {
    let started = Instant::now();
    let raster_bytes = request
        .base_bytes
        .saturating_add(request.image.as_raw().len());
    if raster_bytes.saturating_add(4 * 1024 * 1024) > request.byte_limit {
        return Err(Error::Input(
            "preparation raster and packet reservation exceed budget".into(),
        ));
    }
    let rotated = rotate(request.image, request.rotation);
    let features = backend.extract_features_blocking(&rotated)?;
    let info = backend.validate_features(&features, &rotated)?;
    if info.count > request.feature_budget {
        return Err(Error::Input(
            "prepared feature count exceeds configured limit".into(),
        ));
    }
    let encoding_ms = started.elapsed().as_secs_f64() * 1000.;
    let started = Instant::now();
    let bytes = features.encode();
    let decoded = XFeatFeatures::decode(&bytes, request.feature_limit)?;
    if features != decoded {
        return Err(Error::Input(
            "feature packet round trip changes values".into(),
        ));
    }
    let peak = raster_bytes
        .saturating_add(info.resident_bytes)
        .saturating_add(bytes.capacity())
        .saturating_add(decoded.resident_bytes());
    if peak > request.byte_limit {
        return Err(Error::Input(
            "preparation features and packet exceed budget".into(),
        ));
    }
    drop(decoded);
    drop(features);
    drop(rotated);
    let validation_ms = started.elapsed().as_secs_f64() * 1000.;
    let started = Instant::now();
    let path = directory.join(format!("{}.navfeat", hex(&request.key)));
    assets::write_blocking(&path, &bytes)?;
    let readback = assets::read_blocking(&path, 2 * 1024 * 1024)?;
    if readback != bytes {
        return Err(Error::Input(format!(
            "feature packet readback differs: {}",
            path.display()
        )));
    }
    let peak = peak.max(request.base_bytes + bytes.capacity() + readback.capacity());
    if peak > request.byte_limit {
        return Err(Error::Input(
            "preparation packet readback exceeds budget".into(),
        ));
    }
    let entry = Entry {
        key: request.key,
        sha256: Sha256::digest(&bytes).into(),
        bytes: bytes.len(),
    };
    Ok((
        entry,
        json!({"key":hex(&request.key),"rotation":request.rotation,"features":info.count,"packet_bytes":bytes.len(),"encoding_ms":encoding_ms,"packet_round_trip_ms":validation_ms,"write_readback_ms":started.elapsed().as_secs_f64()*1000.,"peak_accounted_bytes":peak,"exact_round_trip":true}),
    ))
}
