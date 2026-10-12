//! Counterfactual surface checks with frozen image correspondences.
use super::job::SurfaceJob;
use crate::{Error, assets, config::Config, job::Image};
use nalgebra::{Matrix2, Vector2};
use navigate_recording::Pose;
use navigate_visual::{
    Frame, LocalizerConfig, MapRevision, OrthographicReference, PixelMatch, PosePrior, PoseVerifier,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    image_job: Image,
    height_job: Image,
    match_result: Image,
    registration: String,
}
fn verified_blocking<T: serde::de::DeserializeOwned>(
    root: &Path,
    item: &Image,
) -> Result<(T, PathBuf), Error> {
    let path = root.join(&item.path);
    let bytes = assets::read_blocking(&path, 64 * 1024 * 1024)?;
    assets::verify(&bytes, &item.sha256)?;
    let value = serde_json::from_slice(&bytes).map_err(|source| Error::Json {
        path: path.clone(),
        source,
    })?;
    Ok((value, path))
}

pub fn run_blocking(config: &Config, input: &Path, output: &Path) -> Result<(), Error> {
    let root = input.parent().unwrap_or(Path::new("."));
    let input: Input = assets::read_json_blocking(input)?;
    if input.registration.trim().is_empty() {
        return Err(Error::Input(
            "surface probe requires a registration declaration".into(),
        ));
    }
    let (job, job_path): (SurfaceJob, _) = verified_blocking(root, &input.image_job)?;
    let (height_job, height_path): (SurfaceJob, _) = verified_blocking(root, &input.height_job)?;
    let (result, _): (Value, _) = verified_blocking(root, &input.match_result)?;
    let (frame, reference, prior) = job.load_blocking(job_path.parent().unwrap_or(root), config)?;
    let (_, height, _) = height_job.load_blocking(height_path.parent().unwrap_or(root), config)?;
    if result["observation_sha256"].as_str() != Some(&frame.evidence_sha256()) {
        return Err(Error::Input(
            "frozen matches do not identify this query and calibration".into(),
        ));
    }
    let start = Instant::now();
    let alternative = remap(&reference, &height)?;
    let remap_ms = start.elapsed().as_secs_f64() * 1000.0;
    let candidates = result["iterations"][0]["candidates"]
        .as_array()
        .ok_or_else(|| Error::Input("missing frozen candidates".into()))?;
    if candidates.len() > config.limits.candidates {
        return Err(Error::Input("frozen candidate budget".into()));
    }
    let rows = decisions(
        &frame,
        &prior,
        &reference,
        &alternative,
        candidates,
        &root.join(&input.match_result.path),
    )?;
    assets::write_json_blocking(
        output,
        &json!({"image_job":input.image_job,"height_job":input.height_job,"match_result":input.match_result,"registration":input.registration,"rows":rows,"remap_ms":remap_ms,"invalid_height_cells":alternative.heights_m.iter().filter(|v|!v.is_finite()).count(),"total_height_cells":alternative.heights_m.len(),"remapped_map":alternative.map.manifest_sha256,"source_grid":height_job.reference,"target_grid":job.reference,"scope":"Frozen image matches, nearest height at common metric XY. Native and alternative use the same verifier, camera and prior. No new matching work or independent geographic accuracy is measured."}),
    )?;
    tracing::info!(path=%output.display(), "Saved frozen-match surface probe");
    Ok(())
}
fn decisions(
    frame: &Frame,
    prior: &PosePrior,
    reference: &OrthographicReference,
    alternative: &OrthographicReference,
    candidates: &[Value],
    evidence: &Path,
) -> Result<Vec<Value>, Error> {
    let verifier = PoseVerifier::new(LocalizerConfig::default())?;
    let mut rows = Vec::new();
    for candidate in candidates {
        let values: Vec<[f64; 4]> = serde_json::from_value(candidate["correspondences"].clone())
            .map_err(|source| Error::Json {
                path: evidence.into(),
                source,
            })?;
        if values.len() > 100_000 {
            return Err(Error::Input("frozen match budget".into()));
        }
        let pairs: Vec<_> = values
            .iter()
            .map(|p| PixelMatch {
                reference: [p[0], p[1]].into(),
                query: [p[2], p[3]].into(),
            })
            .collect();
        let native = verify(&verifier, frame, reference, prior, &pairs);
        if native["accepted"] != candidate["accepted"] {
            return Err(Error::Input(
                "native replay differs from recorded acceptance".into(),
            ));
        }
        rows.push(json!({"reference_quarter_turns":candidate["reference_quarter_turns"], "matches":pairs.len(), "native":native, "alternative":verify(&verifier,frame,alternative,prior,&pairs)}));
    }
    Ok(rows)
}
fn verify(
    verifier: &PoseVerifier,
    frame: &Frame,
    reference: &OrthographicReference,
    prior: &PosePrior,
    pairs: &[PixelMatch],
) -> Value {
    let start = Instant::now();
    let decision =
        verifier.verify_orthographic(frame, reference, prior, pairs, "frozen-surface-probe");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    match decision {
        Ok(e) => {
            json!({"accepted":true,"geometry_ms":ms,"pose":Pose::from(e.pose),"quality":{"inliers":e.quality.inliers,"spatial_support":e.quality.spatial_support,"occupied_cells":e.quality.occupied_cells,"rms_px":e.quality.reprojection_rms_px}})
        }
        Err(e) => json!({"accepted":false,"geometry_ms":ms,"rejection":e.to_string()}),
    }
}
fn remap(
    target: &OrthographicReference,
    source: &OrthographicReference,
) -> Result<OrthographicReference, Error> {
    source.validate()?;
    target.validate()?;
    let inverse = Matrix2::from_columns(&[source.geometry.column, source.geometry.row])
        .try_inverse()
        .ok_or_else(|| Error::Input("source height basis".into()))?;
    let mut heights = Vec::with_capacity(target.heights_m.len());
    for y in 0..target.image.height() {
        for x in 0..target.image.width() {
            let xy = target.geometry.origin
                + target.geometry.column * f64::from(x)
                + target.geometry.row * f64::from(y);
            let p: Vector2<f64> = inverse * (xy - source.geometry.origin);
            heights.push(source.world_at(p).map_or(f32::NAN, |p| p.z as f32));
        }
    }
    let mut hash = Sha256::new();
    hash.update(b"diagnostic-height-remap-nearest-v1");
    hash.update(target.map.manifest_sha256.as_bytes());
    hash.update(source.map.manifest_sha256.as_bytes());
    for h in &heights {
        hash.update(h.to_le_bytes());
    }
    Ok(OrthographicReference {
        map: MapRevision {
            release_id: "diagnostic-height-remap".into(),
            manifest_sha256: hex::encode(hash.finalize()),
        },
        image: target.image.clone(),
        heights_m: heights,
        geometry: target.geometry,
    })
}
#[cfg(test)]
mod tests;
