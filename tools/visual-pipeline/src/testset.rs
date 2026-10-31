//! Photo test-set preparation, flight grouping, and unit-sphere prior interpolation.

mod priors;

pub use priors::interpolate_flight_priors;

use crate::{
    PipelineError,
    metadata::{PhotoClaims, decode_oriented_rgb, extract_photo_claims},
    util::{file_sha256, parse_iso8601_ms, write_contact_sheets, write_json},
};
use image::{GrayImage, Luma, imageops::FilterType};
use priors::great_circle_m;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub(crate) const MAX_FLIGHT_GAP_S: f64 = 7200.0;
const MAX_FLIGHT_DISTANCE_M: f64 = 250_000.0;
pub(crate) const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Flight group entry inside `manifest.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlightGroup {
    /// Grouping basis description.
    pub basis: String,
    /// Ordered photo identifiers in this flight.
    pub photos: Vec<String>,
}

/// Prepared case entry inside `profiles/with-prior.json` or `profiles/no-prior.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhotoCase {
    /// Deterministic photo identifier (`"q-" + sha256[..20]`).
    pub id: String,
    /// Relative path to the upright PNG image.
    pub image: String,
    /// SHA-256 digest of `image`.
    pub image_sha256: String,
    /// Image dimensions `[width, height]`.
    pub size: [u32; 2],
    /// Relative path to the valid-pixel mask PNG.
    #[serde(default)]
    pub valid_mask: String,
    /// SHA-256 digest of `valid_mask`.
    #[serde(default)]
    pub mask_sha256: String,
    /// Mask provenance status.
    #[serde(default)]
    pub mask_status: String,
    /// Optional semantic label image relative path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_labels: Option<String>,
    /// Optional semantic label SHA-256 digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_sha256: Option<String>,
    /// Optional EXIF or interpolated search prior.
    #[serde(default)]
    pub prior: Option<PhotoClaims>,
    /// Optional 0-indexed frame sequence number for sequential video frames.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<u64>,
    /// Optional capture time in seconds from video start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_s: Option<f64>,
}

/// Prepare a photo test set in `output_dir` from photos in `source_dir`.
///
/// Refuses to write inside `source_dir` so raw photos stay untouched.
///
/// # Errors
///
/// Returns [`PipelineError`] when `output_dir` is inside `source_dir`, no photos exist,
/// or image decoding or file writing fails.
pub fn prepare_photo_testset(
    source_dir: &Path,
    output_dir: &Path,
    long_edge: u32,
) -> Result<Value, PipelineError> {
    if long_edge < 64 {
        return Err(PipelineError::Invalid(
            "long_edge must be at least 64 pixels".into(),
        ));
    }
    let src_abs = source_dir
        .canonicalize()
        .map_err(|source| PipelineError::Io {
            path: source_dir.to_path_buf(),
            source,
        })?;
    fs::create_dir_all(output_dir).map_err(|source| PipelineError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;
    let out_abs = output_dir
        .canonicalize()
        .map_err(|source| PipelineError::Io {
            path: output_dir.to_path_buf(),
            source,
        })?;
    if out_abs.starts_with(&src_abs) {
        return Err(PipelineError::Invalid(
            "output_dir must not reside inside source_dir".into(),
        ));
    }

    for sub in ["images", "masks", "thumbnails", "metadata", "profiles"] {
        let dir = output_dir.join(sub);
        fs::create_dir_all(&dir).map_err(|source| PipelineError::Io { path: dir, source })?;
    }

    let mut photo_paths = collect_photo_files(&src_abs)?;
    photo_paths.sort();
    if photo_paths.is_empty() {
        return Err(PipelineError::Invalid(format!(
            "no supported photos found in {}",
            source_dir.display()
        )));
    }

    let mut items = Vec::with_capacity(photo_paths.len());
    let mut claims_by_id = Vec::with_capacity(photo_paths.len());
    let mut ids = Vec::with_capacity(photo_paths.len());

    for src_path in &photo_paths {
        let src_sha = file_sha256(src_path)?;
        let id = format!("q-{}", &src_sha[..20]);
        let (raw_tags, claims, orientation) = extract_photo_claims(src_path);
        let decoded = decode_oriented_rgb(src_path, orientation)?;
        let (w0, h0) = decoded.dimensions();
        let max_dim = w0.max(h0).max(1);
        let resized = if max_dim > long_edge {
            let scale = f64::from(long_edge) / f64::from(max_dim);
            let nw = ((f64::from(w0) * scale).round() as u32).max(1);
            let nh = ((f64::from(h0) * scale).round() as u32).max(1);
            image::imageops::resize(&decoded, nw, nh, FilterType::Lanczos3)
        } else {
            decoded
        };
        let (w, h) = resized.dimensions();

        let img_rel = format!("images/{id}.png");
        let img_path = output_dir.join(&img_rel);
        resized
            .save(&img_path)
            .map_err(|source| PipelineError::Image {
                path: img_path.clone(),
                source,
            })?;
        let img_sha = file_sha256(&img_path)?;

        let mask_rel = format!("masks/{id}.png");
        let mask_path = output_dir.join(&mask_rel);
        let mask_img = GrayImage::from_pixel(w, h, Luma([255u8]));
        mask_img
            .save(&mask_path)
            .map_err(|source| PipelineError::Image {
                path: mask_path.clone(),
                source,
            })?;
        let mask_sha = file_sha256(&mask_path)?;

        let thumb_scale = 320.0 / f64::from(w.max(h).max(1));
        let tw = ((f64::from(w) * thumb_scale).round() as u32).max(1);
        let th = ((f64::from(h) * thumb_scale).round() as u32).max(1);
        let thumb = image::imageops::resize(&resized, tw, th, FilterType::Triangle);
        let thumb_path = output_dir.join("thumbnails").join(format!("{id}.jpg"));
        thumb
            .save(&thumb_path)
            .map_err(|source| PipelineError::Image {
                path: thumb_path,
                source,
            })?;

        let rel_source = src_path
            .strip_prefix(&src_abs)
            .unwrap_or(src_path)
            .display()
            .to_string();
        let folder = src_path
            .parent()
            .and_then(|p| p.strip_prefix(&src_abs).ok())
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        let meta_doc = json!({
            "id": id,
            "source_file": rel_source,
            "source_sha256": src_sha,
            "size": [w, h],
            "claims": claims,
            "exif": raw_tags,
        });
        write_json(
            &output_dir.join("metadata").join(format!("{id}.json")),
            &meta_doc,
        )?;

        ids.push(id.clone());
        claims_by_id.push((id.clone(), folder, claims.clone()));
        items.push(PhotoCase {
            id,
            image: img_rel,
            image_sha256: img_sha,
            size: [w, h],
            valid_mask: mask_rel,
            mask_sha256: mask_sha,
            mask_status: "full_image_unmasked".into(),
            semantic_labels: None,
            semantic_sha256: None,
            prior: Some(claims),
            sequence: None,
            time_s: None,
        });
    }

    let groups = group_flights(&claims_by_id);
    let with_prior = build_cases(&items, true);
    let no_prior = build_cases(&items, false);
    write_json(
        &output_dir.join("profiles").join("with-prior.json"),
        &json!({ "format": "photo-terrain-cases", "version": 1, "cases": with_prior }),
    )?;
    write_json(
        &output_dir.join("profiles").join("no-prior.json"),
        &json!({ "format": "photo-terrain-cases", "version": 1, "cases": no_prior }),
    )?;

    let manifest = json!({
        "format": "photo-terrain-testset",
        "version": 1,
        "photos": items,
        "capture_groups": groups,
    });
    write_json(&output_dir.join("manifest.json"), &manifest)?;
    write_contact_sheets(output_dir, &ids)?;

    Ok(json!({
        "photos": items.len(),
        "flights": groups.len(),
        "output": output_dir.display().to_string(),
    }))
}

/// Group photos into flights by folder, camera model, exposure gap, and great-circle distance.
pub fn group_flights(entries: &[(String, String, PhotoClaims)]) -> Vec<FlightGroup> {
    #[derive(Clone)]
    struct Row {
        id: String,
        folder: String,
        camera: String,
        time_ms: Option<i64>,
        pos: Option<(f64, f64)>,
    }

    let mut rows: Vec<Row> = entries
        .iter()
        .map(|(id, folder, claims)| Row {
            id: id.clone(),
            folder: folder.clone(),
            camera: claims.camera_model.clone().unwrap_or_default(),
            time_ms: claims.capture_time.as_deref().and_then(parse_iso8601_ms),
            pos: claims.position.as_ref().map(|p| (p.latitude, p.longitude)),
        })
        .collect();

    rows.sort_by(|a, b| {
        a.folder
            .cmp(&b.folder)
            .then_with(|| a.camera.cmp(&b.camera))
            .then_with(|| a.time_ms.cmp(&b.time_ms))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut groups: Vec<FlightGroup> = Vec::new();
    let mut prev: Option<Row> = None;
    for row in rows {
        let same_group = if let Some(p) = &prev {
            let same_cam = p.folder == row.folder && p.camera == row.camera;
            let close_time = match (p.time_ms, row.time_ms) {
                (Some(t0), Some(t1)) => ((t1 - t0).abs() as f64) <= MAX_FLIGHT_GAP_S * 1000.0,
                _ => false,
            };
            let close_dist = match (p.pos, row.pos) {
                (Some((lat0, lon0)), Some((lat1, lon1))) => {
                    great_circle_m(lat0, lon0, lat1, lon1) <= MAX_FLIGHT_DISTANCE_M
                }
                _ => true,
            };
            same_cam && close_time && close_dist
        } else {
            false
        };

        if same_group && let Some(last) = groups.last_mut() {
            last.photos.push(row.id.clone());
        } else {
            groups.push(FlightGroup {
                basis: "source folder, camera and exposure time".into(),
                photos: vec![row.id.clone()],
            });
        }
        prev = Some(row);
    }
    groups
}

/// Build profile cases either keeping priors (`include_prior = true`) or stripping them.
pub fn build_cases(items: &[PhotoCase], include_prior: bool) -> Vec<PhotoCase> {
    items
        .iter()
        .map(|c| {
            let mut copy = c.clone();
            if !include_prior {
                copy.prior = None;
            }
            copy
        })
        .collect()
}

fn collect_photo_files(dir: &Path) -> Result<Vec<PathBuf>, PipelineError> {
    let mut out = Vec::new();
    let entries = fs::read_dir(dir).map_err(|source| PipelineError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| PipelineError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir() {
            out.extend(collect_photo_files(&path)?);
        } else if let Some(ext) = path.extension().and_then(|s| s.to_str())
            && matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "heic" | "heif" | "tif" | "tiff"
            )
        {
            out.push(path);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn photo_case_preserves_sequence_and_time_s() {
        let json_str = r#"{
            "id": "case-01",
            "image": "img.png",
            "image_sha256": "abc",
            "size": [800, 600],
            "sequence": 42,
            "time_s": 1.4
        }"#;
        let case: PhotoCase = serde_json::from_str(json_str).expect("deserialize");
        assert_eq!(case.sequence, Some(42));
        assert_eq!(case.time_s, Some(1.4));

        let reserialized = serde_json::to_string(&case).expect("serialize");
        assert!(reserialized.contains("\"sequence\":42"));
        assert!(reserialized.contains("\"time_s\":1.4"));
    }

    #[test]
    fn photo_case_deserializes_without_timestamps() {
        let json_str = r#"{
            "id": "case-02",
            "image": "img.png",
            "image_sha256": "abc",
            "size": [800, 600]
        }"#;
        let case: PhotoCase = serde_json::from_str(json_str).expect("deserialize");
        assert_eq!(case.sequence, None);
        assert_eq!(case.time_s, None);
    }
}
