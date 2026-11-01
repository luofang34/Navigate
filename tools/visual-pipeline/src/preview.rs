//! `sokoly-debrief` v1 preview, `publication.json` inventory, anchor freeze, and reproduction.

use crate::{
    PipelineError,
    pose::load_candidate_pose,
    surface::Surface,
    util::{file_sha256, read_json, run_tool, write_json},
};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// ENU-to-Cesium/Sokoly globe basis matrix `[[0, 1, 0], [0, 0, 1], [1, 0, 0]]`.
pub const TO_GLOBE: [[f64; 3]; 3] = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];

/// Published file record inside `publication.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PublishedFile {
    /// Logical asset group (`"dataset"`, `"catalog"`, `"baseline"`, `"code"`).
    pub group: String,
    /// Relative path inside the reproduction bundle.
    pub path: String,
    /// Hex SHA-256 digest of the file.
    pub sha256: String,
    /// File length in bytes.
    pub bytes: u64,
    /// Public download URL.
    pub url: String,
}

/// Compute `[lon, lat]` ground footprint corners for a candidate pose on `surface`.
pub fn map_corners(
    surface: &Surface,
    eye: [f64; 3],
    rot: &Matrix3<f64>,
    k: &Matrix3<f64>,
    size: [u32; 2],
) -> Option<[[f64; 2]; 4]> {
    let (_, depth, pts) = surface.render(eye, rot, k, [64, 48]);
    let sx = 63.0 / f64::from(size[0].max(1));
    let sy = 47.0 / f64::from(size[1].max(1));
    let _ = (sx, sy);
    let corners_px = [(0u32, 0u32), (63, 0), (63, 47), (0, 47)];
    let mut out = [[0.0f64; 2]; 4];
    for (i, &(cx, cy)) in corners_px.iter().enumerate() {
        let idx = (cy * 64 + cx) as usize;
        if depth[idx] <= 0.5 {
            let [lat, lon, _] = surface.geodetic(eye);
            let d = 0.005;
            out[i] = match i {
                0 => [lon - d, lat + d],
                1 => [lon + d, lat + d],
                2 => [lon + d, lat - d],
                _ => [lon - d, lat - d],
            };
        } else {
            let [lat, lon, _] = surface.geodetic(pts[idx]);
            out[i] = [lon, lat];
        }
    }
    Some(out)
}

/// Convert ENU pose `(eye, world_to_cam)` to Sokoly globe view JSON (`TO_GLOBE` frame).
pub fn globe_view(surface: &Surface, eye: [f64; 3], world_to_cam: &Matrix3<f64>) -> Value {
    let [lat, lon, alt] = surface.geodetic(eye);
    let to_g = Matrix3::from_rows(&[
        Vector3::from(TO_GLOBE[0]).transpose(),
        Vector3::from(TO_GLOBE[1]).transpose(),
        Vector3::from(TO_GLOBE[2]).transpose(),
    ]);
    let cam_to_globe = to_g * world_to_cam.transpose();
    let fwd = cam_to_globe * Vector3::new(0.0, 0.0, 1.0);
    let up = cam_to_globe * Vector3::new(0.0, -1.0, 0.0);
    json!({
        "latitude": lat,
        "longitude": lon,
        "altitude_m": alt,
        "forward_globe": [fwd.x, fwd.y, fwd.z],
        "up_globe": [up.x, up.y, up.z],
    })
}

/// Export `debrief.json` (`sokoly-debrief` v1) and `replay.json` (v3) into `output_dir`.
///
/// # Errors
///
/// Returns [`PipelineError`] when reading run outputs, catalogs, or writing JSON fails.
pub fn export_preview(
    dataset_dir: &Path,
    catalog_path: &Path,
    run_dirs: &[PathBuf],
    output_dir: &Path,
) -> Result<Value, PipelineError> {
    let surface = Surface::from_catalog(catalog_path)?;
    fs::create_dir_all(output_dir).map_err(|source| PipelineError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let mut items = BTreeMap::new();
    for run_dir in run_dirs {
        let same_cat = read_json::<Value>(&run_dir.join("run.json"))
            .ok()
            .and_then(|r| r["catalog_sha256"].as_str().map(str::to_owned))
            .as_deref()
            == Some(surface.catalog_sha256.as_str());
        let Ok(entries) = fs::read_dir(run_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json")
                || path.file_name().and_then(|s| s.to_str()) == Some("run.json")
            {
                continue;
            }
            let doc: Value = read_json(&path)?;
            let Some(id) = doc.get("id").and_then(Value::as_str) else {
                continue;
            };
            let status = doc
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unresolved");
            let npz = run_dir.join(id).join("candidate.npz");
            let mut item = json!({
                "id": id,
                "status": status,
                "image": format!("images/{id}.png"),
                "candidate": doc.get("candidate").cloned().unwrap_or(Value::Null),
            });
            if status == "candidate_requires_review"
                && npz.is_file()
                && let Ok((mut eye, rot, k, sz)) = load_candidate_pose(&npz)
            {
                if !same_cat
                    && let Some(c) = doc.get("candidate")
                    && let (Some(lat), Some(lon), Some(alt)) = (
                        c["latitude"].as_f64(),
                        c["longitude"].as_f64(),
                        c["altitude_m"].as_f64(),
                    )
                {
                    eye = surface.local(lat, lon, alt);
                }
                item["globe_view"] = globe_view(&surface, eye, &rot);
                item["corners"] = json!(map_corners(&surface, eye, &rot, &k, sz));
            }
            items.insert(id.to_owned(), item);
        }
    }

    let frames: Vec<Value> = items.values().cloned().collect();
    let accepted = frames
        .iter()
        .filter(|f| f.get("status").and_then(Value::as_str) == Some("candidate_requires_review"))
        .count();

    let debrief = json!({
        "format": "sokoly-debrief",
        "version": 1,
        "dataset": dataset_dir.display().to_string(),
        "catalog_sha256": surface.catalog_sha256,
        "summary": {
            "total": frames.len(),
            "accepted": accepted,
            "unresolved": frames.len().saturating_sub(accepted),
        },
        "items": frames,
    });
    let replay = json!({
        "version": 3,
        "anchor_lat_lon": [surface.origin_lat, surface.origin_lon],
        "frames": frames,
    });
    write_json(&output_dir.join("debrief.json"), &debrief)?;
    write_json(&output_dir.join("replay.json"), &replay)?;
    Ok(debrief["summary"].clone())
}

/// Freeze accepted candidate poses from `run_dirs` into `output_dir` (`photo_freeze`).
///
/// # Errors
///
/// Returns [`PipelineError`] when reading or copying candidate artifacts fails.
pub fn freeze_anchors(run_dirs: &[PathBuf], output_dir: &Path) -> Result<Value, PipelineError> {
    fs::create_dir_all(output_dir).map_err(|source| PipelineError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;
    let mut frozen = Vec::new();
    for run_dir in run_dirs {
        let run_doc: Value = read_json(&run_dir.join("run.json"))
            .unwrap_or_else(|_| json!({ "catalog_sha256": "unknown" }));
        let cat_sha = run_doc
            .get("catalog_sha256")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let dst_run = output_dir.join(cat_sha);
        fs::create_dir_all(&dst_run).map_err(|source| PipelineError::Io {
            path: dst_run.clone(),
            source,
        })?;
        write_json(
            &dst_run.join("run.json"),
            &json!({ "catalog_sha256": cat_sha }),
        )?;

        let Ok(entries) = fs::read_dir(run_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) != Some("json")
                || p.file_name().and_then(|s| s.to_str()) == Some("run.json")
            {
                continue;
            }
            let doc: Value = read_json(&p)?;
            if doc.get("status").and_then(Value::as_str) != Some("candidate_requires_review") {
                continue;
            }
            let Some(id) = doc.get("id").and_then(Value::as_str) else {
                continue;
            };
            let src_npz = run_dir.join(id).join("candidate.npz");
            if src_npz.is_file() {
                let dst_case = dst_run.join(id);
                fs::create_dir_all(&dst_case).map_err(|source| PipelineError::Io {
                    path: dst_case.clone(),
                    source,
                })?;
                fs::copy(&src_npz, dst_case.join("candidate.npz")).map_err(|source| {
                    PipelineError::Io {
                        path: dst_case.join("candidate.npz"),
                        source,
                    }
                })?;
                write_json(&dst_run.join(format!("{id}.json")), &doc)?;
                frozen.push(id.to_owned());
            }
        }
    }
    frozen.sort();
    frozen.dedup();
    Ok(json!({ "frozen_cases": frozen.len(), "output": output_dir.display().to_string() }))
}

/// Build `publication.json` with SHA-256 digests and verify no raw source files leak (`photo_publication`).
///
/// # Errors
///
/// Returns [`PipelineError`] when forbidden files exist or hashing fails.
pub fn export_publication(
    bundle_files_dir: &Path,
    base_url: &str,
    output_json: &Path,
) -> Result<Value, PipelineError> {
    let mut rel_files = Vec::new();
    collect_bundle_files(bundle_files_dir, bundle_files_dir, &mut rel_files)?;
    rel_files.sort();

    let base = base_url.trim_end_matches('/');
    let mut records = Vec::with_capacity(rel_files.len());
    let mut catalogs = Vec::new();
    for rel in rel_files {
        if rel.starts_with("source/") || rel.contains("source-not-published") {
            return Err(PipelineError::Invalid(format!(
                "refusing to publish raw source path: {rel}"
            )));
        }
        let full = bundle_files_dir.join(&rel);
        let bytes = fs::metadata(&full)
            .map_err(|source| PipelineError::Io {
                path: full.clone(),
                source,
            })?
            .len();
        let sha256 = file_sha256(&full)?;
        if rel.ends_with("/catalog.json") || rel == "catalog.json" {
            catalogs.push(rel.clone());
        }
        let group = if rel.contains("/reference") {
            "catalog"
        } else if rel.contains("/baseline/") {
            "baseline"
        } else {
            "dataset"
        };
        records.push(PublishedFile {
            group: group.into(),
            url: format!("{base}/{rel}"),
            path: rel,
            sha256,
            bytes,
        });
    }

    let doc = json!({
        "format": "photo-publication",
        "version": 1,
        "catalogs": catalogs,
        "files": records,
    });
    write_json(output_json, &doc)?;
    Ok(json!({ "files": records.len(), "catalogs": catalogs.len() }))
}

/// Hydrate and SHA-256-verify files listed in `publication.json` into `target_dir` (`photo_reproduce`).
///
/// Optional `prefix_filter` downloads only matching relative paths so tests stay bounded.
///
/// # Errors
///
/// Returns [`PipelineError`] when downloading fails or any SHA-256 digest mismatches.
pub fn reproduce_bundle(
    publication_json: &Path,
    local_files_root: Option<&Path>,
    target_dir: &Path,
    prefix_filter: Option<&str>,
    limit: Option<usize>,
) -> Result<Value, PipelineError> {
    let doc: Value = read_json(publication_json)?;
    let files: Vec<PublishedFile> =
        serde_json::from_value(doc["files"].clone()).map_err(|source| PipelineError::Json {
            context: "publication.files".into(),
            source,
        })?;

    let mut verified = 0usize;
    for item in &files {
        if let Some(pref) = prefix_filter
            && !item.path.starts_with(pref)
        {
            continue;
        }
        if let Some(max_n) = limit
            && verified >= max_n
        {
            break;
        }
        let rel = Path::new(&item.path);
        if rel.is_absolute()
            || rel
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(PipelineError::Invalid(format!(
                "unsafe path in publication.json: {}",
                item.path
            )));
        }
        let dst = target_dir.join(rel);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let already_ok = dst.is_file() && file_sha256(&dst).ok().as_deref() == Some(&item.sha256);
        if !already_ok {
            let local_src = local_files_root.map(|r| r.join(rel));
            let local_valid = local_src.as_ref().is_some_and(|p| {
                p.is_file() && file_sha256(p).ok().as_deref() == Some(&item.sha256)
            });
            if let (true, Some(src)) = (local_valid, local_src) {
                fs::copy(src, &dst).map_err(|source| PipelineError::Io {
                    path: dst.clone(),
                    source,
                })?;
            } else if let Some(local_url) = item.url.strip_prefix("file://") {
                fs::copy(local_url, &dst).map_err(|source| PipelineError::Io {
                    path: dst.clone(),
                    source,
                })?;
            } else {
                let dst_s = dst.to_string_lossy().into_owned();
                run_tool("curl", &["-fsSL", &item.url, "-o", &dst_s])?;
            }
            let actual = file_sha256(&dst)?;
            if actual != item.sha256 {
                drop(fs::remove_file(&dst));
                return Err(PipelineError::Invalid(format!(
                    "SHA-256 mismatch for {}: expected {}, got {actual}",
                    item.path, item.sha256
                )));
            }
        }
        verified += 1;
    }

    Ok(json!({
        "verified_files": verified,
        "target_dir": target_dir.display().to_string(),
    }))
}

fn collect_bundle_files(
    root: &Path,
    cur: &Path,
    out: &mut Vec<String>,
) -> Result<(), PipelineError> {
    let entries = fs::read_dir(cur).map_err(|source| PipelineError::Io {
        path: cur.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| PipelineError::Io {
            path: cur.to_path_buf(),
            source,
        })?;
        let p = entry.path();
        if p.is_dir() {
            collect_bundle_files(root, &p, out)?;
        } else if let Ok(rel) = p.strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}
