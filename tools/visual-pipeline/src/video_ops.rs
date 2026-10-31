//! Video rotation interpolation, media import, frame extraction, and Navigate debriefs.

mod debrief;

pub use debrief::prepare_navigate_debrief;
pub use navigate_locate::corridor::{kabsch_3d, relative_rotation_ransac};

use crate::{
    PipelineError,
    clock::{PriorTrack, capture_anchor, movie_duration_s, quicktime_udta_tags},
    util::{file_sha256, read_json, run_tool, write_json},
};
use nalgebra::{Matrix3, Rotation3, UnitQuaternion};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// Spherical linear interpolation between two `3 x 3` rotation matrices at `t` in `[0, 1]`.
pub fn slerp_mat3(r0: &Matrix3<f64>, r1: &Matrix3<f64>, t: f64) -> Matrix3<f64> {
    let q0 = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r0));
    let q1 = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r1));
    *q0.slerp(&q1, t.clamp(0.0, 1.0))
        .to_rotation_matrix()
        .matrix()
}

/// Propagate and SLERP-close dense per-frame rotations between Sparse `anchor_rotations`.
pub fn dense_rotations(
    frame_count: usize,
    step_rel: &[Matrix3<f64>],
    anchors: &BTreeMap<usize, Matrix3<f64>>,
) -> Vec<Matrix3<f64>> {
    if frame_count == 0 {
        return Vec::new();
    }
    let mut out = vec![Matrix3::identity(); frame_count];
    let Some((&first_idx, &first_r)) = anchors.iter().next() else {
        for i in 1..frame_count {
            if let Some(rel) = step_rel.get(i - 1) {
                out[i] = rel * out[i - 1];
            }
        }
        return out;
    };

    out[first_idx] = first_r;
    for i in (0..first_idx).rev() {
        if let Some(rel) = step_rel.get(i) {
            out[i] = rel.transpose() * out[i + 1];
        }
    }
    let anchor_keys: Vec<usize> = anchors.keys().copied().collect();
    for pair in anchor_keys.windows(2) {
        let (a0, a1) = (pair[0], pair[1]);
        let mut fwd = vec![out[a0]; a1 - a0 + 1];
        for k in a0..a1 {
            if let Some(rel) = step_rel.get(k) {
                fwd[k - a0 + 1] = rel * fwd[k - a0];
            }
        }
        let target_end = anchors[&a1];
        let drift = target_end * fwd[a1 - a0].transpose();
        let ident = Matrix3::identity();
        for k in a0..=a1 {
            let frac = ((k - a0) as f64) / ((a1 - a0).max(1) as f64);
            let corr = slerp_mat3(&ident, &drift, frac);
            out[k] = corr * fwd[k - a0];
        }
    }
    if let Some(&last_idx) = anchor_keys.last() {
        out[last_idx] = anchors[&last_idx];
        for i in (last_idx + 1)..frame_count {
            if let Some(rel) = step_rel.get(i - 1) {
                out[i] = rel * out[i - 1];
            }
        }
    }
    out
}

/// Linearly interpolate camera optical centres across `(frame_index, eye)` anchors.
pub fn interpolate_eyes(frame_count: usize, anchors: &BTreeMap<usize, [f64; 3]>) -> Vec<[f64; 3]> {
    if frame_count == 0 || anchors.is_empty() {
        return vec![[0.0; 3]; frame_count];
    }
    let keys: Vec<usize> = anchors.keys().copied().collect();
    let mut out = vec![[0.0; 3]; frame_count];
    let first = keys[0];
    for item in out.iter_mut().take(first + 1) {
        *item = anchors[&first];
    }
    for pair in keys.windows(2) {
        let (i0, i1) = (pair[0], pair[1]);
        let (e0, e1) = (anchors[&i0], anchors[&i1]);
        for (offset, slot) in out[i0..=i1].iter_mut().enumerate() {
            let f = (offset as f64) / ((i1 - i0).max(1) as f64);
            *slot = [
                e0[0] + f * (e1[0] - e0[0]),
                e0[1] + f * (e1[1] - e0[1]),
                e0[2] + f * (e1[2] - e0[2]),
            ];
        }
    }
    if let Some(&last) = keys.last() {
        for item in out.iter_mut().skip(last) {
            *item = anchors[&last];
        }
    }
    out
}

/// Compute the median focal length in pixels across accepted keyframe estimates.
pub fn shared_focal(focals: &[f64]) -> Option<f64> {
    let mut valid: Vec<f64> = focals
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 10.0)
        .collect();
    if valid.is_empty() {
        return None;
    }
    valid.sort_by(f64::total_cmp);
    Some(valid[valid.len() / 2])
}

/// Import a raw video container (`import-video`), recording SHA-256, QuickTime clock, and tags.
///
/// # Errors
///
/// Returns [`PipelineError`] when reading the video or writing `source.json` fails.
pub fn import_video_media(video_path: &Path, output_dir: &Path) -> Result<Value, PipelineError> {
    fs::create_dir_all(output_dir).map_err(|source| PipelineError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;
    let sha = file_sha256(video_path)?;
    let anchor = capture_anchor(video_path)?;
    let duration_s = movie_duration_s(video_path)?;
    let udta = quicktime_udta_tags(video_path)?;

    let doc = json!({
        "format": "video-import",
        "version": 1,
        "source_file": video_path.file_name().and_then(|s| s.to_str()).unwrap_or("video.mp4"),
        "source_sha256": sha,
        "duration_s": duration_s,
        "time_anchor": anchor,
        "quicktime_udta": udta,
    });
    write_json(&output_dir.join("source.json"), &doc)?;
    Ok(doc)
}

/// Extract evenly spaced keyframes from `video_path` via `ffmpeg` (`video_testset`).
///
/// # Errors
///
/// Returns [`PipelineError`] when `ffmpeg` fails or no frames are extracted.
pub fn prepare_video_testset(
    video_path: &Path,
    output_dir: &Path,
    fps: f64,
    long_edge: u32,
) -> Result<Value, PipelineError> {
    let img_dir = output_dir.join("images");
    let mask_dir = output_dir.join("masks");
    let prof_dir = output_dir.join("profiles");
    for d in [&img_dir, &mask_dir, &prof_dir] {
        fs::create_dir_all(d).map_err(|source| PipelineError::Io {
            path: d.clone(),
            source,
        })?;
    }
    let src_s = video_path.to_string_lossy().into_owned();
    let pat = img_dir.join("f-%05d.png").to_string_lossy().into_owned();
    let vf = format!(
        "fps={:.4},scale='min({},iw)':-2",
        fps.max(0.05),
        long_edge.max(160)
    );
    run_tool("ffmpeg", &["-y", "-i", &src_s, "-vf", &vf, &pat])?;

    let mut frame_files: Vec<PathBuf> = fs::read_dir(&img_dir)
        .map_err(|source| PipelineError::Io {
            path: img_dir.clone(),
            source,
        })?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    frame_files.sort();
    if frame_files.is_empty() {
        return Err(PipelineError::Invalid(
            "ffmpeg extracted zero video frames".into(),
        ));
    }

    let mut cases = Vec::with_capacity(frame_files.len());
    for (idx, fp) in frame_files.iter().enumerate() {
        let stem = fp
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("f-00001")
            .to_owned();
        let img = image::open(fp).map_err(|source| PipelineError::Image {
            path: fp.clone(),
            source,
        })?;
        let (w, h) = (img.width(), img.height());
        let mp = mask_dir.join(format!("{stem}.png"));
        image::GrayImage::from_pixel(w, h, image::Luma([255u8]))
            .save(&mp)
            .map_err(|source| PipelineError::Image {
                path: mp.clone(),
                source,
            })?;
        cases.push(json!({
            "id": stem,
            "sequence": idx,
            "time_s": (idx as f64) / fps.max(0.05),
            "image": format!("images/{stem}.png"),
            "image_sha256": file_sha256(fp)?,
            "size": [w, h],
            "valid_mask": format!("masks/{stem}.png"),
            "mask_sha256": file_sha256(&mp)?,
            "mask_status": "full_image_unmasked",
            "prior": null,
        }));
    }

    let profile = json!({ "format": "photo-terrain-cases", "version": 1, "cases": cases });
    write_json(&prof_dir.join("no-prior.json"), &profile)?;
    write_json(&prof_dir.join("with-prior.json"), &profile)?;
    write_json(
        &output_dir.join("manifest.json"),
        &json!({
            "format": "video-terrain-testset",
            "version": 1,
            "source_sha256": file_sha256(video_path)?,
            "fps": fps,
            "frames": cases.len(),
        }),
    )?;
    Ok(json!({ "frames": cases.len(), "output": output_dir.display().to_string() }))
}

/// Populate `profiles/with-prior.json` from an `acquisition.json` [`PriorTrack`] (`video_priors`).
///
/// # Errors
///
/// Returns [`PipelineError`] when reading `acquisition.json` or updating the profile fails.
pub fn apply_video_priors(
    dataset_dir: &Path,
    acquisition_json: &Path,
    max_gap_s: f64,
) -> Result<Value, PipelineError> {
    let acq: Value = read_json(acquisition_json)?;
    let track = PriorTrack::from_acquisition(&acq)?
        .ok_or_else(|| PipelineError::Invalid("acquisition.json contains no prior_track".into()))?;
    let profile_path = dataset_dir.join("profiles").join("with-prior.json");
    let mut profile: Value = read_json(&profile_path)?;
    let cases = profile
        .get_mut("cases")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PipelineError::Invalid("missing cases in with-prior.json".into()))?;

    let mut seeded = 0usize;
    for c in cases {
        let t_s = c.get("time_s").and_then(Value::as_f64).unwrap_or(0.0);
        if let Some(entry) = track.nearest(t_s, max_gap_s) {
            c["prior"] = json!({
                "direction_deg": entry.heading_deg,
                "direction_reference": "T",
                "position": {
                    "latitude": entry.latitude,
                    "longitude": entry.longitude,
                    "altitude_m": track.terrain_m + entry.agl_m,
                    "unsigned_altitude_m": (track.terrain_m + entry.agl_m).abs(),
                    "altitude_datum": "prior_track_unverified",
                }
            });
            seeded += 1;
        }
    }
    write_json(&profile_path, &profile)?;
    Ok(json!({ "seeded_frames": seeded }))
}
