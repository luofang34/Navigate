//! Ground footprint suitability, perspective mosaic composition, and multi-flight survey debrief.

use crate::{
    PipelineError,
    util::{read_json, write_json},
};
use image::{RgbImage, Rgba, RgbaImage};
use nalgebra::Matrix3;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

const MIN_COS_GRAZING: f64 = 0.342_020_143_325_668_8; // cos(70 deg)

/// Check whether four `[lon, lat]` corners form a suitable non-grazing convex ground quad.
///
/// Rejects empty, non-finite, self-intersecting, or steep grazing (`cos < cos(70 deg)`) footprints.
pub fn suitability(
    corners: &[[f64; 2]; 4],
    cos_incidence: Option<f64>,
) -> Result<(), &'static str> {
    if !corners.iter().all(|c| c[0].is_finite() && c[1].is_finite()) {
        return Err("non-finite corner coordinates");
    }
    if let Some(ci) = cos_incidence
        && ci < MIN_COS_GRAZING
    {
        return Err("grazing angle exceeds 70 degrees");
    }
    let mut sign = 0.0f64;
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let c = corners[(i + 2) % 4];
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() < 1e-14 {
            return Err("degenerate footprint edge");
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return Err("non-convex or twisted footprint");
        }
    }
    Ok(())
}

/// Compose a feathered perspective ground mosaic (`mosaic.png`) from located frames (`ground_mosaic`).
///
/// # Errors
///
/// Returns [`PipelineError`] when reading frame images or saving `output_png` fails.
pub fn compose_ground_mosaic(
    frames: &[(RgbImage, [[f64; 2]; 4])],
    canvas_width: u32,
    output_png: &Path,
) -> Result<Value, PipelineError> {
    let valid: Vec<&(RgbImage, [[f64; 2]; 4])> = frames
        .iter()
        .filter(|(_, c)| suitability(c, None).is_ok())
        .collect();
    if valid.is_empty() {
        return Err(PipelineError::Invalid(
            "no suitable convex footprints for ground mosaic".into(),
        ));
    }

    let mut west = f64::INFINITY;
    let mut south = f64::INFINITY;
    let mut east = f64::NEG_INFINITY;
    let mut north = f64::NEG_INFINITY;
    for (_, corners) in &valid {
        for c in corners {
            west = west.min(c[0]);
            east = east.max(c[0]);
            south = south.min(c[1]);
            north = north.max(c[1]);
        }
    }
    let mid_lat = 0.5 * (south + north);
    let lon_scale = mid_lat.to_radians().cos().abs().max(0.1);
    let span_x = ((east - west) * lon_scale).max(1e-6);
    let span_y = (north - south).max(1e-6);
    let cw = canvas_width.clamp(64, 4096);
    let ch = (((f64::from(cw) * span_y) / span_x).round() as u32).clamp(64, 4096);

    let mut accum_r = vec![0.0f64; (cw * ch) as usize];
    let mut accum_g = vec![0.0f64; (cw * ch) as usize];
    let mut accum_b = vec![0.0f64; (cw * ch) as usize];
    let mut accum_w = vec![0.0f64; (cw * ch) as usize];

    for (img, corners) in &valid {
        let (iw, ih) = img.dimensions();
        if iw < 2 || ih < 2 {
            continue;
        }
        let dst_quad: [[f64; 2]; 4] = corners.map(|c| {
            [
                ((c[0] - west) * lon_scale / span_x) * f64::from(cw - 1),
                ((north - c[1]) / span_y) * f64::from(ch - 1),
            ]
        });
        let src_quad = [
            [0.0, 0.0],
            [f64::from(iw - 1), 0.0],
            [f64::from(iw - 1), f64::from(ih - 1)],
            [0.0, f64::from(ih - 1)],
        ];
        let Some(h_inv) = homography_4pt(&dst_quad, &src_quad) else {
            continue;
        };

        let min_x = dst_quad
            .iter()
            .map(|p| p[0].floor() as i64)
            .min()
            .unwrap_or(0)
            .clamp(0, i64::from(cw) - 1) as u32;
        let max_x = dst_quad
            .iter()
            .map(|p| p[0].ceil() as i64)
            .max()
            .unwrap_or(0)
            .clamp(0, i64::from(cw) - 1) as u32;
        let min_y = dst_quad
            .iter()
            .map(|p| p[1].floor() as i64)
            .min()
            .unwrap_or(0)
            .clamp(0, i64::from(ch) - 1) as u32;
        let max_y = dst_quad
            .iter()
            .map(|p| p[1].ceil() as i64)
            .max()
            .unwrap_or(0)
            .clamp(0, i64::from(ch) - 1) as u32;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let denom =
                    h_inv[(2, 0)] * f64::from(px) + h_inv[(2, 1)] * f64::from(py) + h_inv[(2, 2)];
                if denom.abs() <= 1e-9 {
                    continue;
                }
                let u =
                    (h_inv[(0, 0)] * f64::from(px) + h_inv[(0, 1)] * f64::from(py) + h_inv[(0, 2)])
                        / denom;
                let v =
                    (h_inv[(1, 0)] * f64::from(px) + h_inv[(1, 1)] * f64::from(py) + h_inv[(1, 2)])
                        / denom;
                if u >= 0.0 && v >= 0.0 && u <= f64::from(iw - 1) && v <= f64::from(ih - 1) {
                    let fx = (u / f64::from(iw - 1)).min(1.0 - u / f64::from(iw - 1));
                    let fy = (v / f64::from(ih - 1)).min(1.0 - v / f64::from(ih - 1));
                    let weight = (4.0 * fx * fy).clamp(0.05, 1.0);
                    let p = img.get_pixel(u.round() as u32, v.round() as u32);
                    let idx = (py * cw + px) as usize;
                    accum_r[idx] += f64::from(p[0]) * weight;
                    accum_g[idx] += f64::from(p[1]) * weight;
                    accum_b[idx] += f64::from(p[2]) * weight;
                    accum_w[idx] += weight;
                }
            }
        }
    }

    let mut out = RgbaImage::new(cw, ch);
    for y in 0..ch {
        for x in 0..cw {
            let idx = (y * cw + x) as usize;
            let w = accum_w[idx];
            if w > 0.0 {
                out.put_pixel(
                    x,
                    y,
                    Rgba([
                        (accum_r[idx] / w).round().clamp(0.0, 255.0) as u8,
                        (accum_g[idx] / w).round().clamp(0.0, 255.0) as u8,
                        (accum_b[idx] / w).round().clamp(0.0, 255.0) as u8,
                        255,
                    ]),
                );
            } else {
                out.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
        }
    }
    if let Some(parent) = output_png.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    out.save(output_png)
        .map_err(|source| PipelineError::Image {
            path: output_png.to_path_buf(),
            source,
        })?;

    Ok(json!({
        "frames_composited": valid.len(),
        "size": [cw, ch],
        "bbox": [west, south, east, north],
        "output": output_png.display().to_string(),
    }))
}

/// Merge multiple single-flight `sokoly-debrief` v1 directories onto a shared UTC timeline (`compose-survey`).
///
/// # Errors
///
/// Returns [`PipelineError`] when reading flight debriefs, checking anchors, or writing fails.
pub fn compose_survey_debrief(
    flight_dirs: &[PathBuf],
    output_dir: &Path,
) -> Result<Value, PipelineError> {
    if flight_dirs.is_empty() {
        return Err(PipelineError::Invalid(
            "compose_survey_debrief requires at least one flight directory".into(),
        ));
    }
    std::fs::create_dir_all(output_dir).map_err(|source| PipelineError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let mut flights = Vec::new();
    let mut global_start_ms = i64::MAX;
    let mut global_end_ms = i64::MIN;

    for (idx, dir) in flight_dirs.iter().enumerate() {
        let debrief_path = if dir.join("debrief.json").is_file() {
            dir.join("debrief.json")
        } else {
            dir.join("replay.json")
        };
        let doc: Value = read_json(&debrief_path)?;
        let anchor_ms = doc
            .get("time_anchor")
            .and_then(|a| a.get("unix_ms"))
            .and_then(Value::as_i64)
            .unwrap_or((idx as i64) * 60_000);
        let dur_ms = doc
            .get("duration_ms")
            .and_then(Value::as_i64)
            .unwrap_or(30_000);
        global_start_ms = global_start_ms.min(anchor_ms);
        global_end_ms = global_end_ms.max(anchor_ms + dur_ms);
        flights.push((idx, dir.clone(), anchor_ms, dur_ms, doc));
    }

    flights.sort_by_key(|f| f.2);
    let mut merged_streams = Vec::new();
    for (idx, dir, anchor_ms, dur_ms, doc) in &flights {
        let offset_ms = anchor_ms - global_start_ms;
        merged_streams.push(json!({
            "stream_id": format!("flight-{:02}", idx + 1),
            "source_dir": dir.display().to_string(),
            "offset_ms": offset_ms,
            "duration_ms": dur_ms,
            "start_unix_ms": anchor_ms,
            "items": doc.get("items").or_else(|| doc.get("frames")).cloned().unwrap_or_else(|| json!([])),
        }));
    }

    let survey = json!({
        "format": "sokoly-debrief",
        "version": 1,
        "kind": "multi-flight-survey",
        "start_unix_ms": global_start_ms,
        "duration_ms": (global_end_ms - global_start_ms).max(0),
        "streams": merged_streams,
    });
    write_json(&output_dir.join("debrief.json"), &survey)?;
    Ok(json!({
        "streams": merged_streams.len(),
        "start_unix_ms": global_start_ms,
        "duration_ms": (global_end_ms - global_start_ms).max(0),
    }))
}

fn homography_4pt(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<Matrix3<f64>> {
    let mut a = nalgebra::SMatrix::<f64, 8, 8>::zeros();
    let mut b = nalgebra::SVector::<f64, 8>::zeros();
    for i in 0..4 {
        let (x, y) = (src[i][0], src[i][1]);
        let (u, v) = (dst[i][0], dst[i][1]);
        let r0 = i * 2;
        let r1 = r0 + 1;
        a[(r0, 0)] = x;
        a[(r0, 1)] = y;
        a[(r0, 2)] = 1.0;
        a[(r0, 6)] = -x * u;
        a[(r0, 7)] = -y * u;
        b[r0] = u;

        a[(r1, 3)] = x;
        a[(r1, 4)] = y;
        a[(r1, 5)] = 1.0;
        a[(r1, 6)] = -x * v;
        a[(r1, 7)] = -y * v;
        b[r1] = v;
    }
    let h = a.lu().solve(&b)?;
    Some(Matrix3::new(
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0,
    ))
}
