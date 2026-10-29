//! Unit-sphere SLERP interpolation of missing photo GPS priors (`photo_priors`).

use super::{EARTH_RADIUS_M, FlightGroup, MAX_FLIGHT_GAP_S, PhotoCase};
use crate::{
    PipelineError,
    metadata::PositionClaim,
    util::{parse_iso8601_ms, read_json, unix_ms_to_iso8601, write_json},
};
use serde_json::Value;
use std::path::Path;

/// Interpolate missing GPS position priors along each flight on the unit sphere (`photo_priors`).
///
/// # Errors
///
/// Returns [`PipelineError`] when `manifest.json` or `profiles/with-prior.json` cannot be updated.
pub fn interpolate_flight_priors(dataset_dir: &Path) -> Result<usize, PipelineError> {
    let manifest_path = dataset_dir.join("manifest.json");
    let profile_path = dataset_dir.join("profiles").join("with-prior.json");
    let manifest: Value = read_json(&manifest_path)?;
    let mut profile: Value = read_json(&profile_path)?;

    let groups: Vec<FlightGroup> = manifest
        .get("capture_groups")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|source| PipelineError::Json {
            context: "manifest.capture_groups".into(),
            source,
        })?
        .unwrap_or_default();

    let cases_arr = profile
        .get_mut("cases")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PipelineError::Invalid("missing cases array in with-prior.json".into()))?;
    let mut cases: Vec<PhotoCase> = cases_arr
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()
        .map_err(|source| PipelineError::Json {
            context: "with-prior.cases".into(),
            source,
        })?;

    let mut filled = 0usize;
    for group in &groups {
        let indices: Vec<usize> = group
            .photos
            .iter()
            .filter_map(|id| cases.iter().position(|c| &c.id == id))
            .collect();
        let known: Vec<(usize, i64, PositionClaim)> = indices
            .iter()
            .filter_map(|&idx| {
                let pr = cases[idx].prior.as_ref()?;
                let t = pr.capture_time.as_deref().and_then(parse_iso8601_ms)?;
                let pos = pr.position.clone()?;
                Some((idx, t, pos))
            })
            .collect();
        if known.is_empty() {
            continue;
        }

        for &idx in &indices {
            let Some(pr) = cases[idx].prior.as_mut() else {
                continue;
            };
            if pr.position.is_some() {
                continue;
            }
            let Some(t_ms) = pr.capture_time.as_deref().and_then(parse_iso8601_ms) else {
                continue;
            };
            let before = known.iter().rfind(|(_, tk, _)| *tk <= t_ms);
            let after = known.iter().find(|(_, tk, _)| *tk >= t_ms);
            let interp = match (before, after) {
                (Some((_, t0, p0)), Some((_, t1, p1))) if t1 > t0 => {
                    let frac = ((t_ms - t0) as f64) / ((t1 - t0) as f64);
                    let (lat, lon) =
                        slerp_lat_lon(p0.latitude, p0.longitude, p1.latitude, p1.longitude, frac);
                    let alt = match (p0.altitude_m, p1.altitude_m) {
                        (Some(a0), Some(a1)) => Some(a0 + frac * (a1 - a0)),
                        (Some(a0), None) => Some(a0),
                        (None, Some(a1)) => Some(a1),
                        (None, None) => None,
                    };
                    let dt_s = ((t1 - t0) as f64) * 1e-3;
                    Some(PositionClaim {
                        latitude: lat,
                        longitude: lon,
                        reported_horizontal_error_m: Some((dt_s * 60.0).max(250.0)),
                        altitude_m: alt,
                        unsigned_altitude_m: alt.map(f64::abs),
                        altitude_reference_text: Some(format!(
                            "interpolated between {} and {}",
                            unix_ms_to_iso8601(*t0),
                            unix_ms_to_iso8601(*t1)
                        )),
                        altitude_datum: "flight_interpolated_unverified".into(),
                    })
                }
                (Some((_, t0, p0)), None) | (None, Some((_, t0, p0))) => {
                    let dt_s = ((t_ms - t0).abs() as f64) * 1e-3;
                    (dt_s <= MAX_FLIGHT_GAP_S).then(|| PositionClaim {
                        latitude: p0.latitude,
                        longitude: p0.longitude,
                        reported_horizontal_error_m: Some((dt_s * 75.0).max(500.0)),
                        altitude_m: p0.altitude_m,
                        unsigned_altitude_m: p0.unsigned_altitude_m,
                        altitude_reference_text: Some("nearest flight fix".into()),
                        altitude_datum: "flight_extrapolated_unverified".into(),
                    })
                }
                _ => None,
            };
            if let Some(claim) = interp {
                pr.position = Some(claim);
                filled += 1;
            }
        }
    }

    profile["cases"] = serde_json::to_value(&cases).map_err(|source| PipelineError::Json {
        context: "serialize cases".into(),
        source,
    })?;
    write_json(&profile_path, &profile)?;
    Ok(filled)
}

pub(super) fn unit_vec(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let (slat, clat) = lat_deg.to_radians().sin_cos();
    let (slon, clon) = lon_deg.to_radians().sin_cos();
    [clat * clon, clat * slon, slat]
}

fn slerp_lat_lon(lat0: f64, lon0: f64, lat1: f64, lon1: f64, t: f64) -> (f64, f64) {
    let u0 = unit_vec(lat0, lon0);
    let u1 = unit_vec(lat1, lon1);
    let dot = (u0[0] * u1[0] + u0[1] * u1[1] + u0[2] * u1[2]).clamp(-1.0, 1.0);
    let omega = dot.acos();
    let v = if omega.abs() < 1e-9 {
        [
            u0[0] + t * (u1[0] - u0[0]),
            u0[1] + t * (u1[1] - u0[1]),
            u0[2] + t * (u1[2] - u0[2]),
        ]
    } else {
        let s = omega.sin();
        let a = ((1.0 - t) * omega).sin() / s;
        let b = (t * omega).sin() / s;
        [
            a * u0[0] + b * u1[0],
            a * u0[1] + b * u1[1],
            a * u0[2] + b * u1[2],
        ]
    };
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    let lat = (v[2] / norm).clamp(-1.0, 1.0).asin().to_degrees();
    let lon = v[1].atan2(v[0]).to_degrees();
    (lat, lon)
}

pub(super) fn great_circle_m(lat0: f64, lon0: f64, lat1: f64, lon1: f64) -> f64 {
    let u0 = unit_vec(lat0, lon0);
    let u1 = unit_vec(lat1, lon1);
    let dot = (u0[0] * u1[0] + u0[1] * u1[1] + u0[2] * u1[2]).clamp(-1.0, 1.0);
    EARTH_RADIUS_M * dot.acos()
}
