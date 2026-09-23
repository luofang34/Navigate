//! GeoJSON output retains visual rejections as gaps between track segments.
//!
//! Positions use longitude and latitude. Altitude remains a property because
//! the map's vertical datum need not be the WGS84 ellipsoid used by GeoJSON.

use crate::{BenchError, read_blocking, stream::write_record_blocking, trial::writer_blocking};
use serde_json::{Value, json};
use std::path::Path;

pub(crate) fn export_blocking(input: &Path, output: &Path) -> Result<(), BenchError> {
    let bytes = read_blocking(input)?;
    let mut records = Vec::new();
    for line in bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
        records.push(
            serde_json::from_slice::<Value>(line).map_err(|source| BenchError::Json {
                path: input.to_owned(),
                source,
            })?,
        );
    }
    let track = collection(&records)?;
    let mut writer = writer_blocking(output)?;
    write_record_blocking(&mut writer, output, &track)
}

fn collection(records: &[Value]) -> Result<Value, BenchError> {
    let mut features = Vec::new();
    let mut segment = Vec::new();
    let mut context = None;
    for record in records {
        let selected = record["map_manifest_sha256"].as_str();
        if record["accepted"] == false || (context.is_some() && selected != context) {
            end_segment(&mut features, &mut segment);
        }
        context = selected;
        if record["accepted"] == false {
            continue;
        }
        let lon = record["longitude_deg"].as_f64();
        let lat = record["latitude_deg"].as_f64();
        let (Some(lon), Some(lat)) = (lon, lat) else {
            return Err(BenchError::Record {
                reason: "accepted observation has no geographic coordinates".into(),
            });
        };
        let point = json!([lon, lat]);
        segment.push(point.clone());
        features.push(json!({"type":"Feature","geometry":{"type":"Point","coordinates":point},"properties":record}));
    }
    end_segment(&mut features, &mut segment);
    Ok(json!({"type":"FeatureCollection","features":features}))
}

fn end_segment(features: &mut Vec<Value>, segment: &mut Vec<Value>) {
    if segment.len() >= 2 {
        features.push(
            json!({"type":"Feature","geometry":{"type":"LineString","coordinates":segment},
            "properties":{"kind":"visual-observation-track"}}),
        );
    }
    segment.clear();
}

#[cfg(test)]
mod tests;
