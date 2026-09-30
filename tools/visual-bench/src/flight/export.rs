//! Export each candidate as its own conditional path, with explicit gaps.
use crate::{BenchError, stream::write_record_blocking, trial::writer_blocking};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader},
    path::Path,
};
#[derive(Default)]
struct Segment {
    context: Option<(String, String)>,
    coordinates: Vec<Value>,
    observations: Vec<Value>,
}
pub(super) fn write_blocking(input: &Path, output: &Path) -> Result<(), BenchError> {
    let file = std::fs::File::open(input).map_err(|source| BenchError::Io {
        path: input.to_owned(),
        source,
    })?;
    let records = BufReader::new(file)
        .lines()
        .map(|line| {
            let line = line.map_err(|source| BenchError::Io {
                path: input.to_owned(),
                source,
            })?;
            serde_json::from_str::<Value>(&line).map_err(|source| BenchError::Json {
                path: input.to_owned(),
                source,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let track = collection(&records)?;
    write_record_blocking(&mut writer_blocking(output)?, output, &track)
}
pub(crate) fn collection(records: &[Value]) -> Result<Value, BenchError> {
    let mut branches: BTreeMap<u64, Segment> = BTreeMap::new();
    let mut features = Vec::new();
    let mut gaps = Vec::new();
    for row in records {
        for h in row["candidate_hypotheses"].as_array().into_iter().flatten() {
            let id = h["candidate_id"]
                .as_u64()
                .ok_or_else(|| BenchError::Record {
                    reason: "candidate has no identity".into(),
                })?;
            let segment = branches.entry(id).or_default();
            let context = h["anchor_observation_sha256"]
                .as_str()
                .zip(row["context"]["map_manifest_sha256"].as_str())
                .map(|(a, m)| (a.to_owned(), m.to_owned()));
            if h["accepted"] != true && h["tracking_supported"] != true {
                end(&mut features, id, segment);
                gaps.push(json!({"candidate_id":id,"capture_time_ns":row["capture_time_ns"],"rejection":h}));
                continue;
            }
            if h["continuity_break"] == true || context.is_none() || segment.context != context {
                end(&mut features, id, segment)
            }
            segment.context = context;
            let point = coordinates(h)?;
            segment.coordinates.push(point.clone());
            segment.observations.push(row["observation_sha256"].clone());
            features.push(json!({"type":"Feature","geometry":{"type":"Point","coordinates":point},"properties":{
                "kind":"conditional camera estimate","hypothesis":h,"capture_time_ns":row["capture_time_ns"],"context":row["context"]}}));
            if h["map_check"]["accepted"] == true {
                let checked = &h["map_check"];
                features.push(json!({"type":"Feature","geometry":{"type":"Point","coordinates":coordinates(checked)?},
                    "properties":{"kind":"diagnostic map check","candidate_id":id,"capture_time_ns":row["capture_time_ns"],"hypothesis":checked}}));
            }
        }
    }
    for (id, mut segment) in branches {
        end(&mut features, id, &mut segment)
    }
    Ok(
        json!({"type":"FeatureCollection","features":features,"gaps":gaps,
        "accuracy":"not independently measured","correlation":"unknown; shared map and estimated surface depth",
        "scope":"Conditional paths remain separate. Map checks are separate points. Rejections and anchor changes split paths."}),
    )
}
fn coordinates(value: &Value) -> Result<Value, BenchError> {
    let lon = value["longitude_deg"].as_f64();
    let lat = value["latitude_deg"].as_f64();
    match (lon, lat) {
        (Some(lon), Some(lat))
            if (-180.0..=180.0).contains(&lon) && (-90.0..=90.0).contains(&lat) =>
        {
            Ok(json!([lon, lat]))
        }
        _ => Err(BenchError::Record {
            reason: "supported pose has invalid geographic coordinates".into(),
        }),
    }
}
fn end(features: &mut Vec<Value>, id: u64, segment: &mut Segment) {
    if segment.coordinates.len() > 1 {
        features.push(json!({"type":"Feature","geometry":{"type":"LineString","coordinates":segment.coordinates},
            "properties":{"kind":"conditional path segment","candidate_id":id,"anchor_and_map_identity":segment.context,"observations":segment.observations}}));
    }
    segment.coordinates.clear();
    segment.observations.clear();
}
#[cfg(test)]
mod tests;
