//! Bind replay inputs and preserve every recorded candidate in search order.
use super::{Candidate, Matches};
use crate::{Error, assets, job::Image};
use navigate_visual::PixelMatch;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
pub(super) fn verified_blocking<T: DeserializeOwned>(
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
pub(super) fn candidates(
    source: &Value,
    path: &Path,
    limit: usize,
) -> Result<Vec<Candidate>, Error> {
    let mut result = Vec::new();
    append(&mut result, "coarse", None, &source["coarse"], path, limit)?;
    for (field, stage) in [("windows", "local"), ("fine_windows", "fine")] {
        let windows = source[field]
            .as_array()
            .ok_or_else(|| Error::Input(format!("missing frozen {field}")))?;
        for row in windows {
            let window =
                serde_json::from_value(row["window"].clone()).map_err(|source| Error::Json {
                    path: path.into(),
                    source,
                })?;
            append(
                &mut result,
                stage,
                Some(window),
                &row["candidates"],
                path,
                limit,
            )?;
        }
    }
    if result.is_empty() {
        return Err(Error::Input("empty frozen acquisition".into()));
    }
    Ok(result)
}
fn append(
    result: &mut Vec<Candidate>,
    stage: &'static str,
    window: Option<[u32; 4]>,
    rows: &Value,
    path: &Path,
    limit: usize,
) -> Result<(), Error> {
    let rows = rows
        .as_array()
        .ok_or_else(|| Error::Input("missing frozen candidate array".into()))?;
    if result.len().saturating_add(rows.len()) > limit.min(76) {
        return Err(Error::Input("frozen acquisition candidate limit".into()));
    }
    for row in rows {
        let values: Vec<[f64; 4]> = serde_json::from_value(row["correspondences"].clone())
            .map_err(|source| Error::Json {
                path: path.into(),
                source,
            })?;
        let rotation = row["reference_quarter_turns"]
            .as_u64()
            .filter(|r| *r < 4)
            .ok_or_else(|| Error::Input("frozen rotation must be0 through3".into()))?
            as u8;
        if values.len() > 16384
            || row["matches"].as_u64() != Some(values.len() as u64)
            || !row["accepted"].is_boolean()
        {
            return Err(Error::Input(
                "invalid frozen match count or decision".into(),
            ));
        }
        result.push(Candidate {
            stage,
            window,
            original: row.clone(),
            matches: Matches {
                rotation,
                matching_ms: 0.0,
                pairs: values
                    .iter()
                    .map(|p| PixelMatch {
                        reference: [p[0], p[1]].into(),
                        query: [p[2], p[3]].into(),
                    })
                    .collect(),
            },
        });
    }
    Ok(())
}
pub(super) fn decision(row: &Value) -> Value {
    if row["accepted"] == true {
        json!({"accepted":true,"pose":row["pose"],"quality":row["quality"]})
    } else {
        json!({"accepted":false,"rejection":row["rejection"]})
    }
}
pub(super) fn differences(a: &Value, b: &Value) -> Vec<&'static str> {
    ["accepted", "pose", "quality", "rejection"]
        .into_iter()
        .filter(|k| a[k] != b[k])
        .collect()
}
