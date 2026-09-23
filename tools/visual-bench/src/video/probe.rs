//! Read frame presentation times without assuming a fixed frame rate.

use crate::BenchError;
use navigate_visual::CameraModel;
use serde::Deserialize;
use std::{path::Path, process::Command};

#[derive(Deserialize)]
struct Probe {
    frames: Vec<ProbeFrame>,
}

#[derive(Deserialize)]
struct ProbeFrame {
    best_effort_timestamp_time: Option<String>,
    width: u32,
    height: u32,
}

pub(super) fn timestamps_blocking(
    path: &Path,
    camera: CameraModel,
) -> Result<Vec<u64>, BenchError> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time,width,height",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|source| BenchError::Spawn {
            program: "ffprobe",
            source,
        })?;
    if !output.status.success() {
        return Err(BenchError::Process {
            program: "ffprobe",
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let probe: Probe =
        serde_json::from_slice(&output.stdout).map_err(|source| BenchError::Json {
            path: path.to_owned(),
            source,
        })?;
    timeline(&probe.frames, camera)
}

fn timeline(frames: &[ProbeFrame], camera: CameraModel) -> Result<Vec<u64>, BenchError> {
    if frames.is_empty() {
        return Err(BenchError::EmptyStream);
    }
    let mut start = None;
    let mut previous = None;
    let mut timestamps = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        if (frame.width, frame.height) != (camera.width, camera.height) {
            return Err(BenchError::ChangedCamera);
        }
        let value =
            frame
                .best_effort_timestamp_time
                .as_ref()
                .ok_or_else(|| BenchError::Record {
                    reason: format!("video frame {index} has no finite presentation time"),
                })?;
        let seconds = value
            .parse::<f64>()
            .map_err(|source| BenchError::Timestamp {
                frame: index,
                value: value.clone(),
                source,
            })?;
        if !seconds.is_finite() {
            return Err(BenchError::Record {
                reason: format!("video frame {index} has non-finite presentation time {value}"),
            });
        }
        let relative = seconds - *start.get_or_insert(seconds);
        if relative < 0.0 || relative >= u64::MAX as f64 / 1e9 {
            return Err(BenchError::Record {
                reason: format!("video frame {index} time is outside the timestamp range"),
            });
        }
        let nanos = (relative * 1e9).round() as u64;
        if previous.is_some_and(|time| nanos <= time) {
            return Err(BenchError::Record {
                reason: format!("video frame {index} time does not increase"),
            });
        }
        previous = Some(nanos);
        timestamps.push(nanos);
    }
    Ok(timestamps)
}

#[cfg(test)]
mod tests;
