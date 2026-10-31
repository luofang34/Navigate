//! QuickTime movie clock reader (`capture_clock`) and prior tracks (`prior_track`).

mod live_prior;

pub use live_prior::seed_live_prior;

use crate::PipelineError;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub(crate) const SECONDS_1904_TO_1970: i64 = 2_082_844_800;

/// Civil-clock anchor read from a QuickTime `moov/mvhd` atom.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeAnchor {
    /// Media timestamp in milliseconds (`0` at recording start).
    pub at: i64,
    /// Civil UTC timestamp in Unix milliseconds.
    pub unix_ms: i64,
    /// Provenance note for the clock value.
    pub source: String,
    /// Time scale (`"utc"` or `"unix"`).
    pub source_scale: String,
    /// Clock uncertainty in milliseconds (`None` when unverified).
    pub uncertainty_ms: Option<i64>,
    /// Clock resolution in milliseconds (`1000` for QuickTime `mvhd`).
    pub resolution_ms: i64,
    /// Calendar epoch note.
    pub original_calendar: String,
}

/// Four-byte QuickTime atom tag and borrowed payload slice.
pub type AtomSlice<'a> = ([u8; 4], &'a [u8]);

/// Read top-level QuickTime or MP4 atoms from `data`.
///
/// # Errors
///
/// Returns [`PipelineError::Invalid`] when an atom header or size is truncated.
pub fn atoms(data: &[u8]) -> Result<Vec<AtomSlice<'_>>, PipelineError> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 8 <= data.len() {
        let mut size =
            u32::from_be_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as usize;
        let kind = [data[at + 4], data[at + 5], data[at + 6], data[at + 7]];
        let mut header = 8usize;
        if size == 1 {
            if at + 16 > data.len() {
                return Err(PipelineError::Invalid(
                    "truncated extended QuickTime atom".into(),
                ));
            }
            let mut ext = [0u8; 8];
            ext.copy_from_slice(&data[at + 8..at + 16]);
            size = u64::from_be_bytes(ext) as usize;
            header = 16;
        } else if size == 0 {
            size = data.len() - at;
        }
        if size < header || at + size > data.len() {
            return Err(PipelineError::Invalid("invalid QuickTime atom size".into()));
        }
        out.push((kind, &data[at + header..at + size]));
        at += size;
    }
    Ok(out)
}

/// Read an explicit QuickTime `moov/mvhd` creation time from `path`.
///
/// This function ignores filesystem timestamps.
/// It returns `Ok(None)` when `moov/mvhd` is absent or stores zero seconds.
///
/// # Errors
///
/// Returns [`PipelineError`] when `path` cannot be opened or has an invalid `moov` header.
pub fn capture_anchor(path: &Path) -> Result<Option<TimeAnchor>, PipelineError> {
    let Some(moov) = read_moov_body(path)? else {
        return Ok(None);
    };
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("video.mp4");
    for (kind, body) in atoms(&moov)? {
        if &kind != b"mvhd" {
            continue;
        }
        if body.len() < 12 || (body[0] != 0 && body[0] != 1) {
            return Err(PipelineError::Invalid(
                "unsupported movie clock header".into(),
            ));
        }
        let seconds = if body[0] == 1 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&body[4..12]);
            u64::from_be_bytes(b) as i64
        } else {
            i64::from(u32::from_be_bytes([body[4], body[5], body[6], body[7]]))
        };
        if seconds == 0 {
            return Ok(None);
        }
        let unix_ms = (seconds - SECONDS_1904_TO_1970) * 1000;
        return Ok(Some(TimeAnchor {
            at: 0,
            unix_ms,
            source: format!(
                "{name}: moov/mvhd creation_time={seconds} seconds since 1904; interpreted as UTC per QuickTime specification; camera clock not GPS-verified"
            ),
            source_scale: "utc".into(),
            uncertainty_ms: None,
            resolution_ms: 1000,
            original_calendar: "Gregorian; QuickTime 1904 epoch".into(),
        }));
    }
    Ok(None)
}

/// Read movie duration in seconds from `moov/mvhd`, or `Ok(None)` when absent.
///
/// # Errors
///
/// Returns [`PipelineError`] when file reading or atom parsing fails.
pub fn movie_duration_s(path: &Path) -> Result<Option<f64>, PipelineError> {
    let Some(moov) = read_moov_body(path)? else {
        return Ok(None);
    };
    for (kind, body) in atoms(&moov)? {
        if &kind != b"mvhd" {
            continue;
        }
        if body.first() == Some(&1) && body.len() >= 32 {
            let timescale = u32::from_be_bytes([body[20], body[21], body[22], body[23]]);
            let mut db = [0u8; 8];
            db.copy_from_slice(&body[24..32]);
            let duration = u64::from_be_bytes(db);
            return Ok((timescale > 0).then(|| (duration as f64) / f64::from(timescale)));
        } else if body.first() == Some(&0) && body.len() >= 20 {
            let timescale = u32::from_be_bytes([body[12], body[13], body[14], body[15]]);
            let duration = u32::from_be_bytes([body[16], body[17], body[18], body[19]]);
            return Ok((timescale > 0).then(|| f64::from(duration) / f64::from(timescale)));
        }
    }
    Ok(None)
}

/// Read QuickTime user-data tags (`xyz`, `fyw`, `gyw`, `gpt`) from `moov`.
///
/// # Errors
///
/// Returns [`PipelineError`] when file reading fails.
pub fn quicktime_udta_tags(
    path: &Path,
) -> Result<std::collections::BTreeMap<String, String>, PipelineError> {
    let mut out = std::collections::BTreeMap::new();
    let Some(moov) = read_moov_body(path)? else {
        return Ok(out);
    };
    for name in ["xyz", "fyw", "gyw", "gpt"] {
        let mut needle = vec![0xa9u8];
        needle.extend_from_slice(name.as_bytes());
        if let Some(at) = moov.windows(4).position(|w| w == needle.as_slice())
            && at + 8 <= moov.len()
        {
            let count = u16::from_be_bytes([moov[at + 4], moov[at + 5]]) as usize;
            if at + 8 + count <= moov.len()
                && let Ok(text) = std::str::from_utf8(&moov[at + 8..at + 8 + count])
            {
                out.insert(name.to_owned(), text.trim_end_matches('\0').to_owned());
            }
        }
    }
    Ok(out)
}

fn read_moov_body(path: &Path) -> Result<Option<Vec<u8>>, PipelineError> {
    let mut file = File::open(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| PipelineError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut pos = 0u64;
    while pos < file_len {
        if pos + 8 > file_len {
            return Err(PipelineError::Invalid("truncated QuickTime header".into()));
        }
        let mut hdr = [0u8; 8];
        file.read_exact(&mut hdr)
            .map_err(|source| PipelineError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let mut size = u64::from(u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]));
        let kind = [hdr[4], hdr[5], hdr[6], hdr[7]];
        let mut consumed = 8u64;
        if size == 1 {
            let mut ext = [0u8; 8];
            file.read_exact(&mut ext)
                .map_err(|source| PipelineError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
            size = u64::from_be_bytes(ext);
            consumed = 16;
        } else if size == 0 {
            size = file_len - pos;
        }
        if size < consumed || pos + size > file_len {
            return Ok(None);
        }
        let body_len = size - consumed;
        if &kind == b"moov" {
            if body_len > 64 * 1024 * 1024 {
                return Err(PipelineError::Invalid("movie header exceeds 64 MiB".into()));
            }
            let mut body = vec![0u8; body_len as usize];
            file.read_exact(&mut body)
                .map_err(|source| PipelineError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
            return Ok(Some(body));
        }
        file.seek(SeekFrom::Current(body_len as i64))
            .map_err(|source| PipelineError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        pos += size;
    }
    Ok(None)
}

/// Build a nadir `[x, y, z, w]` quaternion with image-up at `heading_deg` clockwise from north.
pub fn rotation_from_heading(heading_deg: f64) -> [f64; 4] {
    let half = (-heading_deg).to_radians() * 0.5;
    [0.0, 0.0, half.sin(), half.cos()]
}

/// One frame entry in a version-1 `navigate-prior-track`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PriorTrackEntry {
    /// Video timestamp in seconds.
    pub video_time_s: f64,
    /// Source live-replay timestamp in seconds.
    pub replay_time_s: f64,
    /// Longitude in degrees.
    pub longitude: f64,
    /// Latitude in degrees.
    pub latitude: f64,
    /// Height above launch in metres.
    pub agl_m: f64,
    /// True heading in degrees clockwise from north.
    pub heading_deg: f64,
    /// Four geographic `[lon, lat]` footprint corners.
    pub corners: [[f64; 2]; 4],
}

/// Flight-controller search-pose track (`navigate-prior-track` v1).
#[derive(Clone, Debug, PartialEq)]
pub struct PriorTrack {
    /// Entries sorted by `video_time_s`.
    pub frames: Vec<PriorTrackEntry>,
    /// Bare-earth launch elevation in metres.
    pub terrain_m: f64,
}

impl PriorTrack {
    /// Parse `prior_track` from `acquisition.json`, or return `Ok(None)` when absent.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Invalid`] when `format`, `version`, or `terrain_m` is invalid.
    pub fn from_acquisition(acquisition: &Value) -> Result<Option<Self>, PipelineError> {
        let Some(track) = acquisition.get("prior_track") else {
            return Ok(None);
        };
        if track.is_null() {
            return Ok(None);
        }
        let fmt = track.get("format").and_then(Value::as_str);
        let ver = track.get("version").and_then(Value::as_i64);
        if fmt != Some("navigate-prior-track") || ver != Some(1) {
            return Err(PipelineError::Invalid(
                "prior track is not a version 1 navigate-prior-track".into(),
            ));
        }
        let terrain_m = acquisition
            .get("live_replay")
            .and_then(|lr| lr.get("terrain_m"))
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                PipelineError::Invalid(
                    "prior track needs the live replay's launch terrain height".into(),
                )
            })?;
        let mut frames: Vec<PriorTrackEntry> = track
            .get("frames")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|source| PipelineError::Json {
                context: "prior_track.frames".into(),
                source,
            })?
            .unwrap_or_default();
        frames.sort_by(|a, b| a.video_time_s.total_cmp(&b.video_time_s));
        Ok(Some(Self { frames, terrain_m }))
    }

    /// Return the entry nearest `time_s` when the time difference is at most `max_gap_s`.
    pub fn nearest(&self, time_s: f64, max_gap_s: f64) -> Option<&PriorTrackEntry> {
        let best = self.frames.iter().min_by(|a, b| {
            (a.video_time_s - time_s)
                .abs()
                .total_cmp(&(b.video_time_s - time_s).abs())
        })?;
        ((best.video_time_s - time_s).abs() <= max_gap_s).then_some(best)
    }

    /// Project the nearest entry into `[east_m, north_m, altitude_m]` around `anchor_lat_lon`.
    pub fn local_pose(
        &self,
        time_s: f64,
        anchor_lat_lon: [f64; 2],
        max_gap_s: f64,
    ) -> Option<Value> {
        let entry = self.nearest(time_s, max_gap_s)?;
        let pos = mercator_local(
            anchor_lat_lon,
            entry.latitude,
            entry.longitude,
            self.terrain_m + entry.agl_m,
        );
        Some(json!({
            "position": pos,
            "rotation": rotation_from_heading(entry.heading_deg),
        }))
    }
}

pub(crate) fn mercator_xy(lat_deg: f64, lon_deg: f64) -> (f64, f64) {
    let x = (lon_deg + 180.0) / 360.0;
    let y = (1.0 - lat_deg.to_radians().tan().asinh() / std::f64::consts::PI) * 0.5;
    (x, y)
}

pub(crate) fn mercator_local(anchor: [f64; 2], lat: f64, lon: f64, alt: f64) -> [f64; 3] {
    let scale = 2.0
        * std::f64::consts::PI
        * navigate_visual::MERCATOR_SPHERE_RADIUS_M
        * anchor[0].to_radians().cos();
    let (ox, oy) = mercator_xy(anchor[0], anchor[1]);
    let (x, y) = mercator_xy(lat, lon);
    [(x - ox) * scale, -(y - oy) * scale, alt]
}
