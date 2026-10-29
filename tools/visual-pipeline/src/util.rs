//! Deterministic SHA-256 hashing, JSON I/O, civil UTC formatting, and contact sheets.

use crate::PipelineError;
use image::{Rgb, RgbImage, imageops::FilterType};
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

/// Compute the lowercase hex SHA-256 digest of `path`.
///
/// # Errors
///
/// Returns [`PipelineError::Io`] when `path` cannot be read.
pub fn file_sha256(path: &Path) -> Result<String, PipelineError> {
    let bytes = fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(bytes_sha256(&bytes))
}

/// Compute the lowercase hex SHA-256 digest of `bytes`.
pub fn bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Read and deserialize a JSON file from `path`.
///
/// # Errors
///
/// Returns [`PipelineError`] when reading or JSON parsing fails.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, PipelineError> {
    let text = fs::read_to_string(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| PipelineError::Json {
        context: path.display().to_string(),
        source,
    })
}

/// Serialize `value` as 2-space pretty JSON with a trailing newline to `path`.
///
/// # Errors
///
/// Returns [`PipelineError`] when serialization or file write fails.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), PipelineError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut text = serde_json::to_string_pretty(value).map_err(|source| PipelineError::Json {
        context: path.display().to_string(),
        source,
    })?;
    text.push('\n');
    fs::write(path, text).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Write one line to standard output without invoking `println!`.
///
/// # Errors
///
/// Returns [`PipelineError::Io`] when writing to stdout fails.
pub fn write_stdout_line(line: &str) -> Result<(), PipelineError> {
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{line}").map_err(|source| PipelineError::Io {
        path: PathBuf::from("stdout"),
        source,
    })
}

/// Format Unix epoch milliseconds as an ISO-8601 UTC string (`YYYY-MM-DDTHH:MM:SS+00:00` or `.sss`).
pub fn unix_ms_to_iso8601(unix_ms: i64) -> String {
    let total_secs = unix_ms.div_euclid(1000);
    let millis = unix_ms.rem_euclid(1000);
    let days = total_secs.div_euclid(86_400);
    let sec_of_day = total_secs.rem_euclid(86_400);
    let hour = sec_of_day / 3600;
    let minute = (sec_of_day % 3600) / 60;
    let second = sec_of_day % 60;
    let (year, month, day) = civil_from_days(days);
    if millis == 0 {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}+00:00")
    } else {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}+00:00")
    }
}

/// Parse an ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM|-HH:MM]`) into Unix milliseconds.
pub fn parse_iso8601_ms(text: &str) -> Option<i64> {
    let s = text.trim();
    if s.len() < 19 {
        return None;
    }
    let year: i64 = s.get(0..4)?.parse().ok()?;
    let month: u32 = s.get(5..7)?.parse().ok()?;
    let day: u32 = s.get(8..10)?.parse().ok()?;
    let hour: i64 = s.get(11..13)?.parse().ok()?;
    let minute: i64 = s.get(14..16)?.parse().ok()?;
    let second: i64 = s.get(17..19)?.parse().ok()?;
    let rest = &s[19..];

    let (frac_ms, tz_part) = if let Some(after_dot) = rest.strip_prefix('.') {
        let digits: String = after_dot
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        let tz = &after_dot[digits.len()..];
        let mut padded = digits;
        while padded.len() < 3 {
            padded.push('0');
        }
        padded.truncate(3);
        let ms: i64 = padded.parse().unwrap_or(0);
        (ms, tz)
    } else {
        (0, rest)
    };

    let offset_secs: i64 = if tz_part.is_empty() || tz_part == "Z" || tz_part == "z" {
        0
    } else if (tz_part.starts_with('+') || tz_part.starts_with('-')) && tz_part.len() >= 5 {
        let sign = if tz_part.starts_with('-') { -1 } else { 1 };
        let oh: i64 = tz_part.get(1..3)?.parse().ok()?;
        let om_start = if tz_part.as_bytes().get(3) == Some(&b':') {
            4
        } else {
            3
        };
        let om: i64 = tz_part.get(om_start..om_start + 2)?.parse().ok()?;
        sign * (oh * 3600 + om * 60)
    } else {
        0
    };

    let days = days_from_civil(year, month, day)?;
    let utc_secs = days * 86_400 + hour * 3600 + minute * 60 + second - offset_secs;
    Some(utc_secs * 1000 + frac_ms)
}

fn civil_from_days(z_raw: i64) -> (i64, u32, u32) {
    let z = z_raw + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32)
}

fn days_from_civil(y_raw: i64, m: u32, d: u32) -> Option<i64> {
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y = y_raw - i64::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = if m > 2 {
        u64::from(m - 3)
    } else {
        u64::from(m + 9)
    };
    let doy = (153 * mp + 2) / 5 + u64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + (doe as i64) - 719_468)
}

/// Write 4-column thumbnail contact sheets (`contact-sheet-01.jpg`, ...) into `root`.
///
/// # Errors
///
/// Returns [`PipelineError`] when thumbnail reading or sheet writing fails.
pub fn write_contact_sheets(root: &Path, ids: &[String]) -> Result<(), PipelineError> {
    if ids.is_empty() {
        return Ok(());
    }
    let cols = 4u32;
    let cell_w = 320u32;
    let cell_h = 240u32;
    let per_page = 24usize;
    for (page, group) in ids.chunks(per_page).enumerate() {
        let rows = (group.len() as u32).div_ceil(cols);
        let mut sheet = RgbImage::from_pixel(cols * cell_w, rows * cell_h, Rgb([24, 24, 24]));
        for (i, id) in group.iter().enumerate() {
            let thumb_path = root.join("thumbnails").join(format!("{id}.jpg"));
            if !thumb_path.is_file() {
                continue;
            }
            let img = image::open(&thumb_path)
                .map_err(|source| PipelineError::Image {
                    path: thumb_path,
                    source,
                })?
                .to_rgb8();
            let scale = (f64::from(cell_w) / f64::from(img.width().max(1)))
                .min(f64::from(cell_h) / f64::from(img.height().max(1)));
            let tw = ((f64::from(img.width()) * scale).round() as u32).clamp(1, cell_w);
            let th = ((f64::from(img.height()) * scale).round() as u32).clamp(1, cell_h);
            let resized = image::imageops::resize(&img, tw, th, FilterType::Triangle);
            let ox = (i as u32 % cols) * cell_w + (cell_w - tw) / 2;
            let oy = (i as u32 / cols) * cell_h + (cell_h - th) / 2;
            for y in 0..th {
                for x in 0..tw {
                    sheet.put_pixel(ox + x, oy + y, *resized.get_pixel(x, y));
                }
            }
        }
        let out_path = root.join(format!("contact-sheet-{:02}.jpg", page + 1));
        sheet
            .save(&out_path)
            .map_err(|source| PipelineError::Image {
                path: out_path,
                source,
            })?;
    }
    Ok(())
}

/// Run an external tool (`ffmpeg`, `ffprobe`, `curl`, `magick`) and return its stdout bytes.
///
/// # Errors
///
/// Returns [`PipelineError::Tool`] when execution fails or exits non-zero.
pub fn run_tool(tool: &str, args: &[&str]) -> Result<Vec<u8>, PipelineError> {
    let output = Command::new(tool)
        .args(args)
        .output()
        .map_err(|e| PipelineError::Tool {
            tool: tool.to_owned(),
            message: e.to_string(),
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(PipelineError::Tool {
            tool: tool.to_owned(),
            message: format!("status {}: {stderr}", output.status),
        });
    }
    Ok(output.stdout)
}
