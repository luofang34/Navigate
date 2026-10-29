//! EXIF metadata extraction and sRGB/orientation photo decoding (`photo_metadata`).

use crate::{PipelineError, util::run_tool};
use exif::{In, Reader, Tag, Value as ExifValue};
use image::{RgbImage, imageops};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs::File, io::BufReader, path::Path};

/// Reported GPS position claim from EXIF tags (treated as an unverified search seed).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PositionClaim {
    /// WGS84 latitude in degrees.
    pub latitude: f64,
    /// WGS84 longitude in degrees.
    pub longitude: f64,
    /// Reported `GPSHPositioningError` in metres, if present.
    pub reported_horizontal_error_m: Option<f64>,
    /// Signed altitude in metres, if present.
    pub altitude_m: Option<f64>,
    /// Unsigned altitude magnitude in metres.
    pub unsigned_altitude_m: Option<f64>,
    /// Raw `GPSAltitudeRef` description.
    pub altitude_reference_text: Option<String>,
    /// Altitude datum status (`"unverified"`).
    pub altitude_datum: String,
}

/// Extracted EXIF claims for one photo (`metadata/{id}.json` -> `claims`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PhotoClaims {
    /// ISO-8601 capture timestamp when EXIF date/offset tags exist.
    pub capture_time: Option<String>,
    /// Provenance of `capture_time`.
    pub clock_source: Option<String>,
    /// Reported sub-second or GPS clock accuracy in seconds.
    pub clock_accuracy_s: Option<f64>,
    /// Reported GPS position claim.
    pub position: Option<PositionClaim>,
    /// Reported camera compass direction in degrees `[0, 360)`.
    pub direction_deg: Option<f64>,
    /// Direction reference (`"T"` true north or `"M"` magnetic north).
    pub direction_reference: Option<String>,
    /// Physical focal length in millimetres.
    pub focal_mm: Option<f64>,
    /// 35 mm equivalent focal length in millimetres.
    pub focal_35mm_mm: Option<f64>,
    /// Camera model string.
    pub camera_model: Option<String>,
    /// Lens model string.
    pub lens_model: Option<String>,
}

/// Extract EXIF tag dictionary and structured [`PhotoClaims`] from `path`.
pub fn extract_photo_claims(path: &Path) -> (BTreeMap<String, String>, PhotoClaims, u32) {
    let mut raw_tags = BTreeMap::new();
    let mut claims = PhotoClaims::default();
    let mut orientation = 1u32;

    let Ok(file) = File::open(path) else {
        return (raw_tags, claims, orientation);
    };
    let mut buf = BufReader::new(file);
    let Ok(exif) = Reader::new().read_from_container(&mut buf) else {
        return (raw_tags, claims, orientation);
    };

    for field in exif.fields() {
        if field.ifd_num == In::PRIMARY {
            raw_tags.insert(
                field.tag.to_string(),
                field.display_value().with_unit(&exif).to_string(),
            );
        }
    }

    if let Some(f) = exif.get_field(Tag::Orientation, In::PRIMARY)
        && let Some(v) = f.value.get_uint(0)
    {
        orientation = v;
    }

    claims.camera_model = string_tag(&exif, Tag::Model);
    claims.lens_model = string_tag(&exif, Tag::LensModel);
    claims.focal_mm = rational_tag(&exif, Tag::FocalLength);
    claims.focal_35mm_mm = exif
        .get_field(Tag::FocalLengthIn35mmFilm, In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .map(f64::from)
        .or_else(|| rational_tag(&exif, Tag::FocalLengthIn35mmFilm));

    if let Some(dir) = rational_tag(&exif, Tag::GPSImgDirection)
        && dir.is_finite()
    {
        claims.direction_deg = Some(dir.rem_euclid(360.0));
        claims.direction_reference = string_tag(&exif, Tag::GPSImgDirectionRef);
    }

    if let Some(dt) = string_tag(&exif, Tag::DateTimeOriginal) {
        let iso_date =
            exif_datetime_to_iso(&dt, string_tag(&exif, Tag::OffsetTimeOriginal).as_deref());
        claims.capture_time = iso_date;
        claims.clock_source = Some("EXIF DateTimeOriginal".into());
    }

    let lat = dms_tag(
        &exif,
        Tag::GPSLatitude,
        string_tag(&exif, Tag::GPSLatitudeRef).as_deref(),
    );
    let lon = dms_tag(
        &exif,
        Tag::GPSLongitude,
        string_tag(&exif, Tag::GPSLongitudeRef).as_deref(),
    );
    if let (Some(latitude), Some(longitude)) = (lat, lon)
        && (-90.0..=90.0).contains(&latitude)
        && (-180.0..=180.0).contains(&longitude)
    {
        let unsigned_alt = rational_tag(&exif, Tag::GPSAltitude);
        let alt_ref = exif
            .get_field(Tag::GPSAltitudeRef, In::PRIMARY)
            .and_then(|f| f.value.get_uint(0))
            .unwrap_or(0);
        let signed_alt = unsigned_alt.map(|a| if alt_ref == 1 { -a } else { a });
        let h_err = rational_tag(&exif, Tag::GPSHPositioningError);
        claims.position = Some(PositionClaim {
            latitude,
            longitude,
            reported_horizontal_error_m: h_err,
            altitude_m: signed_alt,
            unsigned_altitude_m: unsigned_alt.map(f64::abs),
            altitude_reference_text: Some(if alt_ref == 1 {
                "Below Sea Level".into()
            } else {
                "Above Sea Level".into()
            }),
            altitude_datum: "unverified".into(),
        });
    }

    (raw_tags, claims, orientation)
}

/// Decode `path` into an upright sRGB [`RgbImage`].
///
/// Uses native Rust JPEG/PNG decoding first, applying EXIF `orientation`, and falls back to
/// `magick` or macOS `sips` for HEIC/RAW containers.
///
/// # Errors
///
/// Returns [`PipelineError`] when decoding fails.
pub fn decode_oriented_rgb(path: &Path, orientation: u32) -> Result<RgbImage, PipelineError> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if matches!(ext.as_str(), "jpg" | "jpeg" | "png")
        && let Ok(dyn_img) = image::open(path)
    {
        return Ok(apply_exif_orientation(dyn_img.to_rgb8(), orientation));
    }

    let tmp = std::env::temp_dir().join(format!(
        "navigate-decode-{}-{}.png",
        std::process::id(),
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("img")
    ));
    let src_str = path.to_string_lossy().into_owned();
    let dst_str = tmp.to_string_lossy().into_owned();

    let converted = run_tool(
        "magick",
        &[&src_str, "-auto-orient", "-colorspace", "sRGB", &dst_str],
    )
    .or_else(|_| {
        run_tool(
            "sips",
            &["-s", "format", "png", &src_str, "--out", &dst_str],
        )
    });
    if let Err(err) = converted {
        drop(std::fs::remove_file(&tmp));
        let dyn_img = image::open(path).map_err(|_| err)?;
        return Ok(apply_exif_orientation(dyn_img.to_rgb8(), orientation));
    }
    let decoded = image::open(&tmp)
        .map_err(|source| PipelineError::Image {
            path: path.to_path_buf(),
            source,
        })?
        .to_rgb8();
    drop(std::fs::remove_file(&tmp));
    Ok(decoded)
}

fn apply_exif_orientation(img: RgbImage, orientation: u32) -> RgbImage {
    match orientation {
        2 => imageops::flip_horizontal(&img),
        3 => imageops::rotate180(&img),
        4 => imageops::flip_vertical(&img),
        5 => imageops::rotate90(&imageops::flip_horizontal(&img)),
        6 => imageops::rotate90(&img),
        7 => imageops::rotate270(&imageops::flip_horizontal(&img)),
        8 => imageops::rotate270(&img),
        _ => img,
    }
}

fn string_tag(exif: &exif::Exif, tag: Tag) -> Option<String> {
    let f = exif.get_field(tag, In::PRIMARY)?;
    match &f.value {
        ExifValue::Ascii(vecs) => vecs
            .first()
            .and_then(|b| std::str::from_utf8(b).ok())
            .map(|s| s.trim_matches('\0').trim().to_owned())
            .filter(|s| !s.is_empty()),
        _ => {
            let s = f.display_value().to_string();
            let trimmed = s.trim_matches('"').trim().to_owned();
            (!trimmed.is_empty()).then_some(trimmed)
        }
    }
}

fn rational_tag(exif: &exif::Exif, tag: Tag) -> Option<f64> {
    let f = exif.get_field(tag, In::PRIMARY)?;
    match &f.value {
        ExifValue::Rational(v) => v.first().filter(|r| r.denom != 0).map(|r| r.to_f64()),
        ExifValue::SRational(v) => v.first().filter(|r| r.denom != 0).map(|r| r.to_f64()),
        _ => None,
    }
}

fn dms_tag(exif: &exif::Exif, tag: Tag, reference: Option<&str>) -> Option<f64> {
    let f = exif.get_field(tag, In::PRIMARY)?;
    let ExifValue::Rational(v) = &f.value else {
        return None;
    };
    if v.len() < 3 || v[0].denom == 0 || v[1].denom == 0 || v[2].denom == 0 {
        return None;
    }
    let deg = v[0].to_f64() + v[1].to_f64() / 60.0 + v[2].to_f64() / 3600.0;
    let sign = match reference.map(|s| s.trim()) {
        Some("S" | "s" | "W" | "w") => -1.0,
        _ => 1.0,
    };
    Some(sign * deg)
}

fn exif_datetime_to_iso(raw: &str, offset: Option<&str>) -> Option<String> {
    let s = raw.trim();
    if s.len() < 19 {
        return None;
    }
    let date_part = s[0..10].replace(':', "-");
    let time_part = &s[11..19];
    let tz = offset
        .map(str::trim)
        .filter(|o| o.len() >= 5 && (o.starts_with('+') || o.starts_with('-')))
        .unwrap_or("+00:00");
    Some(format!("{date_part}T{time_part}{tz}"))
}
