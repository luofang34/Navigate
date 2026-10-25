//! Camera hints from the metadata of a photo: EXIF focal length and GPS, and the
//! gimbal attitude that DJI drones write as XMP.
//!
//! A hint is a starting value for the user to check, not evidence: the photo can
//! be cropped or edited, and many apps remove the metadata.
use exif::{In, Reader, Tag, Value};
use std::io::Cursor;

/// Diagonal of a 36 × 24 mm frame, in millimetres.
const FULL_FRAME_DIAGONAL_MM: f64 = 43.266_615;

/// Hints read from one photo.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhotoMetadata {
    /// Focal length in pixels of the stored image, when the metadata gives it.
    pub focal_px: Option<f64>,
    /// GPS latitude, longitude and altitude, degrees and metres.
    pub gps: Option<[f64; 3]>,
    /// Gimbal compass heading, degrees clockwise from north (DJI XMP).
    pub gimbal_yaw_deg: Option<f64>,
    /// Gimbal pitch, degrees; negative values look down (DJI XMP).
    pub gimbal_pitch_deg: Option<f64>,
    /// Height above the take-off point, metres (DJI XMP).
    pub relative_altitude_m: Option<f64>,
}

impl PhotoMetadata {
    /// Read the hints of an encoded photo of `width × height` pixels.
    pub fn read(bytes: &[u8], [width, height]: [u32; 2]) -> Self {
        let mut out = Self::default();
        if let Ok(exif) = Reader::new().read_from_container(&mut Cursor::new(bytes)) {
            out.focal_px = focal_px(&exif, [width, height]);
            out.gps = gps(&exif);
        }
        out.gimbal_yaw_deg = xmp_number(bytes, "drone-dji:GimbalYawDegree");
        out.gimbal_pitch_deg = xmp_number(bytes, "drone-dji:GimbalPitchDegree");
        out.relative_altitude_m = xmp_number(bytes, "drone-dji:RelativeAltitude");
        out
    }
}

fn rational(exif: &exif::Exif, tag: Tag) -> Option<f64> {
    match &exif.get_field(tag, In::PRIMARY)?.value {
        Value::Rational(v) => v.first().map(|r| r.to_f64()),
        Value::Short(v) => v.first().map(|&s| f64::from(s)),
        Value::Long(v) => v.first().map(|&l| f64::from(l)),
        _ => None,
    }
}

/// The 35 mm equivalent focal length scales with the frame diagonal; without it,
/// the focal length and the focal-plane resolution give the pixel size.
fn focal_px(exif: &exif::Exif, [width, height]: [u32; 2]) -> Option<f64> {
    let diagonal_px = f64::from(width).hypot(f64::from(height));
    if let Some(f35) = rational(exif, Tag::FocalLengthIn35mmFilm).filter(|f| *f > 0.0) {
        return Some(f35 / FULL_FRAME_DIAGONAL_MM * diagonal_px);
    }
    let focal_mm = rational(exif, Tag::FocalLength).filter(|f| *f > 0.0)?;
    let per_unit = rational(exif, Tag::FocalPlaneXResolution).filter(|r| *r > 0.0)?;
    let unit_mm = match rational(exif, Tag::FocalPlaneResolutionUnit).unwrap_or(2.0) as u32 {
        3 => 10.0,
        4 => 1.0,
        5 => 0.001,
        _ => 25.4,
    };
    let source_width = rational(exif, Tag::PixelXDimension).unwrap_or(f64::from(width));
    let px_per_mm = per_unit / unit_mm * f64::from(width) / source_width.max(1.0);
    Some(focal_mm * px_per_mm)
}

fn gps(exif: &exif::Exif) -> Option<[f64; 3]> {
    let degrees = |tag: Tag| -> Option<f64> {
        match &exif.get_field(tag, In::PRIMARY)?.value {
            Value::Rational(v) if v.len() == 3 => {
                Some(v[0].to_f64() + v[1].to_f64() / 60.0 + v[2].to_f64() / 3600.0)
            }
            _ => None,
        }
    };
    let sign = |tag: Tag, negative: &str| -> f64 {
        match exif.get_field(tag, In::PRIMARY) {
            Some(f) if f.display_value().to_string().starts_with(negative) => -1.0,
            _ => 1.0,
        }
    };
    let lat = degrees(Tag::GPSLatitude)? * sign(Tag::GPSLatitudeRef, "S");
    let lon = degrees(Tag::GPSLongitude)? * sign(Tag::GPSLongitudeRef, "W");
    let below = matches!(&exif.get_field(Tag::GPSAltitudeRef, In::PRIMARY).map(|f| &f.value), Some(Value::Byte(v)) if v.first() == Some(&1));
    let altitude = rational(exif, Tag::GPSAltitude).map_or(0.0, |a| if below { -a } else { a });
    (lat.is_finite() && lon.is_finite() && (lat != 0.0 || lon != 0.0))
        .then_some([lat, lon, altitude])
}

/// A number in an XMP attribute (`name="+12.3"`) or element (`<name>12.3</name>`).
fn xmp_number(bytes: &[u8], name: &str) -> Option<f64> {
    let text = String::from_utf8_lossy(bytes);
    let start = text.find(name)? + name.len();
    let rest = &text[start..];
    let rest = rest.trim_start_matches(['=', '"', '>', ' ']);
    let end = rest.find(|c: char| !(c.is_ascii_digit() || matches!(c, '+' | '-' | '.')))?;
    rest[..end].parse().ok().filter(|v: &f64| v.is_finite())
}

#[cfg(test)]
mod tests;
