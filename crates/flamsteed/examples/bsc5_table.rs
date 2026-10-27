//! Pack the Bright Star Catalogue into the binary table that flamsteed embeds.
//!
//! ```text
//! curl -O https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz && gunzip catalog.gz
//! cargo run -p flamsteed --example bsc5_table -- catalog crates/flamsteed/data/bsc5.bin
//! ```
//!
//! The input is the CDS file `V/50/catalog` (Hoffleit and Warren 1991) with
//! SHA-256 `69797549cc1605aad7ff94e9325e29a1661f2a253917faaa056d9bf20b809afd`.
//! Entries without a J2000 position are omitted.
//!
//! The file is format version 1, all little-endian. A 32-byte header gives
//! the magic `FLAMSTED`, the format version (u16), the record size (u16), the
//! record count (u32), the reference epoch in Julian years from J2000 (f64),
//! the CRC-32 (IEEE) of the records (u32), and the source (u32, 1 for this
//! catalogue, whose identifiers are HR numbers). The records follow,
//! brightest first, then by identifier. Each record is 34 bytes: identifier
//! (u16); the J2000 unit direction (three i32, 2^31 - 1 for 1); the space
//! velocity in 1e-12 reference distances per Julian year (three i32), with
//! the light-time factor and the radial term applied; V magnitude (i16,
//! centimagnitudes); B-V (i16, centimagnitudes, `i16::MIN` if absent);
//! trigonometric parallax (u16, mas, `u16::MAX` if absent or dynamical); and
//! heliocentric radial velocity (i16, km/s, `i16::MIN` if absent). No
//! trigonometry is necessary to read a record.

use std::{f64::consts::PI, path::PathBuf};

const HEADER_BYTES: usize = 32;
const RECORD_BYTES: usize = 34;
/// The velocity unit: reference distances per Julian year.
const VELOCITY_UNIT: f64 = 1e-12;
/// Parsecs per Julian year at 1 km/s.
const PARSEC_PER_YEAR_PER_KM_S: f64 = 1.022_712_165_053_707_7e-6;
const LIGHT_SPEED_KM_S: f64 = 299_792.458;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("usage: bsc5_table <V/50 catalog> <output>")]
    Arguments,
    #[error("cannot read {}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("line {line}: invalid {field} field")]
    Field { line: usize, field: &'static str },
    #[error("cannot write {}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output] = args.as_slice() else {
        return Err(Error::Arguments);
    };
    let text = std::fs::read_to_string(input).map_err(|source| Error::Read {
        path: input.into(),
        source,
    })?;
    let mut records: Vec<(i16, u16, Vec<u8>)> = Vec::new();
    for (index, record) in text.lines().enumerate() {
        let line = index + 1;
        // Bytes 76 to 90 hold the J2000 position. Withdrawn entries leave it blank.
        if field(record, 76, 77).is_empty() {
            continue;
        }
        let number = |first, last, name| {
            field(record, first, last)
                .parse::<f64>()
                .map_err(|_| Error::Field { line, field: name })
        };
        let hr = number(1, 4, "HR")?;
        let ra = (number(76, 77, "RAh")?
            + number(78, 79, "RAm")? / 60.0
            + number(80, 83, "RAs")? / 3600.0)
            * PI
            / 12.0;
        let sign = if field(record, 84, 84) == "-" {
            -1.0
        } else {
            1.0
        };
        let dec = sign
            * (number(85, 86, "DEd")?
                + number(87, 88, "DEm")? / 60.0
                + number(89, 90, "DEs")? / 3600.0)
                .to_radians();
        let b_minus_v = if field(record, 110, 114).is_empty() {
            i16::MIN
        } else {
            (number(110, 114, "B-V")? * 100.0).round() as i16
        };
        let pm_ra = number(149, 154, "pmRA")? / 3600.0 * PI / 180.0;
        let pm_dec = number(155, 160, "pmDE")? / 3600.0 * PI / 180.0;
        let magnitude = (number(103, 107, "Vmag")? * 100.0).round() as i16;
        // A dynamical parallax (flag D) is a model value, not a measurement.
        let parallax_mas = if field(record, 162, 166).is_empty() || field(record, 161, 161) == "D" {
            None
        } else {
            Some(number(162, 166, "Parallax")? * 1000.0)
        };
        let radial_velocity = if field(record, 167, 170).is_empty() {
            None
        } else {
            Some(number(167, 170, "RadVel")?)
        };
        let (position, velocity) =
            space_motion(ra, dec, pm_ra, pm_dec, parallax_mas, radial_velocity);
        let mut table = Vec::with_capacity(RECORD_BYTES);
        table.extend_from_slice(&(hr as u16).to_le_bytes());
        for component in position {
            table.extend_from_slice(
                &((component * f64::from(i32::MAX)).round() as i32).to_le_bytes(),
            );
        }
        for component in velocity {
            table.extend_from_slice(&((component / VELOCITY_UNIT).round() as i32).to_le_bytes());
        }
        table.extend_from_slice(&magnitude.to_le_bytes());
        table.extend_from_slice(&b_minus_v.to_le_bytes());
        table.extend_from_slice(
            &parallax_mas
                .map_or(u16::MAX, |p| p.round() as u16)
                .to_le_bytes(),
        );
        table.extend_from_slice(
            &radial_velocity
                .map_or(i16::MIN, |v| v.round() as i16)
                .to_le_bytes(),
        );
        records.push((magnitude, hr as u16, table));
    }
    records.sort_by_key(|&(magnitude, hr, _)| (magnitude, hr));
    let body: Vec<u8> = records
        .into_iter()
        .flat_map(|(_, _, record)| record)
        .collect();
    let mut file = Vec::with_capacity(HEADER_BYTES + body.len());
    file.extend_from_slice(b"FLAMSTED");
    file.extend_from_slice(&1_u16.to_le_bytes());
    file.extend_from_slice(&(RECORD_BYTES as u16).to_le_bytes());
    file.extend_from_slice(&((body.len() / RECORD_BYTES) as u32).to_le_bytes());
    file.extend_from_slice(&0.0_f64.to_le_bytes());
    file.extend_from_slice(&crc32(&body).to_le_bytes());
    file.extend_from_slice(&1_u32.to_le_bytes());
    file.extend_from_slice(&body);
    std::fs::write(output, file).map_err(|source| Error::Write {
        path: output.into(),
        source,
    })
}

/// The J2000 unit direction and the space velocity, in reference distances
/// per Julian year. The velocity holds the proper motion and, where both a
/// trigonometric parallax and a radial velocity exist, the radial term, all
/// scaled by the light-time factor 1 / (1 - v_r / c) (Stumpff 1985; SOFA
/// eraStarpv).
fn space_motion(
    ra: f64,
    dec: f64,
    pm_ra: f64,
    pm_dec: f64,
    parallax_mas: Option<f64>,
    radial_velocity: Option<f64>,
) -> ([f64; 3], [f64; 3]) {
    let (sin_ra, cos_ra) = ra.sin_cos();
    let (sin_dec, cos_dec) = dec.sin_cos();
    let position = [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec];
    let east = [-sin_ra, cos_ra, 0.0];
    let north = [-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec];
    let (light_time, radial_rate) = match (parallax_mas, radial_velocity) {
        (Some(parallax), Some(velocity)) if parallax > 0.0 => {
            let light_time = 1.0 / (1.0 - velocity / LIGHT_SPEED_KM_S);
            (
                light_time,
                light_time * velocity * parallax / 1000.0 * PARSEC_PER_YEAR_PER_KM_S,
            )
        }
        _ => (1.0, 0.0),
    };
    let velocity = std::array::from_fn(|i| {
        position[i] * radial_rate + light_time * (east[i] * pm_ra + north[i] * pm_dec)
    });
    (position, velocity)
}

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320).
fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(u32::MAX, |crc, &byte| {
        (0..8).fold(crc ^ u32::from(byte), |crc, _| {
            (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg())
        })
    })
}

/// The trimmed text in 1-based inclusive byte columns, as in the CDS ReadMe.
fn field(record: &str, first: usize, last: usize) -> &str {
    record
        .get(first - 1..last.min(record.len()))
        .unwrap_or("")
        .trim()
}
