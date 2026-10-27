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
//! brightest first, then by identifier. Each record is 22 bytes: identifier
//! (u16), right ascension (u32, 2^32 per turn), declination (i32,
//! (2^31 - 1) per quarter turn), proper motion in right ascension times
//! cos(declination) and in declination (i16 each, mas/yr), V magnitude (i16,
//! centimagnitudes), B-V (i16, centimagnitudes, `i16::MIN` if absent),
//! trigonometric parallax (u16, mas, `u16::MAX` if absent or dynamical), and
//! heliocentric radial velocity (i16, km/s, `i16::MIN` if absent).

use std::{f64::consts::PI, path::PathBuf};

const HEADER_BYTES: usize = 32;
const RECORD_BYTES: usize = 22;

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
        let mut table = Vec::with_capacity(RECORD_BYTES);
        table.extend_from_slice(&(hr as u16).to_le_bytes());
        table.extend_from_slice(
            &((ra / (2.0 * PI) * 4_294_967_296.0).round() as u64 as u32).to_le_bytes(),
        );
        table.extend_from_slice(
            &((dec / (PI / 2.0) * f64::from(i32::MAX)).round() as i32).to_le_bytes(),
        );
        for (first, last, name) in [(149, 154, "pmRA"), (155, 160, "pmDE")] {
            table.extend_from_slice(
                &((number(first, last, name)? * 1000.0).round() as i16).to_le_bytes(),
            );
        }
        let magnitude = (number(103, 107, "Vmag")? * 100.0).round() as i16;
        table.extend_from_slice(&magnitude.to_le_bytes());
        table.extend_from_slice(&b_minus_v.to_le_bytes());
        // A dynamical parallax (flag D) is a model value, not a measurement.
        let parallax = if field(record, 162, 166).is_empty() || field(record, 161, 161) == "D" {
            u16::MAX
        } else {
            (number(162, 166, "Parallax")? * 1000.0).round() as u16
        };
        let radial_velocity = if field(record, 167, 170).is_empty() {
            i16::MIN
        } else {
            number(167, 170, "RadVel")?.round() as i16
        };
        table.extend_from_slice(&parallax.to_le_bytes());
        table.extend_from_slice(&radial_velocity.to_le_bytes());
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
