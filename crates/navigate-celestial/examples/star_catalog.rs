//! Pack the Bright Star Catalogue into the binary table that the crate embeds.
//!
//! ```text
//! curl -O https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz && gunzip catalog.gz
//! cargo run -p navigate-celestial --example star_catalog -- \
//!     catalog crates/navigate-celestial/data/bsc5.bin
//! ```
//!
//! The input is the CDS file `V/50/catalog` (Hoffleit and Warren 1991) with
//! SHA-256 `69797549cc1605aad7ff94e9325e29a1661f2a253917faaa056d9bf20b809afd`.
//! Entries without a J2000 position are omitted. Each output record is 18
//! little-endian bytes: HR number (u16), right ascension (u32, 2^32 per
//! turn), declination (i32, (2^31 - 1) per quarter turn), proper motion in
//! right ascension times cos(declination) and in declination (i16 each,
//! mas/yr), V magnitude (i16, centimagnitudes), and B-V (i16,
//! centimagnitudes, `i16::MIN` if absent).

use std::{f64::consts::PI, path::PathBuf};

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("usage: star_catalog <V/50 catalog> <output>")]
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
    let mut table = Vec::new();
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
        table
            .extend_from_slice(&((number(103, 107, "Vmag")? * 100.0).round() as i16).to_le_bytes());
        table.extend_from_slice(&b_minus_v.to_le_bytes());
    }
    std::fs::write(output, table).map_err(|source| Error::Write {
        path: output.into(),
        source,
    })
}

/// The trimmed text in 1-based inclusive byte columns, as in the CDS ReadMe.
fn field(record: &str, first: usize, last: usize) -> &str {
    record
        .get(first - 1..last.min(record.len()))
        .unwrap_or("")
        .trim()
}
