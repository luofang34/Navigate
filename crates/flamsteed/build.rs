//! Check the catalogue table and embed the stars that the features select.
//!
//! The records are brightest first, so a magnitude limit keeps a prefix.
//! Without a magnitude feature the limit is V = 4.0. The `magnitude-5`
//! feature raises it to 5.0, and `all-stars` keeps every record. The output
//! has a new header with the record count and CRC-32 of the prefix.

use std::{io::Write as _, path::PathBuf};

const HEADER_BYTES: usize = 32;
const RECORD_BYTES: usize = 34;
/// The byte offset of the V magnitude in a record.
const MAGNITUDE_OFFSET: usize = 26;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = std::io::stdout();
    writeln!(stdout, "cargo:rerun-if-changed=data/bsc5.bin")?;
    let table = std::fs::read("data/bsc5.bin")?;
    let (header, body) = table
        .split_at_checked(HEADER_BYTES)
        .ok_or("data/bsc5.bin is shorter than its header")?;
    let word = |start: usize| -> Result<u32, String> {
        let bytes = header.get(start..start + 4).ok_or("short header")?;
        Ok(u32::from_le_bytes(
            bytes.try_into().map_err(|_| "short header")?,
        ))
    };
    if header.get(..8) != Some(b"FLAMSTED".as_slice()) || header.get(8..12) != Some(&[1, 0, 34, 0])
    {
        return Err("data/bsc5.bin is not a format version 1 table with 34-byte records".into());
    }
    if usize::try_from(word(12)?)? * RECORD_BYTES != body.len() {
        return Err("data/bsc5.bin has a record count that does not match its length".into());
    }
    if word(24)? != crc32(body) {
        return Err("data/bsc5.bin fails its CRC-32".into());
    }
    let limit = if std::env::var_os("CARGO_FEATURE_ALL_STARS").is_some() {
        i16::MAX
    } else if std::env::var_os("CARGO_FEATURE_MAGNITUDE_5").is_some() {
        500
    } else {
        400
    };
    let kept = body
        .as_chunks::<RECORD_BYTES>()
        .0
        .iter()
        .take_while(|record| {
            let magnitude = [record[MAGNITUDE_OFFSET], record[MAGNITUDE_OFFSET + 1]];
            i16::from_le_bytes(magnitude) <= limit
        })
        .count();
    let prefix = body
        .get(..kept * RECORD_BYTES)
        .ok_or("prefix outside the table")?;
    let mut out = Vec::with_capacity(HEADER_BYTES + prefix.len());
    out.extend_from_slice(header.get(..12).ok_or("short header")?);
    out.extend_from_slice(&u32::try_from(kept)?.to_le_bytes());
    out.extend_from_slice(header.get(16..24).ok_or("short header")?);
    out.extend_from_slice(&crc32(prefix).to_le_bytes());
    out.extend_from_slice(header.get(28..32).ok_or("short header")?);
    out.extend_from_slice(prefix);
    let path = PathBuf::from(std::env::var("OUT_DIR")?).join("stars.bin");
    std::fs::write(path, out)?;
    Ok(())
}

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320).
fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(u32::MAX, |crc, &byte| {
        (0..8).fold(crc ^ u32::from(byte), |crc, _| {
            (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg())
        })
    })
}
