//! Pack the Five Millennium Canon catalogues into the binary tables that
//! eclipse-canon embeds.
//!
//! ```text
//! cargo run -p eclipse-canon --example canon_tables -- PAGES crates/eclipse-canon/data
//! ```
//!
//! PAGES holds the 100 century pages `SE-1999--1900.html` to
//! `SE2901-3000.html` from <https://eclipse.gsfc.nasa.gov/SEcat5/> and
//! `LE-1999--1900.html` to `LE2901-3000.html` from
//! <https://eclipse.gsfc.nasa.gov/LEcat5/>. Eclipse Predictions by Fred
//! Espenak (NASA's GSFC).
//!
//! Each output file is little-endian: a 16-byte header with the magic
//! (`CANONSE1` or `CANONLE1`), the record count (u32), the record size (u16)
//! and a reserved u16 (0), then one record per catalogue row in catalogue
//! order. The record layouts are in `src/solar.rs` and `src/lunar.rs`.

use std::path::{Path, PathBuf};

const FIRST_YEAR: i32 = -1999;
const LAST_YEAR: i32 = 3000;
const SOLAR_BYTES: u16 = 36;
const LUNAR_BYTES: u16 = 40;
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("usage: canon_tables <directory of catalogue pages> <output directory>")]
    Arguments,
    #[error("cannot read {}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{page} catalogue number {number}: invalid {field} {value:?}")]
    Field {
        page: String,
        number: String,
        field: &'static str,
        value: String,
    },
    #[error("{kind} rows are numbered {found} where {expected} is next")]
    Sequence {
        kind: &'static str,
        expected: usize,
        found: u16,
    },
    #[error("cannot write {}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// One catalogue row: its page, for error messages, and its tokens.
struct Row<'a> {
    page: &'a str,
    tokens: Vec<&'a str>,
}

impl Row<'_> {
    fn error(&self, field: &'static str, index: usize) -> Error {
        Error::Field {
            page: self.page.to_owned(),
            number: self.tokens.first().copied().unwrap_or_default().to_owned(),
            field,
            value: self
                .tokens
                .get(index)
                .copied()
                .unwrap_or_default()
                .to_owned(),
        }
    }

    fn token(&self, field: &'static str, index: usize) -> Result<&str, Error> {
        self.tokens
            .get(index)
            .copied()
            .ok_or_else(|| self.error(field, index))
    }

    fn int<T: std::str::FromStr>(&self, field: &'static str, index: usize) -> Result<T, Error> {
        self.token(field, index)?
            .parse()
            .map_err(|_| self.error(field, index))
    }

    /// A decimal field in units of `scale`, as an integer.
    fn scaled<T: TryFrom<i64>>(
        &self,
        field: &'static str,
        index: usize,
        scale: f64,
    ) -> Result<T, Error> {
        let value: f64 = self.int(field, index)?;
        let units = (value * scale).round();
        T::try_from(units as i64).map_err(|_| self.error(field, index))
    }

    /// A decimal field in units of `scale`, or `u16::MAX` for `-`.
    fn optional(&self, field: &'static str, index: usize, scale: f64) -> Result<u16, Error> {
        match self.tokens.get(index) {
            None | Some(&"-") => Ok(u16::MAX),
            Some(_) => self.scaled(field, index, scale),
        }
    }

    /// `37N` as 37 and `88W` as -88.
    fn angle<T: TryFrom<i64>>(
        &self,
        field: &'static str,
        index: usize,
        positive: char,
        negative: char,
    ) -> Result<T, Error> {
        let token = self.token(field, index)?;
        let (digits, hemisphere) = token.split_at(token.len().saturating_sub(1));
        let degrees: i64 = digits.parse().map_err(|_| self.error(field, index))?;
        let signed = match hemisphere.chars().next() {
            Some(c) if c == positive => degrees,
            Some(c) if c == negative => -degrees,
            _ => return Err(self.error(field, index)),
        };
        T::try_from(signed).map_err(|_| self.error(field, index))
    }

    /// A one- or two-character code, with a space for a missing second character.
    fn code(&self, field: &'static str, index: usize) -> Result<[u8; 2], Error> {
        match self.token(field, index)?.as_bytes() {
            [a] => Ok([*a, b' ']),
            [a, b] => Ok([*a, *b]),
            _ => Err(self.error(field, index)),
        }
    }

    /// The fields that solar and lunar rows share: number, date, TD of
    /// greatest eclipse, Delta T, lunation, Saros series, type and the code
    /// of the eclipses a fortnight away. They take the first 20 bytes and the
    /// 4 after.
    fn common(&self, record: &mut Vec<u8>) -> Result<(), Error> {
        let year: i16 = self.int("year", 1)?;
        let month = MONTHS
            .iter()
            .position(|&m| m == self.token("month", 2).unwrap_or_default())
            .ok_or_else(|| self.error("month", 2))?;
        let day: u8 = self.int("day", 3)?;
        let td = self.token("time", 4)?;
        let parts: Vec<u32> = td
            .split(':')
            .map(str::parse)
            .collect::<Result<_, _>>()
            .map_err(|_| self.error("time", 4))?;
        let &[h, m, s] = parts.as_slice() else {
            return Err(self.error("time", 4));
        };
        record.extend_from_slice(&self.int::<u16>("number", 0)?.to_le_bytes());
        record.extend_from_slice(&year.to_le_bytes());
        record.push(u8::try_from(month + 1).map_err(|_| self.error("month", 2))?);
        record.push(day);
        record.extend_from_slice(&(h * 3600 + m * 60 + s).to_le_bytes());
        record.extend_from_slice(&self.int::<i32>("Delta T", 5)?.to_le_bytes());
        record.extend_from_slice(&self.int::<i32>("lunation", 6)?.to_le_bytes());
        record.extend_from_slice(&self.int::<i16>("Saros", 7)?.to_le_bytes());
        record.extend_from_slice(&self.code("type", 8)?);
        record.extend_from_slice(&self.code("fortnight code", 9)?);
        record.extend_from_slice(&self.scaled::<i16>("gamma", 10, 1e4)?.to_le_bytes());
        Ok(())
    }

    fn solar(&self) -> Result<Vec<u8>, Error> {
        let mut record = Vec::with_capacity(usize::from(SOLAR_BYTES));
        self.common(&mut record)?;
        record.extend_from_slice(&self.scaled::<u16>("magnitude", 11, 1e4)?.to_le_bytes());
        record.push(self.angle::<i8>("latitude", 12, 'N', 'S')? as u8);
        record.push(self.int::<u8>("Sun altitude", 14)?);
        record.extend_from_slice(&self.angle::<i16>("longitude", 13, 'E', 'W')?.to_le_bytes());
        record.extend_from_slice(&self.optional("path width", 15, 1.0)?.to_le_bytes());
        let duration = match self.tokens.get(16) {
            None | Some(&"-") => u16::MAX,
            Some(token) => duration_s(token).ok_or_else(|| self.error("duration", 16))?,
        };
        record.extend_from_slice(&duration.to_le_bytes());
        Ok(record)
    }

    fn lunar(&self) -> Result<Vec<u8>, Error> {
        let mut record = Vec::with_capacity(usize::from(LUNAR_BYTES));
        self.common(&mut record)?;
        record.extend_from_slice(
            &self
                .scaled::<i16>("penumbral magnitude", 11, 1e4)?
                .to_le_bytes(),
        );
        record.extend_from_slice(
            &self
                .scaled::<i16>("umbral magnitude", 12, 1e4)?
                .to_le_bytes(),
        );
        for (field, index) in [("penumbral", 13), ("partial", 14), ("total", 15)] {
            record.extend_from_slice(&self.optional(field, index, 10.0)?.to_le_bytes());
        }
        record.push(self.angle::<i8>("zenith latitude", 16, 'N', 'S')? as u8);
        record.push(0);
        record.extend_from_slice(
            &self
                .angle::<i16>("zenith longitude", 17, 'E', 'W')?
                .to_le_bytes(),
        );
        Ok(record)
    }
}

/// `06m29s` as 389.
fn duration_s(token: &str) -> Option<u16> {
    let (minutes, seconds) = token.strip_suffix('s')?.split_once('m')?;
    Some(minutes.parse::<u16>().ok()? * 60 + seconds.parse::<u16>().ok()?)
}

fn signed(year: i32) -> String {
    if year < 0 {
        format!("-{:04}", -year)
    } else {
        format!("{year:04}")
    }
}

/// The page text without tags. The pages are ASCII apart from entities in
/// the column headings, which no row uses.
fn page_text(path: &Path) -> Result<String, Error> {
    let raw = std::fs::read(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut text = String::with_capacity(raw.len());
    let mut in_tag = false;
    for &byte in &raw {
        match byte {
            b'<' => in_tag = true,
            b'>' => in_tag = false,
            _ if !in_tag => text.push(char::from(byte)),
            _ => {}
        }
    }
    Ok(text)
}

/// Whether a line is a catalogue row: a five-digit number, then a year in
/// the five millennia.
fn row_tokens(line: &str) -> Option<Vec<&str>> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let number = tokens.first()?;
    let year: i32 = tokens.get(1)?.parse().ok()?;
    (number.len() == 5
        && number.bytes().all(|b| b.is_ascii_digit())
        && (FIRST_YEAR..=LAST_YEAR).contains(&year))
    .then_some(tokens)
}

fn table(
    pages: &Path,
    prefix: &str,
    kind: &'static str,
    magic: &[u8; 8],
    record_bytes: u16,
    encode: fn(&Row<'_>) -> Result<Vec<u8>, Error>,
) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    let mut count = 0_usize;
    for start in (FIRST_YEAR..LAST_YEAR).step_by(100) {
        let page = format!("{prefix}{}-{}.html", signed(start), signed(start + 99));
        let text = page_text(&pages.join(&page))?;
        for tokens in text.lines().filter_map(row_tokens) {
            let row = Row {
                page: &page,
                tokens,
            };
            let number: u16 = row.int("number", 0)?;
            if usize::from(number) != count + 1 {
                return Err(Error::Sequence {
                    kind,
                    expected: count + 1,
                    found: number,
                });
            }
            body.extend_from_slice(&encode(&row)?);
            count += 1;
        }
    }
    let mut file = Vec::with_capacity(16 + body.len());
    file.extend_from_slice(magic);
    file.extend_from_slice(&u32::try_from(count).unwrap_or(u32::MAX).to_le_bytes());
    file.extend_from_slice(&record_bytes.to_le_bytes());
    file.extend_from_slice(&0_u16.to_le_bytes());
    file.extend_from_slice(&body);
    Ok(file)
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args_os().skip(1);
    let (Some(pages), Some(output), None) = (args.next(), args.next(), args.next()) else {
        return Err(Error::Arguments);
    };
    let (pages, output) = (PathBuf::from(pages), PathBuf::from(output));
    let tables = [
        (
            "solar.bin",
            table(&pages, "SE", "solar", b"CANONSE1", SOLAR_BYTES, |row| {
                row.solar()
            })?,
        ),
        (
            "lunar.bin",
            table(&pages, "LE", "lunar", b"CANONLE1", LUNAR_BYTES, |row| {
                row.lunar()
            })?,
        ),
    ];
    for (name, file) in tables {
        let path = output.join(name);
        std::fs::write(&path, file).map_err(|source| Error::Write { path, source })?;
    }
    Ok(())
}
