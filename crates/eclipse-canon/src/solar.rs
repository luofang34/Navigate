//! The solar eclipses of the Five Millennium Canon.
//!
//! A record is 36 bytes, little-endian: the catalogue number (u16); the year
//! (i16), month (u8) and day (u8); the TD of greatest eclipse in seconds
//! after 0h (u32); Delta T in seconds (i32); the lunation number (i32); the
//! Saros series (i16); the type and fortnight codes (2 ASCII bytes each, a
//! space for no second character); gamma (i16, 1e-4); the magnitude (u16,
//! 1e-4); the latitude (i8, degrees); the Sun's altitude (u8, degrees); the
//! longitude (i16, degrees); the path width (u16, km) and the central
//! duration (u16, s), each `u16::MAX` when absent.

use crate::{CalendarDate, Code, table};

#[cfg(test)]
mod tests;

const RECORD_BYTES: usize = 36;
type Record = [u8; RECORD_BYTES];

const RECORDS: &[Record] = table::records(include_bytes!("../data/solar.bin"), b"CANONSE1");

/// The kind of a solar eclipse, from the first character of its type code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SolarKind {
    /// No part of the Earth is in the Moon's umbra or antumbra (`P`).
    Partial,
    /// The Moon's antumbra reaches the Earth (`A`).
    Annular,
    /// The Moon's umbra reaches the Earth (`T`).
    Total,
    /// Annular along part of the path and total along the rest (`H`).
    Hybrid,
}

/// One row of the catalogue of solar eclipses.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SolarEclipse {
    /// The catalogue number, from 1 for the first eclipse of -1999.
    pub number: u16,
    /// The date of greatest eclipse in TD.
    pub date: CalendarDate,
    /// The TD of greatest eclipse, in seconds after 0h of [`Self::date`].
    pub greatest_eclipse_td_s: u32,
    /// The Delta T (TD - UT) that the catalogue uses, in seconds.
    pub delta_t_s: i32,
    /// The lunation number, 0 for the new Moon of 2000 January 6.
    pub lunation: i32,
    /// The Saros series.
    pub saros: i16,
    /// The kind of eclipse.
    pub kind: SolarKind,
    /// The catalogue's full type code, for example `T`, `Am` or `H2`.
    pub type_code: Code,
    /// The code of the lunar eclipses a fortnight before and after.
    pub fortnight_code: Code,
    /// The least distance of the axis of the Moon's shadow from the Earth's
    /// centre, in equatorial radii of the Earth, positive where the axis
    /// passes north of the centre.
    pub gamma: f64,
    /// The eclipse magnitude at greatest eclipse: the fraction of the Sun's
    /// diameter that the Moon covers, or for a central eclipse the ratio of
    /// the apparent diameters of the Moon and the Sun.
    pub magnitude: f64,
    /// The latitude of greatest eclipse, in degrees, positive north.
    pub latitude_deg: i8,
    /// The longitude of greatest eclipse, in degrees, positive east.
    pub longitude_deg: i16,
    /// The Sun's altitude at greatest eclipse, in degrees.
    pub sun_altitude_deg: u8,
    /// The width of the central path at greatest eclipse, in kilometres.
    pub path_width_km: Option<u16>,
    /// The duration of the central phase at greatest eclipse, in seconds.
    pub central_duration_s: Option<u16>,
}

impl SolarEclipse {
    /// The eclipse with catalogue number `number`, if the catalogue has it.
    #[must_use]
    pub fn by_number(number: u16) -> Option<Self> {
        let record = RECORDS.get(usize::from(number).checked_sub(1)?)?;
        Some(decode(record)).filter(|eclipse| eclipse.number == number)
    }

    /// The Julian date of greatest eclipse in TT, which is TD.
    #[must_use]
    pub fn greatest_eclipse_jd_tt(&self) -> f64 {
        self.date.julian_date_at_midnight() + f64::from(self.greatest_eclipse_td_s) / 86_400.0
    }
}

/// Every solar eclipse of the catalogue, -1999 to 3000, in date order.
pub fn solar_eclipses() -> impl ExactSizeIterator<Item = SolarEclipse> + DoubleEndedIterator + Clone
{
    RECORDS.iter().map(decode)
}

fn decode(record: &Record) -> SolarEclipse {
    let type_code = Code(table::bytes(record, 20));
    SolarEclipse {
        number: table::u16_at(record, 0),
        date: table::date(record),
        greatest_eclipse_td_s: table::u32_at(record, 6),
        delta_t_s: table::i32_at(record, 10),
        lunation: table::i32_at(record, 14),
        saros: table::i16_at(record, 18),
        kind: match type_code.first() {
            b'A' => SolarKind::Annular,
            b'T' => SolarKind::Total,
            b'H' => SolarKind::Hybrid,
            _ => SolarKind::Partial,
        },
        type_code,
        fortnight_code: Code(table::bytes(record, 22)),
        gamma: f64::from(table::i16_at(record, 24)) * 1e-4,
        magnitude: f64::from(table::u16_at(record, 26)) * 1e-4,
        latitude_deg: i8::from_le_bytes(table::bytes(record, 28)),
        sun_altitude_deg: u8::from_le_bytes(table::bytes(record, 29)),
        longitude_deg: table::i16_at(record, 30),
        path_width_km: table::optional_u16(record, 32),
        central_duration_s: table::optional_u16(record, 34),
    }
}
