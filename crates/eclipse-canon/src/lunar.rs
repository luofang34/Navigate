//! The lunar eclipses of the Five Millennium Canon.
//!
//! A record is 40 bytes, little-endian. The first 26 bytes are as for a
//! solar eclipse, to gamma. Then: the penumbral and umbral magnitudes (i16,
//! 1e-4); the penumbral, partial and total durations (u16, 0.1 minute,
//! `u16::MAX` when absent); the latitude where the Moon is in the zenith
//! (i8, degrees); a zero byte; and the longitude (i16, degrees).

use crate::{CalendarDate, Code, table};

#[cfg(test)]
mod tests;

const RECORD_BYTES: usize = 40;
type Record = [u8; RECORD_BYTES];

const RECORDS: &[Record] = table::records(include_bytes!("../data/lunar.bin"), b"CANONLE1");

/// The kind of a lunar eclipse, from the first character of its type code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LunarKind {
    /// The Moon enters only the Earth's penumbra (`N`).
    Penumbral,
    /// Part of the Moon enters the Earth's umbra (`P`).
    Partial,
    /// All of the Moon enters the Earth's umbra (`T`).
    Total,
}

/// One row of the catalogue of lunar eclipses.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct LunarEclipse {
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
    pub kind: LunarKind,
    /// The catalogue's full type code, for example `N`, `Nx` or `T+`.
    pub type_code: Code,
    /// The code of the solar eclipses a fortnight before and after.
    pub fortnight_code: Code,
    /// The least distance of the Moon's centre from the axis of the Earth's
    /// shadow, in equatorial radii of the Earth, positive where the Moon
    /// passes north of the axis.
    pub gamma: f64,
    /// The penumbral magnitude: the fraction of the Moon's diameter in the
    /// penumbra at greatest eclipse.
    pub penumbral_magnitude: f64,
    /// The umbral magnitude: the fraction of the Moon's diameter in the
    /// umbra at greatest eclipse, negative when the Moon misses the umbra.
    pub umbral_magnitude: f64,
    /// The duration of the penumbral phase, in minutes.
    pub penumbral_duration_min: Option<f64>,
    /// The duration of the partial phase, in minutes.
    pub partial_duration_min: Option<f64>,
    /// The duration of the total phase, in minutes.
    pub total_duration_min: Option<f64>,
    /// The latitude where the Moon is in the zenith at greatest eclipse, in
    /// degrees, positive north.
    pub zenith_latitude_deg: i8,
    /// The longitude where the Moon is in the zenith at greatest eclipse, in
    /// degrees, positive east.
    pub zenith_longitude_deg: i16,
}

impl LunarEclipse {
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

/// Every lunar eclipse of the catalogue, -1999 to 3000, in date order.
pub fn lunar_eclipses() -> impl ExactSizeIterator<Item = LunarEclipse> + DoubleEndedIterator + Clone
{
    RECORDS.iter().map(decode)
}

fn minutes(record: &Record, offset: usize) -> Option<f64> {
    table::optional_u16(record, offset).map(|tenths| f64::from(tenths) / 10.0)
}

fn decode(record: &Record) -> LunarEclipse {
    let type_code = Code(table::bytes(record, 20));
    LunarEclipse {
        number: table::u16_at(record, 0),
        date: table::date(record),
        greatest_eclipse_td_s: table::u32_at(record, 6),
        delta_t_s: table::i32_at(record, 10),
        lunation: table::i32_at(record, 14),
        saros: table::i16_at(record, 18),
        kind: match type_code.first() {
            b'P' => LunarKind::Partial,
            b'T' => LunarKind::Total,
            _ => LunarKind::Penumbral,
        },
        type_code,
        fortnight_code: Code(table::bytes(record, 22)),
        gamma: f64::from(table::i16_at(record, 24)) * 1e-4,
        penumbral_magnitude: f64::from(table::i16_at(record, 26)) * 1e-4,
        umbral_magnitude: f64::from(table::i16_at(record, 28)) * 1e-4,
        penumbral_duration_min: minutes(record, 30),
        partial_duration_min: minutes(record, 32),
        total_duration_min: minutes(record, 34),
        zenith_latitude_deg: i8::from_le_bytes(table::bytes(record, 36)),
        zenith_longitude_deg: table::i16_at(record, 38),
    }
}
