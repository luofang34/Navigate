//! The embedded catalogue and its file format.
//!
//! The source is CDS catalogue V/50 (Hoffleit and Warren 1991), prepared by
//! the NASA Astronomical Data Center. It is a United States government work
//! and is in the public domain. The 9096 entries with a J2000 position are
//! embedded as `data/bsc5.bin`, which the `bsc5_table` example writes; the
//! example documents format version 1.

const BSC5: &[u8] = include_bytes!("../data/bsc5.bin");
const MAGIC: &[u8; 8] = b"FLAMSTED";
const FORMAT_VERSION: u16 = 1;
pub(crate) const HEADER_BYTES: usize = 32;
pub(crate) const RECORD_BYTES: usize = 22;
/// Source 1 is the Bright Star Catalogue, whose identifiers are HR numbers.
const SOURCE_BSC5: u32 = 1;
const MAS: f64 = std::f64::consts::PI / 648_000_000.0;

/// A star identifier, with the catalogue that gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum StarId {
    /// Harvard Revised (HR) number, the Bright Star Catalogue identifier.
    Hr(u16),
}

/// A set of stars, brightest first.
#[derive(Clone, Copy, Debug)]
pub struct Catalog<'a> {
    records: &'a [[u8; RECORD_BYTES]],
    reference_epoch_years: f64,
}

impl Catalog<'static> {
    /// The Yale Bright Star Catalogue: the 9096 entries with a J2000 position,
    /// about every star to visual magnitude 6.5.
    pub fn bright_star_catalogue() -> Self {
        // The tests check the embedded header and checksum.
        parse(BSC5).unwrap_or(Self {
            records: &[],
            reference_epoch_years: 0.0,
        })
    }
}

impl<'a> Catalog<'a> {
    /// The number of stars.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the catalogue has no stars.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The stars, brightest first, then by identifier.
    pub fn stars(&self) -> impl ExactSizeIterator<Item = Star> + 'a {
        let epoch = self.reference_epoch_years;
        self.records.iter().map(move |record| decode(record, epoch))
    }

    /// The stars at least as bright as visual magnitude `v_magnitude`,
    /// brightest first. The records are in magnitude order, so this reads a
    /// prefix and does not decode the fainter stars.
    pub fn brighter_than(&self, v_magnitude: f64) -> impl ExactSizeIterator<Item = Star> + 'a {
        let epoch = self.reference_epoch_years;
        let end = self
            .records
            .partition_point(|record| magnitude(record) <= v_magnitude);
        self.records
            .get(..end)
            .unwrap_or_default()
            .iter()
            .map(move |record| decode(record, epoch))
    }
}

/// One catalogue star at its reference epoch.
///
/// Fields can be added in a later version, so code outside this crate reads
/// a `Star` but does not construct one.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Star {
    /// The catalogue identifier.
    pub id: StarId,
    /// The epoch of the position, in Julian years (TT) from J2000.
    pub reference_epoch_years: f64,
    /// Right ascension at the reference epoch, in radians.
    pub right_ascension_rad: f64,
    /// Declination at the reference epoch, in radians.
    pub declination_rad: f64,
    /// Proper motion in right ascension times cos(declination), in radians
    /// per Julian year.
    pub proper_motion_ra_rad_per_year: f64,
    /// Proper motion in declination, in radians per Julian year.
    pub proper_motion_dec_rad_per_year: f64,
    /// Trigonometric parallax in milliarcseconds, if the catalogue gives one.
    pub parallax_mas: Option<f64>,
    /// Heliocentric radial velocity in km/s, if the catalogue gives one.
    pub radial_velocity_km_s: Option<f64>,
    /// Visual magnitude in the Johnson V band.
    pub v_magnitude: f64,
    /// B-V colour index, if the catalogue gives one.
    pub b_minus_v: Option<f64>,
}

impl Star {
    /// The HR number, if the identifier is one.
    #[inline]
    pub fn hr(&self) -> Option<u16> {
        match self.id {
            StarId::Hr(number) => Some(number),
        }
    }
}

/// The header fields of a format version 1 file, and its records.
fn parse(bytes: &[u8]) -> Option<Catalog<'_>> {
    let (header, body) = bytes.split_at_checked(HEADER_BYTES)?;
    let field = |start: usize, end: usize| header.get(start..end);
    let u16_at = |start| {
        Some(u16::from_le_bytes(
            field(start, start + 2)?.try_into().ok()?,
        ))
    };
    let u32_at = |start| {
        Some(u32::from_le_bytes(
            field(start, start + 4)?.try_into().ok()?,
        ))
    };
    let valid = field(0, 8)? == MAGIC
        && u16_at(8)? == FORMAT_VERSION
        && usize::from(u16_at(10)?) == RECORD_BYTES
        && u32_at(28)? == SOURCE_BSC5;
    let (records, rest) = body.as_chunks::<RECORD_BYTES>();
    let count = usize::try_from(u32_at(12)?).ok()?;
    if !valid || !rest.is_empty() || records.len() != count {
        return None;
    }
    let reference_epoch_years = f64::from_le_bytes(field(16, 24)?.try_into().ok()?);
    Some(Catalog {
        records,
        reference_epoch_years,
    })
}

/// The CRC-32 (IEEE 802.3) of `bytes`.
#[cfg(test)]
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(u32::MAX, |crc, &byte| {
        (0..8).fold(crc ^ u32::from(byte), |crc, _| {
            (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg())
        })
    })
}

#[inline]
fn magnitude(record: &[u8; RECORD_BYTES]) -> f64 {
    let &[.., v0, v1, _, _, _, _, _, _] = record;
    f64::from(i16::from_le_bytes([v0, v1])) / 100.0
}

#[inline]
fn decode(record: &[u8; RECORD_BYTES], reference_epoch_years: f64) -> Star {
    let &[
        h0,
        h1,
        r0,
        r1,
        r2,
        r3,
        d0,
        d1,
        d2,
        d3,
        a0,
        a1,
        p0,
        p1,
        v0,
        v1,
        c0,
        c1,
        x0,
        x1,
        w0,
        w1,
    ] = record;
    let colour = i16::from_le_bytes([c0, c1]);
    let parallax = u16::from_le_bytes([x0, x1]);
    let radial_velocity = i16::from_le_bytes([w0, w1]);
    Star {
        id: StarId::Hr(u16::from_le_bytes([h0, h1])),
        reference_epoch_years,
        right_ascension_rad: f64::from(u32::from_le_bytes([r0, r1, r2, r3])) / 4_294_967_296.0
            * std::f64::consts::TAU,
        declination_rad: f64::from(i32::from_le_bytes([d0, d1, d2, d3])) / f64::from(i32::MAX)
            * std::f64::consts::FRAC_PI_2,
        proper_motion_ra_rad_per_year: f64::from(i16::from_le_bytes([a0, a1])) * MAS,
        proper_motion_dec_rad_per_year: f64::from(i16::from_le_bytes([p0, p1])) * MAS,
        parallax_mas: (parallax != u16::MAX).then(|| f64::from(parallax)),
        radial_velocity_km_s: (radial_velocity != i16::MIN).then(|| f64::from(radial_velocity)),
        v_magnitude: f64::from(i16::from_le_bytes([v0, v1])) / 100.0,
        b_minus_v: (colour != i16::MIN).then(|| f64::from(colour) / 100.0),
    }
}

#[cfg(test)]
mod tests;
