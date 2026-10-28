//! The catalogues and their file format.
//!
//! The embedded source is CDS catalogue V/50 (Hoffleit and Warren 1991),
//! prepared by the NASA Astronomical Data Center. It is a United States
//! government work and is in the public domain. The 9096 entries with a
//! J2000 position are in `data/bsc5.bin`, which the `bsc5_table` example
//! writes in format version 1. The build script embeds the magnitude prefix
//! that the features select. [`Catalog::from_bytes`] also reads format
//! version 2 (`catalog/v2.rs`).
//!
//! A file starts with a 32-byte header, little-endian: the magic `FLAMSTED`,
//! the format version (u16), the record size (u16), the record count (u32),
//! the reference epoch in Julian years from J2000 (f64), the CRC-32 (IEEE) of
//! the records (u32), and the source (u32). The records follow, brightest
//! first.

use crate::{Float, FormatError, Motion, float};

mod v1;
mod v2;

const STARS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stars.bin"));
const MAGIC: &[u8; 8] = b"FLAMSTED";
pub(crate) const HEADER_BYTES: usize = 32;
/// Source 1: Bright Star Catalogue records with HR identifiers.
const SOURCE_BSC5: u32 = 1;
/// Source 3: HR identifiers, with the astrometry of each record from the
/// catalogue that its provenance names.
const SOURCE_BY_RECORD: u32 = 3;

/// The embedded table, checked when the crate compiles.
const EMBEDDED: Result<Catalog<'static>, FormatError> = parse(STARS);
const _: () = assert!(EMBEDDED.is_ok(), "the embedded star table is malformed");

/// A star identifier, with the catalogue that gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum StarId {
    /// Harvard Revised (HR) number, the Bright Star Catalogue identifier.
    Hr(u16),
}

/// The catalogue that gives a star's position and motion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Astrometry {
    /// The Bright Star Catalogue.
    BrightStarCatalogue,
    /// Another catalogue, by the provenance code of the file that holds the
    /// star, with that catalogue's identifier of the star.
    Other {
        /// The provenance code of the file.
        provenance: u16,
        /// The identifier of the star in that catalogue.
        source_id: u64,
    },
}

/// The equatorial axes of a catalogue's positions and motions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Axes {
    /// The mean equator and equinox of J2000, as FK5 realizes them. They
    /// differ from ICRS by less than 0.1 arcsecond.
    Fk5J2000,
    /// The International Celestial Reference System.
    Icrs,
}

#[derive(Clone, Copy, Debug)]
enum Records<'a> {
    V1(&'a [v1::Record]),
    V2(&'a [v2::Record]),
}

#[derive(Clone, Copy)]
enum Record<'a> {
    V1(&'a v1::Record),
    V2(&'a v2::Record),
}

impl Record<'_> {
    #[inline]
    fn magnitude(self) -> f64 {
        match self {
            Self::V1(record) => v1::magnitude(record),
            Self::V2(record) => v2::magnitude(record),
        }
    }

    #[inline]
    fn star(self, reference_epoch_years: f64) -> Star {
        match self {
            Self::V1(record) => v1::star(record, reference_epoch_years),
            Self::V2(record) => v2::star(record, reference_epoch_years),
        }
    }

    #[inline]
    fn motion<T: Float>(self, reference_epoch_years: T) -> Motion<T> {
        match self {
            Self::V1(record) => v1::motion(record, reference_epoch_years),
            Self::V2(record) => v2::motion(record, reference_epoch_years),
        }
    }
}

enum RecordIter<'a> {
    V1(core::slice::Iter<'a, v1::Record>),
    V2(core::slice::Iter<'a, v2::Record>),
}

impl<'a> Iterator for RecordIter<'a> {
    type Item = Record<'a>;

    #[inline]
    fn next(&mut self) -> Option<Record<'a>> {
        match self {
            Self::V1(records) => records.next().map(Record::V1),
            Self::V2(records) => records.next().map(Record::V2),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::V1(records) => records.size_hint(),
            Self::V2(records) => records.size_hint(),
        }
    }
}

impl ExactSizeIterator for RecordIter<'_> {}

impl<'a> Records<'a> {
    const fn len(self) -> usize {
        match self {
            Self::V1(records) => records.len(),
            Self::V2(records) => records.len(),
        }
    }

    fn iter(self) -> RecordIter<'a> {
        match self {
            Self::V1(records) => RecordIter::V1(records.iter()),
            Self::V2(records) => RecordIter::V2(records.iter()),
        }
    }

    /// The records at least as bright as `v_magnitude`, a prefix.
    fn brighter_than(self, v_magnitude: f64) -> Self {
        match self {
            Self::V1(records) => {
                let end = records.partition_point(|r| v1::magnitude(r) <= v_magnitude);
                Self::V1(records.get(..end).unwrap_or_default())
            }
            Self::V2(records) => {
                let end = records.partition_point(|r| v2::magnitude(r) <= v_magnitude);
                Self::V2(records.get(..end).unwrap_or_default())
            }
        }
    }
}

/// A set of stars, brightest first.
#[derive(Clone, Copy, Debug)]
pub struct Catalog<'a> {
    records: Records<'a>,
    reference_epoch_years: f64,
}

impl Catalog<'static> {
    /// The embedded Yale Bright Star Catalogue. With the default `all-stars`
    /// feature it has all 9096 entries with a J2000 position, about every star
    /// to visual magnitude 6.5. With `magnitude-5` it has the stars to
    /// magnitude 5.0, and without either the stars to magnitude 4.0.
    pub const fn bright_star_catalogue() -> Self {
        match EMBEDDED {
            Ok(catalog) => catalog,
            Err(_) => Self {
                records: Records::V1(&[]),
                reference_epoch_years: 0.0,
            },
        }
    }
}

impl<'a> Catalog<'a> {
    /// The catalogue in `bytes`, a file in format version 1, the format of
    /// the embedded table, or format version 2. The catalogue borrows the
    /// bytes and does not copy them, so a file in flash or in a memory map
    /// needs no RAM for its records.
    ///
    /// The function checks the header, the length, the CRC-32 of the
    /// records, a finite reference epoch, the magnitude order that
    /// [`Catalog::brighter_than`] needs, and, in format version 2, that each
    /// position is a unit vector and each velocity is finite. Format version
    /// 1 is in FK5 J2000 axes and version 2 in ICRS axes
    /// ([`Catalog::axes`]).
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, FormatError> {
        let catalog = parse(bytes)?;
        let stored = bytes
            .get(24..28)
            .and_then(|word| word.try_into().ok())
            .map_or(0, u32::from_le_bytes);
        let computed = crc32(bytes.get(HEADER_BYTES..).unwrap_or_default());
        if stored != computed {
            return Err(FormatError::Checksum { stored, computed });
        }
        if !catalog.reference_epoch_years.is_finite() {
            return Err(FormatError::Epoch);
        }
        if let Records::V2(records) = catalog.records
            && let Some(index) = records.iter().position(|record| !v2::is_valid(record))
        {
            return Err(FormatError::Record { index });
        }
        let mut brightest = f64::NEG_INFINITY;
        for (index, record) in catalog.records.iter().enumerate() {
            let magnitude = record.magnitude();
            if magnitude < brightest {
                return Err(FormatError::Order { index });
            }
            brightest = magnitude;
        }
        Ok(catalog)
    }

    /// The equatorial axes of the positions and the motions.
    pub const fn axes(&self) -> Axes {
        match self.records {
            Records::V1(_) => Axes::Fk5J2000,
            Records::V2(_) => Axes::Icrs,
        }
    }

    /// The number of stars.
    pub const fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the catalogue has no stars.
    pub const fn is_empty(&self) -> bool {
        self.records.len() == 0
    }

    /// The stars at least as bright as visual magnitude `v_magnitude`. The
    /// records are in magnitude order, so this is a prefix of the catalogue.
    pub fn brighter_than(&self, v_magnitude: f64) -> Self {
        Self {
            records: self.records.brighter_than(v_magnitude),
            reference_epoch_years: self.reference_epoch_years,
        }
    }

    /// The stars, brightest first, then by identifier.
    pub fn stars(&self) -> impl ExactSizeIterator<Item = Star> + 'a {
        let epoch = self.reference_epoch_years;
        self.records.iter().map(move |record| record.star(epoch))
    }

    /// The motions of the stars in the precision `T`, in the order of
    /// [`Catalog::stars`]. For the embedded table the decode converts
    /// integers to `T` and does no other arithmetic, so `f32` motions need
    /// no double-precision operation. Format version 2 stores `f64`, which
    /// the decode rounds to `T`.
    pub fn motions<T: Float>(&self) -> impl ExactSizeIterator<Item = Motion<T>> + 'a {
        let epoch = T::from_f64(self.reference_epoch_years);
        self.records.iter().map(move |record| record.motion(epoch))
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
    /// The catalogue that gives the position and the motion.
    pub astrometry: Astrometry,
    /// The epoch of the position, in Julian years (TT) from J2000.
    pub reference_epoch_years: f64,
    /// The unit direction at the reference epoch, in the axes of the
    /// catalogue ([`Catalog::axes`]).
    pub position: [f64; 3],
    /// The space velocity in reference distances per Julian year: the proper
    /// motion and, where the catalogue gives a trigonometric parallax and a
    /// radial velocity, the radial term, with the light-time factor applied.
    pub velocity: [f64; 3],
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

    /// The distance at the reference epoch, in parsecs, if the catalogue gives
    /// a trigonometric parallax.
    #[inline]
    pub fn distance_pc(&self) -> Option<f64> {
        self.parallax_mas
            .filter(|parallax| *parallax > 0.0)
            .map(|parallax| 1000.0 / parallax)
    }

    /// The position from the solar system barycentre `years` Julian years
    /// (TT) after J2000, in parsecs, in the axes of the catalogue, if the
    /// catalogue gives a trigonometric parallax.
    ///
    /// The star moves on the straight line of [`Star::motion`]. Without a
    /// radial velocity the line has no radial part. The catalogue parallax
    /// limits the accuracy: the embedded table stores it to 1
    /// milliarcsecond, and the distance error grows as the square of the
    /// distance.
    #[inline]
    pub fn barycentric_position_pc_at(&self, years: f64) -> Option<[f64; 3]> {
        let distance = self.distance_pc()?;
        Some(self.motion().position_at(years).map(|c| c * distance))
    }

    /// Right ascension at the reference epoch, in radians from 0 to 2 pi.
    pub fn right_ascension_rad(&self) -> f64 {
        let [x, y, _] = self.position;
        let angle = float::atan2(y, x);
        if angle < 0.0 {
            angle + core::f64::consts::TAU
        } else {
            angle
        }
    }

    /// Declination at the reference epoch, in radians.
    pub fn declination_rad(&self) -> f64 {
        let [x, y, z] = self.position;
        float::atan2(z, (x * x + y * y).sqrt())
    }

    /// Proper motion in right ascension times cos(declination) and in
    /// declination, in radians per Julian year, with the light-time factor.
    pub fn proper_motion_rad_per_year(&self) -> [f64; 2] {
        let [x, y, z] = self.position;
        let rho = (x * x + y * y).sqrt();
        let east = [-y / rho, x / rho, 0.0];
        let north = [-z * x / rho, -z * y / rho, rho];
        let dot = |axis: [f64; 3]| axis.iter().zip(self.velocity).map(|(a, v)| a * v).sum();
        [dot(east), dot(north)]
    }
}

/// The catalogue of a file, if the header and the length agree. The build
/// script and [`Catalog::from_bytes`] check the CRC-32.
const fn parse(bytes: &[u8]) -> Result<Catalog<'_>, FormatError> {
    let Some((header, body)) = bytes.split_at_checked(HEADER_BYTES) else {
        return Err(FormatError::TooShort {
            length: bytes.len(),
        });
    };
    let mut index = 0;
    while index < MAGIC.len() {
        if header[index] != MAGIC[index] {
            return Err(FormatError::Magic);
        }
        index += 1;
    }
    let version = u16::from_le_bytes([header[8], header[9]]);
    let record_bytes = u16::from_le_bytes([header[10], header[11]]);
    let count = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
    let source = u32::from_le_bytes([header[28], header[29], header[30], header[31]]);
    let (records, expected_source, length, remainder) = match (version, record_bytes as usize) {
        (1, v1::RECORD_BYTES) => {
            let (records, rest) = body.as_chunks::<{ v1::RECORD_BYTES }>();
            (Records::V1(records), SOURCE_BSC5, records.len(), rest.len())
        }
        (2, v2::RECORD_BYTES) => {
            let (records, rest) = body.as_chunks::<{ v2::RECORD_BYTES }>();
            (
                Records::V2(records),
                SOURCE_BY_RECORD,
                records.len(),
                rest.len(),
            )
        }
        _ => {
            return Err(FormatError::Version {
                version,
                record_bytes,
            });
        }
    };
    if source != expected_source {
        return Err(FormatError::Source { source });
    }
    if remainder != 0 || length != count {
        return Err(FormatError::Length {
            count,
            record_bytes: body.len(),
        });
    }
    let epoch = [
        header[16], header[17], header[18], header[19], header[20], header[21], header[22],
        header[23],
    ];
    Ok(Catalog {
        records,
        reference_epoch_years: f64::from_le_bytes(epoch),
    })
}

/// The CRC-32 (IEEE 802.3) of `bytes`.
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(u32::MAX, |crc, &byte| {
        (0..8).fold(crc ^ u32::from(byte), |crc, _| {
            (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg())
        })
    })
}

#[cfg(test)]
mod tests;
