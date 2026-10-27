//! The embedded catalogue and its file format.
//!
//! The source is CDS catalogue V/50 (Hoffleit and Warren 1991), prepared by
//! the NASA Astronomical Data Center. It is a United States government work
//! and is in the public domain. The 9096 entries with a J2000 position are in
//! `data/bsc5.bin`, which the `bsc5_table` example writes; the example
//! documents format version 1. The build script embeds the magnitude prefix
//! that the features select.

use crate::{Float, Motion, float};

const STARS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stars.bin"));
const MAGIC: &[u8; 8] = b"FLAMSTED";
const FORMAT_VERSION: u16 = 1;
pub(crate) const HEADER_BYTES: usize = 32;
pub(crate) const RECORD_BYTES: usize = 34;
/// Source 1 is the Bright Star Catalogue, whose identifiers are HR numbers.
const SOURCE_BSC5: u32 = 1;
/// The fixed-point scale of a unit-vector component.
const POSITION_SCALE: f64 = 1.0 / i32::MAX as f64;
/// The fixed-point scale of a velocity component, per Julian year.
const VELOCITY_SCALE: f64 = 1e-12;

/// The embedded table, checked when the crate compiles.
const EMBEDDED: Option<Catalog<'static>> = parse(STARS);
const _: () = assert!(EMBEDDED.is_some(), "the embedded star table is malformed");

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
    /// The embedded Yale Bright Star Catalogue. With the default `all-stars`
    /// feature it has all 9096 entries with a J2000 position, about every star
    /// to visual magnitude 6.5. With `magnitude-5` it has the stars to
    /// magnitude 5.0, and without either the stars to magnitude 4.0.
    pub const fn bright_star_catalogue() -> Self {
        match EMBEDDED {
            Some(catalog) => catalog,
            None => Self {
                records: &[],
                reference_epoch_years: 0.0,
            },
        }
    }
}

impl<'a> Catalog<'a> {
    /// The number of stars.
    pub const fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the catalogue has no stars.
    pub const fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The stars at least as bright as visual magnitude `v_magnitude`. The
    /// records are in magnitude order, so this is a prefix of the catalogue.
    pub fn brighter_than(&self, v_magnitude: f64) -> Self {
        let end = self
            .records
            .partition_point(|record| magnitude(record) <= v_magnitude);
        Self {
            records: self.records.get(..end).unwrap_or_default(),
            reference_epoch_years: self.reference_epoch_years,
        }
    }

    /// The stars, brightest first, then by identifier.
    pub fn stars(&self) -> impl ExactSizeIterator<Item = Star> + 'a {
        let epoch = self.reference_epoch_years;
        self.records.iter().map(move |record| decode(record, epoch))
    }

    /// The motions of the stars in the precision `T`, in the order of
    /// [`Catalog::stars`]. The decode converts integers to `T` and does no
    /// other arithmetic, so `f32` motions need no double-precision operation.
    pub fn motions<T: Float>(&self) -> impl ExactSizeIterator<Item = Motion<T>> + 'a {
        let epoch = T::from_f64(self.reference_epoch_years);
        let position_scale = T::from_f64(POSITION_SCALE);
        let velocity_scale = T::from_f64(VELOCITY_SCALE);
        self.records.iter().map(move |record| {
            let (position, velocity) = vectors(record);
            Motion {
                position: position.map(|c| T::from_i32(c) * position_scale),
                velocity: velocity.map(|c| T::from_i32(c) * velocity_scale),
                reference_epoch_years: epoch,
            }
        })
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
    /// The unit direction at the reference epoch, in J2000 equatorial axes.
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

/// The catalogue of a format version 1 file, if the header and the length
/// agree. The build script checks the CRC-32.
const fn parse(bytes: &[u8]) -> Option<Catalog<'_>> {
    let Some((header, body)) = bytes.split_at_checked(HEADER_BYTES) else {
        return None;
    };
    let mut index = 0;
    while index < MAGIC.len() {
        if header[index] != MAGIC[index] {
            return None;
        }
        index += 1;
    }
    let version = u16::from_le_bytes([header[8], header[9]]);
    let record_bytes = u16::from_le_bytes([header[10], header[11]]) as usize;
    let count = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
    let source = u32::from_le_bytes([header[28], header[29], header[30], header[31]]);
    let (records, rest) = body.as_chunks::<RECORD_BYTES>();
    if version != FORMAT_VERSION
        || record_bytes != RECORD_BYTES
        || source != SOURCE_BSC5
        || !rest.is_empty()
        || records.len() != count
    {
        return None;
    }
    let epoch = [
        header[16], header[17], header[18], header[19], header[20], header[21], header[22],
        header[23],
    ];
    Some(Catalog {
        records,
        reference_epoch_years: f64::from_le_bytes(epoch),
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
fn vectors(record: &[u8; RECORD_BYTES]) -> ([i32; 3], [i32; 3]) {
    let &[
        _,
        _,
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        q0,
        q1,
        q2,
        q3,
        q4,
        q5,
        q6,
        q7,
        q8,
        q9,
        q10,
        q11,
        ..,
    ] = record;
    (
        [
            i32::from_le_bytes([p0, p1, p2, p3]),
            i32::from_le_bytes([p4, p5, p6, p7]),
            i32::from_le_bytes([p8, p9, p10, p11]),
        ],
        [
            i32::from_le_bytes([q0, q1, q2, q3]),
            i32::from_le_bytes([q4, q5, q6, q7]),
            i32::from_le_bytes([q8, q9, q10, q11]),
        ],
    )
}

#[inline]
fn decode(record: &[u8; RECORD_BYTES], reference_epoch_years: f64) -> Star {
    let &[h0, h1, .., v0, v1, c0, c1, x0, x1, w0, w1] = record;
    let (position, velocity) = vectors(record);
    let colour = i16::from_le_bytes([c0, c1]);
    let parallax = u16::from_le_bytes([x0, x1]);
    let radial_velocity = i16::from_le_bytes([w0, w1]);
    Star {
        id: StarId::Hr(u16::from_le_bytes([h0, h1])),
        reference_epoch_years,
        position: position.map(|c| f64::from(c) * POSITION_SCALE),
        velocity: velocity.map(|c| f64::from(c) * VELOCITY_SCALE),
        parallax_mas: (parallax != u16::MAX).then(|| f64::from(parallax)),
        radial_velocity_km_s: (radial_velocity != i16::MIN).then(|| f64::from(radial_velocity)),
        v_magnitude: f64::from(i16::from_le_bytes([v0, v1])) / 100.0,
        b_minus_v: (colour != i16::MIN).then(|| f64::from(colour) / 100.0),
    }
}

#[cfg(test)]
mod tests;
