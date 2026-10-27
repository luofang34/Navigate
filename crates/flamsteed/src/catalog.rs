//! The embedded catalogue and its record layout.
//!
//! The source is CDS catalogue V/50 (Hoffleit and Warren 1991), prepared by
//! the NASA Astronomical Data Center. It is a United States government work
//! and is in the public domain. The 9096 entries with a J2000 position are
//! embedded as `data/bsc5.bin`, which the `bsc5_table` example writes; the
//! example documents the 22-byte record layout.

const BSC5: &[u8] = include_bytes!("../data/bsc5.bin");
pub(crate) const RECORD_BYTES: usize = 22;
const MAS: f64 = std::f64::consts::PI / 648_000_000.0;

/// A set of stars in the 22-byte record layout.
#[derive(Clone, Copy, Debug)]
pub struct Catalog<'a> {
    records: &'a [[u8; RECORD_BYTES]],
}

impl Catalog<'static> {
    /// The Yale Bright Star Catalogue: the 9096 entries with a J2000 position,
    /// about every star to visual magnitude 6.5.
    pub fn bright_star_catalogue() -> Self {
        Self {
            records: BSC5.as_chunks::<RECORD_BYTES>().0,
        }
    }
}

impl Catalog<'_> {
    /// The number of stars.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the catalogue has no stars.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The stars, in catalogue order.
    pub fn stars(&self) -> impl Iterator<Item = Star> + '_ {
        self.records.iter().map(decode)
    }
}

/// One catalogue star at its J2000 position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Star {
    /// Harvard Revised (HR) catalogue number.
    pub hr: u16,
    /// Right ascension at J2000, in radians.
    pub right_ascension_rad: f64,
    /// Declination at J2000, in radians.
    pub declination_rad: f64,
    /// Proper motion in right ascension times cos(declination), in radians
    /// per Julian year.
    pub proper_motion_ra_rad_per_year: f64,
    /// Proper motion in declination, in radians per Julian year.
    pub proper_motion_dec_rad_per_year: f64,
    /// Trigonometric parallax in milliarcseconds, or zero if the catalogue
    /// gives none.
    pub parallax_mas: f64,
    /// Heliocentric radial velocity in km/s, or zero if the catalogue gives
    /// none.
    pub radial_velocity_km_s: f64,
    /// Visual magnitude in the Johnson V band.
    pub v_magnitude: f64,
    /// B-V colour index, if the catalogue gives one.
    pub b_minus_v: Option<f64>,
}

fn decode(record: &[u8; RECORD_BYTES]) -> Star {
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
    Star {
        hr: u16::from_le_bytes([h0, h1]),
        right_ascension_rad: f64::from(u32::from_le_bytes([r0, r1, r2, r3])) / 4_294_967_296.0
            * std::f64::consts::TAU,
        declination_rad: f64::from(i32::from_le_bytes([d0, d1, d2, d3])) / f64::from(i32::MAX)
            * std::f64::consts::FRAC_PI_2,
        proper_motion_ra_rad_per_year: f64::from(i16::from_le_bytes([a0, a1])) * MAS,
        proper_motion_dec_rad_per_year: f64::from(i16::from_le_bytes([p0, p1])) * MAS,
        parallax_mas: f64::from(u16::from_le_bytes([x0, x1])),
        radial_velocity_km_s: f64::from(i16::from_le_bytes([w0, w1])),
        v_magnitude: f64::from(i16::from_le_bytes([v0, v1])) / 100.0,
        b_minus_v: (colour != i16::MIN).then(|| f64::from(colour) / 100.0),
    }
}

#[cfg(test)]
mod tests;
